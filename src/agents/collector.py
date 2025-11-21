from models import init_kimi_k2
from prompts.template import apply_prompt_template
from settings import settings
from utils import get_parsed_content_by_selector
from parser.HTMLSelector import HTMLSelector, to_html_selector
from agents.html_parse_agent import get_html_selector_by_llm
from utils import extract_text_from_message_content, extract_json_from_codeblock
from rag.retriever import get_rag_client_by_provider

from langchain.tools import tool
from langchain.agents import create_agent
from ddgs import DDGS
import logging
import json
from typing import List

from pathlib import Path
import os

# 项目根路径 (基于当前文件位置向上两级: src/agents -> src -> project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SAVED_HTML_DIR: Path = PROJECT_ROOT / "saved_html_content"
HTML_SELECTORS_DIR: Path = PROJECT_ROOT / "html_selectors"
@tool
def get_parsed_html(url: str, conference: str, year: int, round: str) -> str:
    '''
    Parse a webpage, save the parsed content JSON, and return the absolute path.
    Checks if the conference data already exists in the DB before parsing.

    Args:
        url: the URL of the webpage.
        conference: the short name of the conference (acronym).
        year: the year of the conference (e.g. 2025).
        round: the round of the conference (e.g. 'fall', 'cycle1').
    Return:
        Absolute path (string) to saved parsed content JSON file, or a message if skipped.
    '''
    # Check existence
    rag_client = get_rag_client_by_provider(settings.rag_provider)
    if rag_client.check_conference_exists(conference, year, round):
        logging.info(f"Conference {conference} {year} {round} already exists. Skipping.")
        return f"Conference {conference} {year} {round} already exists. Skipping."

    logging.info(f"Getting parsed HTML content for URL: {url}, conference name:{conference}")
    
    # Generate filename: {conference}_{yy}_{round}
    yy = str(year)[-2:]
    filename_base = f"{conference}_{yy}_{round}"
    
    SAVED_HTML_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SAVED_HTML_DIR / f"{filename_base}.json"
    if file_path.exists():
        logging.info(f"Cached parsed HTML content for URL: {url}, conference name:{conference}")
        return str(file_path.resolve())

    selectors_obj = get_or_write_html_selector(url, filename_base)
    selectors_json = selectors_obj.model_dump_json()
    parsed_content = get_parsed_content_by_selector(url, selectors_json)
    file_path.write_text(parsed_content, encoding='utf-8')
    return f"Paper of {conference} collect complete! Json Path: {str(file_path.resolve())}"

@tool
def search_by_ddg(topic: str):
    '''
    Search DuckDuckGo and return a JSON-serializable list of results.

    Args:
        topic: query sent to DuckDuckGo.
    Return:
        List of search result dicts or an error dict.
    '''
    logging.info(f"Searching DuckDuckGo for topic: {topic}")
    try:
        return list(DDGS().text(query=topic, region='us-en', safesearch='Off', time='y', max_results=10))
    except Exception as e:
        return {"error": str(e)}
    
def get_or_write_html_selector(url: str, conference: str) -> HTMLSelector:
    '''
    Parse the html and extract all the text inside specific tags.
    Tags file always in "html_selectors/{conference}.json".

    Args:
        url: the URL of the webpage.
        conference: the name of the conference.
    '''
    # We assume all selectors for one conference are same.
    selector_name = conference.split('_')[0]
    HTML_SELECTORS_DIR.mkdir(parents=True, exist_ok=True)
    selector_file = HTML_SELECTORS_DIR / f"{selector_name}.json"
    if selector_file.exists():
        logging.info(f"Using cached selectors for conference: {selector_name}")
        selectors_json = selector_file.read_text(encoding='utf-8')
        # Validate and parse to HTMLSelector using helper (handles str/dict/instances)
        try:
            selectors = to_html_selector(selectors_json)
            return selectors
        except Exception as e:
            logging.error(f"Invalid selector file: {selector_file}, {e}")
            exit(0)
    else:
        logging.info(f"Generating selectors for conference: {selector_name}")
        selectors = get_html_selector_by_llm(url)

        if not selectors:
            logging.error(f"Failed to get selectors for URL: {url}")
            exit(0)

        # Ensure we have an HTMLSelector instance (LLM may return dict/str)
        try:
            selectors = to_html_selector(selectors)
        except Exception as e:
            logging.error(f"Failed to convert LLM response to HTMLSelector: {e}")
            exit(0)

        # Write to file
        selector_file.write_text(selectors.model_dump_json(), encoding='utf-8')
        return selectors

def _extract_paths_from_final_json(text: str) -> List[Path]:
    """Attempt to parse JSON containing 'parsed_paths'. Return list of Path objects."""
    def try_load(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    payload = try_load(text)
    if payload is None:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            payload = try_load(text[start:end+1])
    if isinstance(payload, dict) and isinstance(payload.get('parsed_paths'), list):
        return [Path(p) for p in payload['parsed_paths']]
    return []

def invoke_collector(conference_name: str, year: int, round: str="all") -> List[Path]:
    """Invoke collector agent and return list of parsed content file paths."""
    collector_agent = create_agent(
        model=init_kimi_k2(),
        tools=[search_by_ddg, get_parsed_html],
    )
    msgs = apply_prompt_template("collector", {
        "conference_name": conference_name,
        "year": year,
        "round": round
    })

    msgs.append({"role": "user", "content": f"""Collect papers for conference {conference_name} {year} {round}."""})
    
    ai_msg = collector_agent.invoke(input={"messages": msgs})
    content_text = extract_text_from_message_content(getattr(ai_msg["messages"][-1], "content", ai_msg["messages"][-1]))
    paths = _extract_paths_from_final_json(extract_json_from_codeblock(content_text))
    if not paths:
        logging.warning(f"No parsed_paths JSON found in agent output. Raw content: {content_text[:500]}")
        raise RuntimeError("Collector agent failed to produce parsed paths.")

    # Insert all papers to RAG.
    rag_client = get_rag_client_by_provider(settings.rag_provider)
    for path in paths:
        if not path.exists():
            logging.warning(f"Parsed content file does not exist: {path}")
            continue

        # Extract round from filename to use the actual found round instead of the requested one
        # Filename format is expected to be: {acronym}_{yy}_{round}
        actual_round = round
        try:
            parts = path.stem.split('_')
            if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) == 2:
                actual_round = parts[-1]
                logging.info(f"Using extracted round '{actual_round}' from file {path.name}")
        except Exception:
            pass

        with open(path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            for paper in papers:
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                url = paper.get('url', '')
                rag_client.insert_document(title=title, abstract=abstract, url=url, conference_name=conference_name, conference_year=year, conference_round=actual_round)
    
    return paths