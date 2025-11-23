from models import init_kimi_k2
from prompts.template import apply_prompt_template
from settings import settings
from utils import get_parsed_content_by_selector, get_details_from_html
from parser.HTMLSelector import HTMLSelector, to_html_selector
from agents.html_parse_agent import get_html_selector_by_llm
from utils import extract_text_from_message_content, extract_json_from_codeblock
from rag.retriever import get_rag_client_by_provider

from langchain.tools import tool
from langchain.agents import create_agent
from ddgs import DDGS
import json
from typing import List

from pathlib import Path
import os
from logging_config import logger

# 项目根路径 (基于当前文件位置向上两级: src/agents -> src -> project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SAVED_HTML_DIR: Path = PROJECT_ROOT / "saved_html_content"
HTML_SELECTORS_DIR: Path = PROJECT_ROOT / "html_selectors"

@tool
def get_existing_rounds_from_db(conference: str, year: int) -> List[str]:
    """
    Check which rounds of a conference are already in the database.
    
    Args:
        conference: The conference acronym (e.g. 'usenix').
        year: The year (e.g. 2025).
    Return:
        List of round names that exist (e.g. ['fall', 'summer']).
    """
    rag_client = get_rag_client_by_provider(settings.rag_provider)
    rounds = rag_client.get_existing_rounds(conference, year)
    logger.success(f"Checked existing rounds for {conference} {year}: {rounds}")
    return rounds

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
        logger.success(f"Conference {conference} {year} {round} already exists. Skipping.")
        return f"Conference {conference} {year} {round} already exists. Skipping."

    logger.info(f"Getting parsed HTML content for URL: {url}, conference name:{conference}, name:{year}, round:{round}")
    
    # Generate filename: {conference}_{yy}_{round}
    yy = str(year)[-2:]
    filename_base = f"{conference}_{yy}_{round}"
    
    SAVED_HTML_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SAVED_HTML_DIR / f"{filename_base}.json"
    if file_path.exists():
        logger.success(f"Cached parsed HTML content for URL: {url}, conference name:{conference}")
        return str(file_path.resolve())

    selectors_obj = get_or_write_html_selector(url, filename_base)
    selectors_json = selectors_obj.model_dump_json()
    parsed_content = get_parsed_content_by_selector(url, selectors_json)
    if parsed_content == "":
        logger.error(f"Failed to parse content for URL: {url}")
        return f"Failed to parse content for URL: {url}, maybe unfinished round yet."
    file_path.write_text(parsed_content, encoding='utf-8')
    logger.success(f"Parsed content saved to {file_path.resolve()}")
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
    logger.info(f"Searching DuckDuckGo for topic: {topic}")
    try:
        return list(DDGS().text(query=topic, region='us-en', safesearch='Off', time='y', max_results=10))
    except Exception as e:
        return {"error": str(e)}

@tool
def enrich_papers_with_details(json_path: str, conference: str) -> str:
    """
    For each paper in the JSON file, visit its detail URL (if present).
    Extract the PDF link.
    If the abstract is missing, extract the abstract from the detail page.
    Update the JSON file.
    
    Args:
        json_path: Path to the JSON file containing papers.
        conference: Conference name (used for caching selectors).
    """
    path = Path(json_path)
    if not path.exists():
        return f"File not found: {json_path}"
        
    try:
        papers = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        return f"Failed to load JSON: {e}"
        
    updated_count = 0
    detail_selector = None
    
    logger.info(f"Enriching papers in {json_path} with details (PDF, Abstract)")
    
    for paper in papers:
        detail_url = paper.get('url')
        if not detail_url:
            continue
            
        if not detail_selector:
            try:
                detail_selector = get_or_write_html_selector(detail_url, conference, "detail")
            except Exception as e:
                logger.error(f"Failed to get detail selector: {e}")
                continue
        
        try:
            details = get_details_from_html(detail_url, detail_selector)
            
            changed = False
            if 'pdf_url' in details:
                paper['pdf_url'] = details['pdf_url']
                changed = True
                logger.info(f"Found PDF link for {paper.get('title', 'unknown')}: {details['pdf_url']}")
            
            if not paper.get('abstract') and 'abstract' in details:
                paper['abstract'] = details['abstract']
                changed = True
                logger.info(f"Found missing abstract for {paper.get('title', 'unknown')}")
                
            if changed:
                updated_count += 1
                
        except Exception as e:
            logger.error(f"Error getting details for {detail_url}: {e}")
            
    path.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.success(f"Updated {updated_count} papers with details.")
    return f"Updated {updated_count} papers with details."

@tool
def report_progress(message: str) -> str:
    """
    Emit a progress log for debugging/visibility.

    Args:
        message: Any short status summary to log.
    Return:
        Echoes the message back for the transcript.
    """
    logger.success(f"[CollectorProgress] {message}")
    return message
    
def get_or_write_html_selector(url: str, conference: str, selector_suffix: str = "") -> HTMLSelector:
    '''
    Parse the html and extract all the text inside specific tags.
    Tags file always in "html_selectors/{conference}.json".

    Args:
        url: the URL of the webpage.
        conference: the name of the conference.
        selector_suffix: suffix for the selector filename (e.g. "detail").
    '''
    # We assume all selectors for one conference are same.
    selector_name = conference.split('_')[0]
    if selector_suffix:
        selector_name = f"{selector_name}_{selector_suffix}"
    
    HTML_SELECTORS_DIR.mkdir(parents=True, exist_ok=True)
    selector_file = HTML_SELECTORS_DIR / f"{selector_name}.json"
    if selector_file.exists():
        logger.success(f"Using cached selectors for conference: {selector_name}")
        selectors_json = selector_file.read_text(encoding='utf-8')
        # Validate and parse to HTMLSelector using helper (handles str/dict/instances)
        try:
            selectors = to_html_selector(selectors_json)
            return selectors
        except Exception as e:
            logger.error(f"Invalid selector file: {selector_file}, {e}")
            exit(0)
    else:
        logger.info(f"Generating selectors for conference: {selector_name}")
        selectors = get_html_selector_by_llm(url)

        if not selectors:
            logger.error(f"Failed to get selectors for URL: {url}")
            exit(0)

        # Ensure we have an HTMLSelector instance (LLM may return dict/str)
        try:
            selectors = to_html_selector(selectors)
        except Exception as e:
            logger.error(f"Failed to convert LLM response to HTMLSelector: {e}")
            exit(0)

        # Write to file
        selector_file.write_text(selectors.model_dump_json(), encoding='utf-8')
        logger.success(f"Generated selectors for conference: {selector_name}")
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

def invoke_collector(conference_name: str, year: int, round: str="unspecified") -> List[Path]:
    """
    Invoke collector agent and return list of parsed content file paths.
    round="unspecified" means the agent should discover available rounds itself.
    """
    collector_agent = create_agent(
        model=init_kimi_k2(),
        tools=[search_by_ddg, get_parsed_html, get_existing_rounds_from_db, report_progress, enrich_papers_with_details],
    )
    msgs = apply_prompt_template("collector", {
        "conference_name": conference_name,
        "year": year,
        "round": round
    })

    msgs.append({"role": "user", "content": f"""Collect papers for conference {conference_name} {year} {round}."""})
    
    ai_msg = collector_agent.invoke(input={"messages": msgs},config={"recursion_limit": 100},)
    content_text = extract_text_from_message_content(getattr(ai_msg["messages"][-1], "content", ai_msg["messages"][-1]))
    paths = _extract_paths_from_final_json(extract_json_from_codeblock(content_text))
    if not paths:
        logger.warning(f"No parsed_paths JSON found in agent output. Raw content: {content_text[:500]}")
        return []

    # Insert all papers to RAG.
    rag_client = get_rag_client_by_provider(settings.rag_provider)
    for path in paths:
        if not path.exists():
            logger.warning(f"Parsed content file does not exist: {path}")
            continue

        # Extract round from filename to use the actual found round instead of the requested one
        # Filename format is expected to be: {acronym}_{yy}_{round}
        actual_round = round
        try:
            parts = path.stem.split('_')
            if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) == 2:
                actual_round = parts[-1]
                logger.success(f"Using extracted round '{actual_round}' from file {path.name}")
        except Exception:
            pass

        with open(path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            for paper in papers:
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                url = paper.get('url', '')
                pdf_url = paper.get('pdf_url', '')
                final_url = pdf_url if pdf_url else url
                rag_client.insert_document(title=title, abstract=abstract, url=final_url, conference_name=conference_name, conference_year=year, conference_round=actual_round)
    
    return paths
