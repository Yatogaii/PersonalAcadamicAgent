from models import init_kimi_k2
from prompts.template import apply_prompt_template
from utils import get_parsed_content_by_selector
from parser.HTMLSelector import HTMLSelector
from html_parse_agent import get_html_selector_by_llm

from langchain.tools import tool
from langchain.agents import create_agent
from ddgs import DDGS
import logging

from pathlib import Path
import os

# 项目根路径 (基于当前文件位置向上两级: src/agents -> src -> project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SAVED_HTML_DIR: Path = PROJECT_ROOT / "saved_html_content"
HTML_SELECTORS_DIR: Path = PROJECT_ROOT / "html_selectors"
@tool
def get_parsed_html(url: str, conference: str):
    '''
    Get the HTML parsed content of a webpage.

    Args:
        url: the URL of the webpage.
        conference: the short name of the conference, will be the filename.
    Return:
        the HTML content of the webpage.
    '''
    SAVED_HTML_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SAVED_HTML_DIR / f"{conference}.json"
    if file_path.exists():
        logging.debug(f"!!!!!Cached parsed HTML content for URL: {url}, conference name:{conference}")
        return file_path.read_text(encoding='utf-8')
    
    selectors_obj = get_or_write_html_selector(url, conference)
    selectors_json = selectors_obj.model_dump_json()
    parsep_content = get_parsed_content_by_selector(url, selectors_json)
    file_path.write_text(parsep_content, encoding='utf-8')
    return parsep_content

@tool
def search_by_ddg(topic: str):
    '''
    Search by DuckDuckGo.

    Args:
        topic: the query that will send to DuckDuckGo.
    Return:
        the json format of search results.
    '''
    logging.info(f"!!!!!Searching DuckDuckGo for topic: {topic}")
    try:
        search = DDGS().text(query=topic, region='us-en', safesearch='Off', time='y', max_results=10)
        return search
    except Exception as e:
        return f"An error occurred: {e}"
    
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
        print(f"!!!!!Using cached selectors for conference: {selector_name}")
        selectors_json = selector_file.read_text(encoding='utf-8')
        # Validate and parse to HTMLSelector
        try:
            return HTMLSelector.model_validate_json(selectors_json)
        except Exception as e:
            logging.error(f"!!!!!Invalid selector file: {selector_file}, {e}")
            exit(0)
    else:
        logging.info(f"!!!!!Generating selectors for conference: {selector_name}")
        selectors : HTMLSelector = get_html_selector_by_llm(url)
        
        if not selectors:
            logging.error(f"!!!!!Failed to get selectors for URL: {url}")
            exit(0)
        
        # Write to file
        selector_file.write_text(selectors.model_dump_json(), encoding='utf-8')
        return selectors

def invoke_collector(conference_name: str, year: int, round: str="all") -> bool:
    """
    Param:
        conference_name: The name of the conference to collect papers from.
        year: The year of the conference.
        round: The round of the conference (e.g., "all", "spring", "summer", "fall"). If the conference has only one round, use "all".
    Return:
        Whether this collect succeed.
    """
    # Here would be the logic to invoke the collector agent.
    # For now, we just return True to indicate success.
    collector_agent = create_agent(
        model=init_kimi_k2(),
        tools=[search_by_ddg, get_parsed_html],  # Add relevant tools for the collector agent
    )
    
    msgs = apply_prompt_template("collector", {
        "conference_name": conference_name,
        "year": year,
        "round": round
    })

    res = collector_agent.invoke(input={"messages": msgs})
    return True