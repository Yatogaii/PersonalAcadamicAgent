import subprocess
import html
from bs4 import BeautifulSoup
from langchain.tools import tool
import requests
from ddgs import DDGS
from utils import get_html
from pydantic import BaseModel, Field
from agents.html_parse_agent import get_html_selector_by_llm

import os

from coding_agent import HTMLParserAgent
import logging

class HTMLSelector(BaseModel):
    title: str = Field(description="Css Selector of the paper title")
    abstract: str = Field(description="Css Selector of the paper abstract")

@tool
def get_raw_html_content(url: str, filename: str) -> bool:
    '''
    Get the HTML content of a webpage.
    And write the content to a file with the parameter `filename` inside raw_htmls folder.

    Args:
        url: the URL of the webpage.
        filename: the name of the file to save the HTML content.
    Return:
        Success of Failed.
    '''
    logging.debug("!!!!!Getting raw HTML content for URL:", url, "and saving to file:", filename)
    return get_html(url, filename)

@tool
def search_by_ddg(topic: str):
    '''
    Search by DuckDuckGo.

    Args:
        topic: the query that will send to DuckDuckGo.
    Return:
        the json format of search results.
    '''
    print("!!!!!Searching DuckDuckGo for topic:", topic)
    try:
        search = DDGS().text(query=topic, region='us-en', safesearch='Off', time='y', max_results=10)
        return search
    except Exception as e:
        return f"An error occurred: {e}"

def get_or_write_html_parser(url: str, conference: str):
    '''
    Parse the html and extract all the text inside <p> tags.

    Args:
        url: the URL of the webpage.
        conference: the name of the conference.
    '''
    pass

import json

def get_parsed_content_by_selector(url: str, selectors: str):
    '''
    Parse the html and extract all the text inside specific tags.

    Args:
        url: the URL of the webpage.
        selectors: the selectors in json format.
    '''
    tmp_file = "temp_html.html"
    if not get_html(url, tmp_file):
        logging.debug("!!!!!Failed to get HTML content for URL:", url)
        exit(0)
    
    # Read the HTML file
    with open(f'raw_htmls/{tmp_file}', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Parse selectors from JSON string
    try:
        selector_dict = json.loads(selectors)
    except json.JSONDecodeError:
        return f"Invalid JSON format for selectors: {selectors}"
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Convert to list of paper objects
    papers = []
    
    # Check if we have title and abstract selectors
    if 'title' in selector_dict and 'abstract' in selector_dict:
        # Find all paper containers (assuming they're in article.node-paper)
        paper_containers = soup.select('article.node-paper')
        
        if paper_containers:
            # Extract from each paper container separately
            print(f"!!!!!Found {len(paper_containers)} paper containers")
            for paper_elem in paper_containers:
                # Extract title from this paper
                title_elem = paper_elem.select_one(selector_dict['title'])
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                # Extract all abstract paragraphs from this paper and join them
                abstract_elems = paper_elem.select(selector_dict['abstract'])
                abstract = ' '.join([elem.get_text(strip=True) for elem in abstract_elems]) if abstract_elems else ""
                
                if title or abstract:  # Only add if we found something
                    papers.append({
                        'title': title,
                        'abstract': abstract
                    })
        else:
            # Fallback: if no paper containers found, try global selection
            print("!!!!!Warning: No paper containers found, falling back to global selection")
            print("!!!!!This may cause title/abstract mismatch if papers have multiple abstract paragraphs")
            
            titles = [elem.get_text(strip=True) for elem in soup.select(selector_dict['title'])]
            abstract_elems = soup.select(selector_dict['abstract'])
            
            # Group abstracts by trying to match count
            if len(titles) > 0:
                abstracts_per_paper = len(abstract_elems) // len(titles) if len(titles) > 0 else 1
                abstracts_per_paper = max(1, abstracts_per_paper)
                
                for i, title in enumerate(titles):
                    start_idx = i * abstracts_per_paper
                    end_idx = start_idx + abstracts_per_paper
                    abstract_texts = [elem.get_text(strip=True) for elem in abstract_elems[start_idx:end_idx]]
                    abstract = ' '.join(abstract_texts)
                    
                    papers.append({
                        'title': title,
                        'abstract': abstract
                    })
        
        return json.dumps(papers, ensure_ascii=False, indent=2)
    else:
        # Fallback: extract content based on selectors without paper structure
        extracted_content = {}
        for key, css_selector in selector_dict.items():
            try:
                elements = soup.select(css_selector)
                texts = [elem.get_text(strip=True) for elem in elements]
                extracted_content[key] = texts if texts else []
            except Exception as e:
                extracted_content[key] = f"Error selecting '{css_selector}': {str(e)}"
        
        return json.dumps(extracted_content, ensure_ascii=False, indent=2)

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
    selector_file = f"html_selectors/{selector_name}.json"
    if os.path.exists(selector_file):
        print(f"!!!!!Using cached selectors for conference: {selector_name}")
        with open(selector_file, 'r', encoding='utf-8') as f:
            selectors_json = f.read()
        # Validate and parse to HTMLSelector
        try:
            return HTMLSelector.model_validate_json(selectors_json)
        except Exception as e:
            print(f"!!!!!Invalid selector file: {selector_file}, {e}")
            exit(0)
    else:
        print(f"!!!!!Generating selectors for conference: {selector_name}")
        selectors : HTMLSelector = get_html_selector_by_llm(url)
        
        if not selectors:
            print("!!!!!Failed to get selectors for URL:", url)
            exit(0)
        
        # Write to file
        with open(selector_file, 'w', encoding='utf-8') as f:
            f.write(selectors.model_dump_json())
        return selectors


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
    if os.path.exists(f'saved_html_content/{conference}.json'):
        print(f"!!!!!Cached parsed HTML content for URL: {url}, conference name:{conference}")
        with open(f'saved_html_content/{conference}.json', 'r', encoding='utf-8') as f:
            return f.read()
    
    selectors_obj = get_or_write_html_selector(url, conference)
    selectors_json = selectors_obj.model_dump_json()
    parsep_content = get_parsed_content_by_selector(url, selectors_json)
    with open(f'saved_html_content/{conference}.json', 'w', encoding='utf-8') as f:
        f.write(parsep_content)
    return parsep_content