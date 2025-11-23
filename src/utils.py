import requests
from logging_config import logger

from bs4 import BeautifulSoup
from parser.HTMLSelector import HTMLSelector


def get_html(url, filename):
    logger.info(f"Getting HTML content for URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        #print("Getted HTML content:", response.text)  # Print first 500 characters
        with open(f"htmls/{filename}", "w", encoding='utf-8') as f:
            f.write(response.text)
        logger.success(f"Saved HTML content for {url} to htmls/{filename}")
        return True
    except Exception as e:
        logger.error(f"Error getting HTML: {e}")
        return False


import json
import re
from urllib.parse import urljoin

def get_parsed_content_by_selector(url: str, selectors: str):
    '''
    Parse the html and extract all the text inside specific tags.

    Args:
        url: the URL of the webpage.
        selectors: the selectors in json format.
    '''
    tmp_file = "temp_html.html"
    if not get_html(url, tmp_file):
        logger.warning(f"Failed to get HTML content for URL: {url}")
        return ""
    
    # Read the HTML file
    with open(f'htmls/{tmp_file}', 'r', encoding='utf-8') as f:
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
            logger.success(f"Found {len(paper_containers)} paper containers")
            for paper_elem in paper_containers:
                # Extract title from this paper
                title_elem = paper_elem.select_one(selector_dict['title'])
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                # Extract all abstract paragraphs from this paper and join them
                abstract_elems = paper_elem.select(selector_dict['abstract'])
                abstract = ' '.join([elem.get_text(strip=True) for elem in abstract_elems]) if abstract_elems else ""
                
                # Extract link
                link = ""
                if 'link' in selector_dict and selector_dict['link']:
                    link_elem = paper_elem.select_one(selector_dict['link'])
                    if link_elem and link_elem.has_attr('href'):
                        href = link_elem['href']
                        if isinstance(href, list):
                            href = href[0]
                        link = urljoin(url, href)

                if title or abstract:  # Only add if we found something
                    papers.append({
                        'title': title,
                        'abstract': abstract,
                        'url': link
                    })
        else:
            # Fallback: if no paper containers found, try global selection
            logger.warning("No paper containers found, falling back to global selection")
            logger.warning("This may cause title/abstract mismatch if papers have multiple abstract paragraphs")
            
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
                    
                    # Try to find links if selector exists
                    link = ""
                    if 'link' in selector_dict and selector_dict['link']:
                        # This is tricky for global selection. We assume links match titles count.
                        link_elems = soup.select(selector_dict['link'])
                        if i < len(link_elems) and link_elems[i].has_attr('href'):
                            href = link_elems[i]['href']
                            if isinstance(href, list):
                                href = href[0]
                            link = urljoin(url, href)

                    papers.append({
                        'title': title,
                        'abstract': abstract,
                        'url': link
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

def extract_text_from_message_content(content) -> str:
    """Normalize AIMessage.content (may be str or list of blocks) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts).strip()
    return str(content)


def extract_json_from_codeblock(s: str) -> str:
    """
    Given a string `s`, return a JSON string in one of two ways:

    1. If `s` itself is valid JSON (json.loads succeeds), return the original stripped string.
    2. Otherwise, search for the first fenced code block labelled ```json
       and return the inner content (stripped).

    Raises:
        TypeError: if `s` is not a string.
        ValueError: if neither (1) nor (2) can be satisfied.

    This is useful when an LLM reply contains both narrative and a JSON code block
    and you want to extract the JSON fragment for further parsing.
    """
    if not isinstance(s, str):
        raise TypeError("extract_json_from_codeblock expects a string")

    candidate = s.strip()
    # Try parsing the whole string as JSON first
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        pass

    # Find explicit ```json ... ``` fenced code block (case-insensitive)
    m = re.search(r"```\s*json\s*\n(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback: find any fenced code block and try to strip a leading 'json' token
    m2 = re.search(r"```\s*\n?(.*?)```", s, flags=re.DOTALL)
    if m2:
        inner = m2.group(1).strip()
        # If the inner block starts with a language token like 'json' remove it
        if inner.lower().startswith('json'):
            # remove the first line that contains 'json'
            parts = inner.split('\n', 1)
            inner = parts[1].strip() if len(parts) > 1 else ''
        if inner:
            return inner

    raise ValueError('No valid JSON found and no ```json``` code block present')

def get_details_from_html(url: str, selector: HTMLSelector) -> dict:
    '''
    Parse the html and extract the PDF link and abstract.
    '''
    tmp_file = "temp_detail.html"
    if not get_html(url, tmp_file):
        return {}
    
    with open(f'htmls/{tmp_file}', 'r', encoding='utf-8') as f:
        html_content = f.read()
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    result = {}
    
    # Extract PDF link
    if selector.pdf_link:
        link_elem = soup.select_one(selector.pdf_link)
        if link_elem and link_elem.has_attr('href'):
            href = link_elem['href']
            if isinstance(href, list):
                href = href[0]
            result['pdf_url'] = urljoin(url, href)

    # Extract Abstract
    if selector.abstract:
        abstract_elems = soup.select(selector.abstract)
        abstract = ' '.join([elem.get_text(strip=True) for elem in abstract_elems]) if abstract_elems else ""
        if abstract:
            result['abstract'] = abstract
            
    return result
