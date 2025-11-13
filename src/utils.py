import requests
import logging

from bs4 import BeautifulSoup
from parser.HTMLSelector import HTMLSelector


def get_html(url, filename):
    logging.info(f"Getting HTML content for URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        #print("Getted HTML content:", response.text)  # Print first 500 characters
        with open(f"htmls/{filename}", "w", encoding='utf-8') as f:
            f.write(response.text)
        return True
    except Exception as e:
        logging.error(f"Error getting HTML: {e}")
        return False


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
        logging.info(f"Failed to get HTML content for URL: {url}")
        exit(0)
    
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
            logging.info(f"Found {len(paper_containers)} paper containers")
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
            logging.info("Warning: No paper containers found, falling back to global selection")
            logging.info("This may cause title/abstract mismatch if papers have multiple abstract paragraphs")
            
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