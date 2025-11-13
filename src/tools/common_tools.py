from langchain.tools import tool
from utils import get_html

import logging


@tool
def get_raw_html_content(url: str, filename: str) -> bool:
    '''
    Get the HTML content of a webpage.
    And write the content to a file with the parameter `filename` inside htmls folder.

    Args:
        url: the URL of the webpage.
        filename: the name of the file to save the HTML content.
    Return:
        Success of Failed.
    '''
    logging.info(f"Getting raw HTML content for URL: {url} and saving to file: {filename}")
    return get_html(url, filename)
