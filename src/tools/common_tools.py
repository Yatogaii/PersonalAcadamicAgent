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
