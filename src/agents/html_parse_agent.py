"""
HTML Parse Agent:
    Recive an URL, and return the CSS Selector of this url.
    
We assume that all years' conference webpages have the same structure,
so we only need to generate the selectors once for each conference.
"""

from parser.HTMLSelector import HTMLSelector

import os
from langchain.agents import create_agent
import requests
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents.structured_output import ToolStrategy
from bs4 import BeautifulSoup
from bs4.element import NavigableString

from models import init_kimi_k2
from tools.common_tools import get_raw_html_content
from prompts.template import apply_prompt_template
from utils import extract_text_from_message_content, extract_json_from_codeblock

import logging

@tool
def bash_exec(cmd:str) -> str:
    '''
    Execute a bash command.

    Args:
        cmd: the bash command to execute.
    Return:
        the output of the command.
    '''
    logging.info(f"Executing bash command: {cmd}")
    try:
        output = os.popen(cmd).read()
        return output
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def read_file(filename: str, offset: int, chunk_size: int) -> str:
    '''
    Read a file from htmls folder by line numbers.

    Args:
        filename: the name of the file to read.
        offset: the starting line number (0-indexed).
        chunk_size: the number of lines to read.

    Returns:
        The content read from the file (line-based).
    '''
    logging.info(f'Reading file: {filename} from line: {offset} with {chunk_size} lines')
    try:
        with open(f"htmls/{filename}", "r", encoding='utf-8') as f:
            lines = f.readlines()
            # Read chunk_size lines starting from offset
            end_line = min(offset + chunk_size, len(lines))
            return "".join(lines[offset:end_line])
    except Exception as e:
        return f"An error occurred: {e}"

def get_parsed_html_by_llm_summary(url: str):
    agent = create_agent(
        model="google_genai:gemini-flash-latest",
        #model=init_kimi_k2(),
        tools=[get_raw_html_content, read_file],
    )
    msgs = apply_prompt_template("html_parse_agent")
    res = agent.invoke(input={"messages": msgs})
    
    return res['messages'][-1].content

def get_html_selector_by_llm(url: str) -> str:
    agent = create_agent(
        #model="google_genai:gemini-flash-latest",
        model=init_kimi_k2(),
        #response_format=ToolStrategy(HTMLSelector),
        tools=[get_raw_html_content, read_file, bash_exec],
    )
    msgs = apply_prompt_template("html_parse_agent")
    msgs.append({"role": "user", "content": f"Please generate HTML selectors for the webpage: {url}, the saved html file name should be tmp.html."})
    res = agent.invoke(input={"messages": msgs},config={"recursion_limit": 100},)

    return extract_json_from_codeblock(extract_text_from_message_content(getattr(res["messages"][-1], "content", res["messages"][-1])))