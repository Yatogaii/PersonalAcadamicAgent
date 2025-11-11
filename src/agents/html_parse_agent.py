import os
from langchain.agents import create_agent
import requests
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents.structured_output import ToolStrategy
from bs4 import BeautifulSoup
from bs4.element import NavigableString

from models import init_kimi_k2
from tools.common_tools import HTMLSelector, get_raw_html_content
from prompts.template import apply_prompt_template

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
    logging.debug("!!!!!Executing bash command:", cmd)
    try:
        output = os.popen(cmd).read()
        return output
    except Exception as e:
        return f"An error occurred: {e}"

@tool
def read_file(filename: str, offset: int, chunk_size: int) -> str:
    '''
    Read a file from raw_htmls folder.

    Args:
        filename: the name of the file to read.
        offset: the offset to start reading from.
        chunk_size: the size of the chunk to read.

    Returns:
        The content read from the file.
    '''
    print('!!! Reading file:', filename, 'from offset:', offset, 'with chunk size:', chunk_size)
    try:
        with open(f"raw_htmls/{filename}", "r") as f:
            f.seek(offset)
            return f.read(chunk_size)
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

def get_html_selector_by_llm(url: str):
    agent = create_agent(
        #model="google_genai:gemini-flash-latest",
        model=init_kimi_k2(),
        #tools=[get_raw_html_content, read_file, bash_exec, generate_html_tree_view],
        response_format=ToolStrategy(HTMLSelector),
        tools=[get_raw_html_content, read_file, bash_exec],
    )
    msgs = apply_prompt_template("html_parse_agent")
    res = agent.invoke({"messages": msgs})

    return res['structured_response']