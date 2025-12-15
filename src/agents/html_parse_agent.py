"""
HTML Parse Agent:
    Recive an URL, and return the CSS Selector of this url.
    
We assume that all years' conference webpages have the same structure,
so we only need to generate the selectors once for each conference.
"""

from parser.HTMLSelector import HTMLSelector

import os
import json
import uuid
import traceback
from datetime import datetime
from pathlib import Path
from langchain.agents import create_agent
import requests
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents.structured_output import ToolStrategy
from bs4 import BeautifulSoup
from bs4.element import NavigableString

from models import get_llm_by_usage
from tools.common_tools import get_raw_html_content
from prompts.template import apply_prompt_template
from utils import extract_text_from_message_content, extract_json_from_codeblock
from utils.trace_logger import append_jsonl, truncate_text, safe_serialize
from utils.selector_verifier import verify_selectors

from logging_config import logger

@tool
def bash_exec(cmd:str) -> str:
    '''
    Execute a bash command.

    Args:
        cmd: the bash command to execute.
    Return:
        the output of the command.
    '''
    logger.info(f"Executing bash command: {cmd}")
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
    logger.info(f"Reading file: {filename} from line: {offset} with {chunk_size} lines")
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

@tool
def verify_selectors_tool(html_path: str, selector_json: str) -> str:
    '''
    Verify HTML selectors against an HTML file and return diagnostic scores.
    
    Args:
        html_path: Path to the HTML file (relative to htmls/ folder or absolute).
        selector_json: Selector configuration as JSON string.
        
    Returns:
        JSON string with verification results (ok, score, metrics, diagnostics).
    '''
    logger.info(f"Verifying selectors for HTML: {html_path}")
    
    # Handle relative paths
    if not os.path.isabs(html_path):
        html_path = os.path.join("htmls", html_path)
    
    try:
        result = verify_selectors(html_path, selector_json)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        error_result = {
            "ok": False,
            "score": 0.0,
            "metrics": {},
            "diagnostics": {"error": str(e)}
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)


def get_html_selector_by_llm(url: str) -> str:
    """
    Generate HTML selectors using LLM agent and log trace for fine-tuning.
    
    Args:
        url: The URL to generate selectors for
        
    Returns:
        JSON string of the generated selectors
    """
    trace_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    # Get trace log path from environment or use default
    trace_path = os.environ.get('PARSER_TRACE_PATH', 'logs/parser_traces.jsonl')
    
    # Initialize trace data
    trace_data = {
        "trace_id": trace_id,
        "url": url,
        "start_time": start_time.isoformat(),
        "end_time": None,
        "model_name": "unknown",
        "prompt_template_name": "html_parse_agent",
        "prompt_template_version": "unknown",
        "recursion_limit": 100,
        "tool_calls": [],
        "messages": [],
        "final_selector_json_raw": None,
        "final_selector_json_parsed": None,
        "verify_score": 0.0,
        "verify_diagnostics": {},
        "error": None
    }
    
    try:
        # Get model
        model = get_llm_by_usage('agentic')
        
        # Try to extract model name
        try:
            if hasattr(model, 'model_name'):
                trace_data["model_name"] = str(getattr(model, 'model_name', 'unknown'))
            elif hasattr(model, 'model'):
                trace_data["model_name"] = str(getattr(model, 'model', 'unknown'))
            elif hasattr(model, '__class__'):
                trace_data["model_name"] = model.__class__.__name__
        except Exception:
            pass
        
        # Create agent with verify_selectors_tool
        agent = create_agent(
            model=model,
            tools=[get_raw_html_content, read_file, bash_exec, verify_selectors_tool],
        )
        
        # Apply prompt template
        msgs = apply_prompt_template("html_parse_agent")
        msgs.append({
            "role": "user", 
            "content": f"Please generate HTML selectors for the webpage: {url}, the saved html file name should be tmp.html."
        })
        
        # Invoke agent
        res = agent.invoke(
            input={"messages": msgs},
            config={"recursion_limit": 100}
        )
        
        # Extract tool calls from messages
        step_idx = 0
        last_tool_record = None
        for msg in res.get("messages", []):
            # Try to extract tool calls from message
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_record = {
                        "step_idx": step_idx,
                        "tool_name": getattr(tool_call, 'name', 'unknown'),
                        "tool_args": safe_serialize(getattr(tool_call, 'args', {})),
                        "tool_output": None,
                        "tool_output_len": 0
                    }
                    trace_data["tool_calls"].append(tool_record)
                    last_tool_record = tool_record
                    step_idx += 1
            
            # Look for tool responses in next messages
            if hasattr(msg, 'type') and msg.type == 'tool':
                if trace_data["tool_calls"] and last_tool_record:
                    output = getattr(msg, 'content', '')
                    last_tool_record["tool_output"] = truncate_text(output, max_chars=4000)
                    last_tool_record["tool_output_len"] = len(str(output))
        
        # Serialize messages (truncate content)
        for msg in res.get("messages", []):
            msg_dict = safe_serialize(msg)
            trace_data["messages"].append(msg_dict)
        
        # Extract final selector JSON
        final_content = extract_text_from_message_content(
            getattr(res["messages"][-1], "content", res["messages"][-1])
        )
        final_selector_str = extract_json_from_codeblock(final_content)
        
        trace_data["final_selector_json_raw"] = truncate_text(final_selector_str, max_chars=8000)
        
        # Try to parse the selector JSON
        try:
            selector_dict = json.loads(final_selector_str)
            trace_data["final_selector_json_parsed"] = selector_dict
            
            # Run verification
            html_path = "htmls/tmp.html"
            if os.path.exists(html_path):
                verify_result = verify_selectors(html_path, selector_dict)
                trace_data["verify_score"] = verify_result.get("score", 0.0)
                trace_data["verify_diagnostics"] = verify_result.get("diagnostics", {})
                trace_data["verify_metrics"] = verify_result.get("metrics", {})
                trace_data["verify_ok"] = verify_result.get("ok", False)
            else:
                trace_data["verify_diagnostics"] = {
                    "error": f"HTML file not found: {html_path}"
                }
        except json.JSONDecodeError as e:
            trace_data["final_selector_json_parsed"] = None
            trace_data["verify_diagnostics"] = {
                "parse_error": f"Failed to parse selector JSON: {str(e)}"
            }
        except Exception as e:
            trace_data["verify_diagnostics"] = {
                "verify_error": f"Verification failed: {str(e)}"
            }
        
        # Record end time
        trace_data["end_time"] = datetime.now().isoformat()
        
        # Write trace to JSONL
        append_jsonl(trace_path, trace_data)
        logger.info(f"Trace logged to {trace_path} with trace_id={trace_id}")
        
        return final_selector_str
        
    except Exception as e:
        # Record error
        trace_data["end_time"] = datetime.now().isoformat()
        trace_data["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        
        # Write trace even on error
        append_jsonl(trace_path, trace_data)
        logger.error(f"Error in get_html_selector_by_llm (trace_id={trace_id}): {e}")
        
        # Re-raise the exception
        raise
