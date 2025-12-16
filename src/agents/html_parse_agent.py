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
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast
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
from utils.trace_logger import COT_INSTRUCTION, TrajectoryCollector

from logging_config import logger


def _best_effort_extract_json(text: str):
    """Return (json_str, parsed_obj_or_None) without raising."""
    if not isinstance(text, str):
        text = str(text)

    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return candidate, json.loads(candidate)
        except Exception:
            return candidate, None

    m = re.search(r"```\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return candidate, json.loads(candidate)
        except Exception:
            return candidate, None

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj : end_obj + 1].strip()
        try:
            return candidate, json.loads(candidate)
        except Exception:
            return candidate, None

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = text[start_arr : end_arr + 1].strip()
        try:
            return candidate, json.loads(candidate)
        except Exception:
            return candidate, None

    return "", None


def _langchain_messages_to_openai(messages):
    """Convert LangChain messages to OpenAI Chat format dicts (best-effort)."""
    import uuid as _uuid

    tool_id_map = {}
    out = []

    def new_id():
        return f"call_{_uuid.uuid4().hex}"

    for msg in messages or []:
        msg_type = getattr(msg, 'type', None) or getattr(msg, 'role', None)
        content = getattr(msg, 'content', '')

        if msg_type in ('system', 'SystemMessage'):
            out.append({'role': 'system', 'content': content or ''})
            continue
        if msg_type in ('human', 'user', 'HumanMessage'):
            out.append({'role': 'user', 'content': content or ''})
            continue
        if msg_type in ('ai', 'assistant', 'AIMessage'):
            assistant = {'role': 'assistant', 'content': content or ''}
            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                converted = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_id = tc.get('id')
                        tc_name = tc.get('name') or tc.get('function', {}).get('name')
                        tc_args = tc.get('args') or tc.get('function', {}).get('arguments')
                    else:
                        tc_id = getattr(tc, 'id', None)
                        tc_name = getattr(tc, 'name', None)
                        tc_args = getattr(tc, 'args', None)

                    if not tc_id or not str(tc_id).startswith('call_'):
                        new_tc_id = new_id()
                        if tc_id:
                            tool_id_map[str(tc_id)] = new_tc_id
                        tc_id = new_tc_id

                    if isinstance(tc_args, str):
                        args_str = tc_args
                    else:
                        try:
                            args_str = json.dumps(tc_args or {}, ensure_ascii=False)
                        except Exception:
                            args_str = json.dumps({'_raw': str(tc_args)}, ensure_ascii=False)

                    converted.append({'id': str(tc_id), 'type': 'function', 'function': {'name': str(tc_name or 'unknown'), 'arguments': args_str}})
                assistant['tool_calls'] = converted
            out.append(assistant)
            continue

        if msg_type in ('tool', 'ToolMessage'):
            raw_tool_call_id = getattr(msg, 'tool_call_id', None)
            mapped = tool_id_map.get(str(raw_tool_call_id), str(raw_tool_call_id))
            name = getattr(msg, 'name', None) or getattr(msg, 'tool', None) or getattr(msg, 'tool_name', None)
            out.append({'role': 'tool', 'tool_call_id': mapped, 'name': str(name or 'unknown'), 'content': content or ''})
            continue

        out.append({'role': 'assistant', 'content': str(content)})

    return out

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


def get_html_selector_by_llm(url: str, selector_target: str | None = None) -> str:
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
        
        # Decide which part of the page to generate selectors for.
        # Priority: explicit arg > env var > simple heuristic.
        if selector_target is None:
            selector_target = os.environ.get("SELECTOR_TARGET")
        if not selector_target:
            u = (url or "").lower()
            if any(k in u for k in ["accepted", "papers", "proceedings", "conference"]):
                selector_target = "list"
            else:
                selector_target = "page"

        # Apply prompt template (parameterized)
        msgs = apply_prompt_template("html_parse_agent", {"selector_target": selector_target})

        # Optional GRPO mode: prepend CoT instruction as extra system message (no prompt file changes)
        grpo_enabled = bool(os.environ.get('GRPO_OUTPUT_PATH') or os.environ.get('GRPO_COLLECT'))
        if grpo_enabled:
            if msgs and isinstance(msgs, list) and msgs[0].get('role') == 'system':
                msgs[0]['content'] = COT_INSTRUCTION + "\n\n" + (msgs[0].get('content') or "")
            else:
                msgs = [{"role": "system", "content": COT_INSTRUCTION}] + (msgs or [])

        msgs.append({
            "role": "user", 
            "content": (
                f"Please generate HTML selectors for the webpage: {url}, "
                f"the saved html file name should be tmp.html. "
                f"Target mode: {selector_target}."
            )
        })
        
        # Invoke agent
        # LangChain's typing for invoke() varies across versions; runtime accepts this shape.
        res = cast(Any, agent).invoke(
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

        original_question = (
            f"Please generate HTML selectors for the webpage: {url}, the saved html file name should be tmp.html. "
            f"Target mode: {selector_target}."
        )

        # Extract selector JSON (strict by default; lenient in GRPO mode)
        final_selector_str = ""
        selector_dict = None
        try:
            final_selector_str = extract_json_from_codeblock(final_content)
            selector_dict = json.loads(final_selector_str)
        except Exception as e:
            if grpo_enabled:
                final_selector_str, selector_dict = _best_effort_extract_json(str(final_content))
                trace_data["verify_diagnostics"] = {
                    **(trace_data.get("verify_diagnostics") or {}),
                    "final_json_extract_warning": f"{type(e).__name__}: {e}",
                }
            else:
                raise
        
        trace_data["final_selector_json_raw"] = truncate_text(final_selector_str, max_chars=8000)
        
        # Try to parse the selector JSON
        try:
            trace_data["final_selector_json_parsed"] = selector_dict
            
            # Run verification
            html_path = "htmls/tmp.html"
            if os.path.exists(html_path):
                verify_result = verify_selectors(html_path, selector_dict or {})
                trace_data["verify_score"] = verify_result.get("score", 0.0)
                trace_data["verify_diagnostics"] = verify_result.get("diagnostics", {})
                trace_data["verify_metrics"] = verify_result.get("metrics", {})
                trace_data["verify_ok"] = verify_result.get("ok", False)
            else:
                trace_data["verify_diagnostics"] = {
                    "error": f"HTML file not found: {html_path}"
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

        # Optional GRPO saving (OpenAI chat format)
        if grpo_enabled:
            try:
                system_prompt_used = ""
                for m in msgs:
                    if m.get('role') == 'system':
                        system_prompt_used = m.get('content', '')
                        break
                collector = TrajectoryCollector(system_prompt=system_prompt_used)
                collector.current_trace = _langchain_messages_to_openai(res.get('messages', []))
                collector.save_for_grpo(
                    original_question=original_question,
                    final_selector=(selector_dict if selector_dict is not None else final_selector_str),
                    file_path="htmls/tmp.html",
                )
            except Exception as e:
                logger.warning(f"GRPO save failed (trace_id={trace_id}): {e}")
        
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
