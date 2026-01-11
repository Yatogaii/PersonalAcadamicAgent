import json
from typing import Any, Dict

from logging_config import logger
from models import get_llm_by_usage
from prompts.template import apply_prompt_template
from utils import extract_json_from_codeblock, extract_text_from_message_content

from graph.state import GraphState, TaskSpec


def _normalize_task(raw: Dict[str, Any]) -> TaskSpec:
    task_type = str(raw.get("type") or "CLARIFY").upper()
    params = raw.get("params")
    if not isinstance(params, dict):
        params = {}

    if task_type == "SEARCH":
        query = params.get("query") or raw.get("query") or ""
        params = {"query": str(query).strip()}
    elif task_type == "COLLECT":
        conference = (
            params.get("conference")
            or params.get("conference_name")
            or raw.get("conference")
            or raw.get("conference_name")
            or ""
        )
        year = params.get("year") or raw.get("year")
        round_value = params.get("round") or raw.get("round") or "unspecified"
        try:
            year = int(year) if year is not None else None
        except Exception:
            year = None
        params = {
            "conference": str(conference).strip().lower(),
            "year": year,
            "round": str(round_value).strip().lower() if round_value else "unspecified",
        }
    elif task_type == "CLARIFY":
        message = params.get("message") or raw.get("message") or "Please clarify your request."
        params = {"message": str(message).strip()}

    return {"type": task_type, "params": params}


def route_task(state: GraphState) -> Dict[str, Any]:
    user_input = (state.get("user_input") or "").strip()
    if not user_input:
        return {"task": {"type": "CLARIFY", "params": {"message": "Please provide a query or collection request."}}}

    prompt_msg = apply_prompt_template("langgraph_router", {"user_input": user_input})
    prompt = prompt_msg[0]["content"]
    llm = get_llm_by_usage("agentic")

    try:
        response = llm.invoke(prompt)
        content = extract_text_from_message_content(
            response.content if hasattr(response, "content") else response
        )
        payload = json.loads(extract_json_from_codeblock(content))
        task = _normalize_task(payload)
        return {"task": task}
    except Exception as exc:
        logger.error(f"Router failed to parse task: {exc}")
        errors = list(state.get("errors", []))
        errors.append(f"router_error: {exc}")
        return {
            "task": {"type": "CLARIFY", "params": {"message": "I couldn't parse the request. Can you clarify?"}},
            "errors": errors,
        }
