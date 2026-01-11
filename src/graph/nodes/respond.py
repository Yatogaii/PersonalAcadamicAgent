import json
from typing import Any, Dict

from logging_config import logger
from models import get_llm_by_usage
from prompts.template import apply_prompt_template
from utils import extract_text_from_message_content

from graph.state import GraphState


def _fallback_response(state: GraphState) -> str:
    result = state.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return "No response generated."


def respond(state: GraphState) -> Dict[str, Any]:
    prompt_msg = apply_prompt_template("langgraph_responder")
    prompt = prompt_msg[0]["content"]
    payload = {
        "user_input": state.get("user_input"),
        "task": state.get("task"),
        "result": state.get("result"),
        "errors": state.get("errors", []),
    }
    llm = get_llm_by_usage("agentic")

    try:
        response = llm.invoke(
            f"{prompt}\n\nInput:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        content = extract_text_from_message_content(
            response.content if hasattr(response, "content") else response
        )
        return {"answer": content}
    except Exception as exc:
        logger.error(f"Responder failed: {exc}")
        errors = list(state.get("errors", []))
        errors.append(f"respond_error: {exc}")
        return {"answer": _fallback_response(state), "errors": errors}
