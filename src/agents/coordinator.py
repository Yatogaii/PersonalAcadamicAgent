from typing import Any

from graph.pipeline import run_pipeline


def invoke_coordinator(user_input: str, enable_clarification: bool = False) -> Any:
    """Compatibility wrapper around the LangGraph pipeline."""
    state = run_pipeline(user_input)
    return state.get("answer") or state.get("result") or state
