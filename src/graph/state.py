from typing import Any, Dict, List, TypedDict


class TaskSpec(TypedDict):
    type: str
    params: Dict[str, Any]


class GraphState(TypedDict, total=False):
    user_input: str
    messages: List[Dict[str, str]]
    task: TaskSpec
    result: Any
    answer: str
    errors: List[str]
    progress: List[str]
    meta: Dict[str, Any]
