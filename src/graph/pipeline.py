from typing import Any, Dict

from langgraph.graph import END, StateGraph

from graph.state import GraphState
from graph.nodes.router import route_task
from graph.nodes.run_collector import run_collector
from graph.nodes.run_searcher import run_searcher
from graph.nodes.respond import respond


def _route_from_task(state: GraphState) -> str:
    task = state.get("task") or {}
    task_type = str(task.get("type") or "CLARIFY").upper()
    if task_type in {"SEARCH", "COLLECT", "CLARIFY", "OTHER"}:
        return task_type
    return "CLARIFY"


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("router", route_task)
    builder.add_node("run_searcher", run_searcher)
    builder.add_node("run_collector", run_collector)
    builder.add_node("respond", respond)

    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        _route_from_task,
        {
            "SEARCH": "run_searcher",
            "COLLECT": "run_collector",
            "CLARIFY": "respond",
            "OTHER": "respond",
        },
    )
    builder.add_edge("run_searcher", "respond")
    builder.add_edge("run_collector", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_pipeline(user_input: str, config: Dict[str, Any] | None = None) -> GraphState:
    graph = get_graph()
    init_state: GraphState = {
        "user_input": user_input,
        "errors": [],
        "progress": [],
    }
    return graph.invoke(init_state, config=config or {})
