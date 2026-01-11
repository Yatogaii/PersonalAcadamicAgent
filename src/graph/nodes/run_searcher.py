from typing import Any, Dict

from agents.searcher import Searcher
from logging_config import logger
from settings import settings

from graph.subgraphs.searcher import run_searcher_subgraph

from graph.state import GraphState


def run_searcher(state: GraphState) -> Dict[str, Any]:
    task = state.get("task") or {}
    params = task.get("params") or {}
    query = params.get("query") or state.get("user_input") or ""
    query = query.strip()
    if not query:
        return {"result": {"error": "Empty search query."}}

    try:
        searcher = Searcher()
        if settings.enable_agentic_rag:
            hits = run_searcher_subgraph(query, k=searcher.top_k)
            for idx, hit in enumerate(hits, 1):
                hit.setdefault("id", idx)
        else:
            hits = searcher.search(query)
        formatted = searcher.format_hits(hits)
        return {
            "result": {
                "query": query,
                "hits": hits,
                "formatted_context": formatted,
            }
        }
    except Exception as exc:
        logger.error(f"run_searcher failed: {exc}")
        errors = list(state.get("errors", []))
        errors.append(f"run_searcher_error: {exc}")
        return {"result": {"error": f"Search failed: {exc}"}, "errors": errors}
