from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from agents.searcher import (
    analyze_query,
    evaluate_retrieval_progress,
    generate_hypothetical_answer,
    get_context_window,
    load_paper_pdfs,
    rerank_results,
    search_abstracts,
    search_paper_content,
)
from logging_config import logger
from settings import settings

from .searcher_state import SearcherState


def _result_key(result: Dict[str, Any]) -> str:
    doc_id = result.get("doc_id", "")
    chunk_id = result.get("chunk_id")
    if chunk_id is None or chunk_id == -1:
        return f"doc:{doc_id}"
    return f"{doc_id}:{chunk_id}"


def _merge_results(
    existing: List[Dict[str, Any]],
    new_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seen = {_result_key(r) for r in existing}
    merged = list(existing)
    for result in new_results:
        key = _result_key(result)
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
    return merged


def _summarize_results(results: List[Dict[str, Any]], limit: int = 10) -> str:
    if not results:
        return "No results yet."
    lines = []
    for item in results[:limit]:
        title = item.get("title") or "Untitled"
        doc_id = item.get("doc_id") or "N/A"
        lines.append(f"- {title} (doc_id: {doc_id})")
    return "\n".join(lines)


def _select_doc_ids(results: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    doc_ids: List[str] = []
    for item in results:
        doc_id = item.get("doc_id")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
        if len(doc_ids) >= limit:
            break
    return doc_ids


def analyze_node(state: SearcherState) -> Dict[str, Any]:
    analysis = analyze_query(state["original_query"])
    sub_queries = analysis.get("sub_queries") or [state["original_query"]]
    max_rounds = min(len(sub_queries), state.get("max_rounds") or 3)
    current_query = sub_queries[0]
    should_use_hyde = bool(analysis.get("should_use_hyde"))
    use_deep_search = analysis.get("estimated_complexity") == "high"

    return {
        "analysis": analysis,
        "sub_queries": sub_queries,
        "current_round": 0,
        "current_query": current_query,
        "search_query": current_query,
        "should_use_hyde": should_use_hyde,
        "max_rounds": max_rounds,
        "use_deep_search": use_deep_search,
        "all_results": [],
    }


def maybe_hyde_node(state: SearcherState) -> Dict[str, Any]:
    if state.get("should_use_hyde"):
        hyde_doc = generate_hypothetical_answer(state.get("current_query", ""))
        return {"hyde_document": hyde_doc, "search_query": hyde_doc}
    return {"search_query": state.get("current_query", "")}


def search_abstracts_node(state: SearcherState) -> Dict[str, Any]:
    query = state.get("search_query") or state.get("original_query")
    k = state.get("top_k") or settings.milvus_top_k
    results = search_abstracts(query, k=k)
    merged = _merge_results(state.get("all_results", []), results)
    return {"round_results": results, "all_results": merged}


def maybe_load_pdfs_node(state: SearcherState) -> Dict[str, Any]:
    if not state.get("use_deep_search"):
        return {}
    if state.get("current_round", 0) > 0:
        return {}

    doc_ids = _select_doc_ids(state.get("round_results", []))
    if not doc_ids:
        return {}

    report = load_paper_pdfs(doc_ids)
    loaded_doc_ids = [
        doc_id
        for doc_id, info in report.get("results", {}).items()
        if info.get("status") in {"success", "already_exists"}
    ]
    return {"pdf_load_report": report, "loaded_doc_ids": loaded_doc_ids}


def search_paper_content_node(state: SearcherState) -> Dict[str, Any]:
    doc_ids = state.get("loaded_doc_ids", [])
    if not doc_ids:
        return {}

    k = state.get("top_k") or settings.milvus_top_k
    results = search_paper_content(state["original_query"], doc_ids=doc_ids, k=k)
    merged = _merge_results(state.get("all_results", []), results)
    return {"deep_results": results, "all_results": merged}


def maybe_expand_context_node(state: SearcherState) -> Dict[str, Any]:
    if not state.get("deep_results"):
        return {}
    top = state["deep_results"][0]
    doc_id = top.get("doc_id")
    chunk_id = top.get("chunk_id")
    if not doc_id or chunk_id is None or chunk_id == -1:
        return {}
    context = get_context_window(doc_id, chunk_id, window=1)
    top["context_window"] = context
    return {"deep_results": state["deep_results"]}


def maybe_rerank_node(state: SearcherState) -> Dict[str, Any]:
    results = state.get("all_results", [])
    if not results:
        return {}
    reranked = rerank_results(state["original_query"], results)
    return {"all_results": reranked or results}


def decide_continue_node(state: SearcherState) -> Dict[str, Any]:
    current_round = state.get("current_round", 0)
    summary = _summarize_results(state.get("all_results", []))
    evaluation = evaluate_retrieval_progress(
        state["original_query"], summary, current_round + 1
    )
    sub_queries = state.get("sub_queries", [])
    max_rounds = state.get("max_rounds", len(sub_queries))
    next_round = current_round + 1

    should_continue = bool(evaluation.get("should_continue"))
    should_continue = (
        should_continue and next_round < max_rounds and next_round < len(sub_queries)
    )

    if not should_continue:
        return {"evaluation": evaluation, "should_continue": False}

    next_query = sub_queries[next_round]
    logger.info(
        "Searcher continuing to round %d with query: %s", next_round + 1, next_query
    )
    return {
        "evaluation": evaluation,
        "should_continue": True,
        "current_round": next_round,
        "current_query": next_query,
        "search_query": next_query,
    }


def build_searcher_subgraph():
    builder = StateGraph(SearcherState)
    builder.add_node("analyze", analyze_node)
    builder.add_node("maybe_hyde", maybe_hyde_node)
    builder.add_node("search_abstracts", search_abstracts_node)
    builder.add_node("load_pdfs", maybe_load_pdfs_node)
    builder.add_node("search_content", search_paper_content_node)
    builder.add_node("expand_context", maybe_expand_context_node)
    builder.add_node("rerank", maybe_rerank_node)
    builder.add_node("decide_continue", decide_continue_node)

    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "maybe_hyde")
    builder.add_edge("maybe_hyde", "search_abstracts")
    builder.add_edge("search_abstracts", "load_pdfs")
    builder.add_edge("load_pdfs", "search_content")
    builder.add_edge("search_content", "expand_context")
    builder.add_edge("expand_context", "rerank")
    builder.add_edge("rerank", "decide_continue")
    builder.add_conditional_edges(
        "decide_continue",
        lambda state: "continue" if state.get("should_continue") else "stop",
        {"continue": "maybe_hyde", "stop": END},
    )

    return builder.compile()


_SEARCHER_GRAPH = None


def get_searcher_graph():
    global _SEARCHER_GRAPH
    if _SEARCHER_GRAPH is None:
        _SEARCHER_GRAPH = build_searcher_subgraph()
    return _SEARCHER_GRAPH


def run_searcher_subgraph(
    query: str,
    k: int | None = None,
    max_rounds: int | None = None,
) -> List[Dict[str, Any]]:
    graph = get_searcher_graph()
    init_state: SearcherState = {
        "original_query": query,
        "top_k": k or settings.milvus_top_k,
        "max_rounds": max_rounds or 3,
        "all_results": [],
    }
    final_state = graph.invoke(init_state, config={"recursion_limit": 50})
    return final_state.get("all_results", [])
