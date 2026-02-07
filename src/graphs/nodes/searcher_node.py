"""Searcher Node for LangGraph.

This node invokes the searcher subgraph to execute the full RAG search workflow.
"""

from langgraph.types import Command
from typing import Literal
from logging_config import logger

from graphs.searcher_graph import searcher_subgraph, SearcherState
from graphs.states import CoordinatorState


def searcher_node(state: CoordinatorState) -> Command[Literal["__end__"]]:
    """Execute searcher subgraph to perform RAG search.

    Invokes the Agentic RAG workflow with query analysis, retrieval,
    and self-reflection loops.

    Args:
        state: Graph state with intent and original_query from coordinator

    Returns:
        Command to end with search results
    """
    logger.info("Searcher node invoked")

    # Extract query from state
    query = state.get("original_query", "")

    if not query:
        logger.error("No query provided to searcher")
        error_result = {
            "status": "error",
            "message": "No query provided",
            "query_type": "unknown",
            "papers_found": 0,
            "answer": "Error: No query provided for search",
        }
        return Command(
            goto="__end__",
            update={
                "messages": [{"role": "assistant", "content": error_result["answer"]}],
            },
        )

    logger.info(f"Searcher query: {query}")

    # Prepare initial state for subgraph
    searcher_input: SearcherState = {
        "query": query,
        "query_analysis": {},
        "sub_queries": [],
        "current_sub_query_index": 0,
        "use_hyde": False,
        "hyde_document": "",
        "retrieval_round": 1,
        "candidate_papers": [],
        "loaded_doc_ids": [],
        "search_results": [],
        "is_sufficient": False,
        "coverage_score": 0.0,
        "reranked_results": [],
        "answer": "",
        "status": "pending",
    }

    try:
        # Invoke the searcher subgraph
        result = searcher_subgraph.invoke(searcher_input)

        # Format result for coordinator
        final_result = {
            "status": result.get("status", "unknown"),
            "message": "Search completed successfully"
            if result.get("status") == "success"
            else "Search completed with issues",
            "query": query,
            "query_type": result.get("query_analysis", {}).get("query_type", "unknown"),
            "papers_found": len(result.get("reranked_results", [])),
            "answer": result.get("answer", "No answer generated"),
            "coverage_score": result.get("coverage_score", 0.0),
        }

        logger.info(
            f"Searcher completed: {len(result.get('reranked_results', []))} papers found"
        )

        return Command(
            goto="__end__",
            update={
                "messages": [{"role": "assistant", "content": final_result["answer"]}],
            },
        )

    except Exception as e:
        logger.error(f"Searcher subgraph failed: {e}")
        error_result = {
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "query": query,
            "query_type": "unknown",
            "papers_found": 0,
            "answer": f"Sorry, the search encountered an error: {str(e)}",
        }

        return Command(
            goto="__end__",
            update={
                "messages": [{"role": "assistant", "content": error_result["answer"]}],
            },
        )
