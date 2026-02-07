"""Searcher Subgraph for RAG search workflow.

Converts the original Searcher Agent (agents/searcher.py) to explicit LangGraph structure.
Implements Agentic RAG with self-reflection loops.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
import json

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from logging_config import logger

# Reuse existing tools
from agents.searcher import (
    analyze_query,
    search_abstracts,
    load_paper_pdfs,
    search_paper_content,
    rerank_results,
    generate_hypothetical_answer,
    evaluate_retrieval_progress,
)


class SearcherState(TypedDict):
    """State for searcher subgraph."""

    # Input
    query: str

    # Analysis results
    query_analysis: Dict[str, Any]
    sub_queries: List[str]
    current_sub_query_index: int
    use_hyde: bool
    hyde_document: str

    # Retrieval state
    retrieval_round: int
    candidate_papers: List[Dict[str, Any]]
    loaded_doc_ids: List[str]
    search_results: List[Dict[str, Any]]

    # Evaluation
    is_sufficient: bool
    coverage_score: float

    # Final result
    reranked_results: List[Dict[str, Any]]
    answer: str
    status: str


def analyze_query_node(state: SearcherState) -> SearcherState:
    """Analyze query and generate retrieval strategy."""
    query = state["query"]

    logger.info(f"Analyzing query: {query}")

    try:
        analysis_json = analyze_query.invoke({"query": query})
        analysis = json.loads(analysis_json)

        state["query_analysis"] = analysis
        state["sub_queries"] = analysis.get("sub_queries", [query])
        state["use_hyde"] = analysis.get("should_use_hyde", False)
        state["current_sub_query_index"] = 0
        state["retrieval_round"] = 1

        logger.info(
            f"Query type: {analysis.get('query_type')}, sub_queries: {len(state['sub_queries'])}"
        )

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        # Fallback
        state["query_analysis"] = {}
        state["sub_queries"] = [query]
        state["use_hyde"] = False
        state["current_sub_query_index"] = 0

    return state


def generate_hyde_node(state: SearcherState) -> Command[Literal["search_abstracts"]]:
    """Generate hypothetical answer if HyDE is enabled."""
    if not state.get("use_hyde"):
        return Command(goto="search_abstracts", update=state)

    query = state["query"]
    logger.info(f"Generating HyDE document for: {query}")

    try:
        hyde_doc = generate_hypothetical_answer.invoke({"query": query})
        state["hyde_document"] = hyde_doc
        logger.info(f"HyDE document generated: {len(hyde_doc)} chars")
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        state["hyde_document"] = ""

    return Command(goto="search_abstracts", update=state)


def search_abstracts_node(state: SearcherState) -> SearcherState:
    """Search abstracts for current sub-query."""
    sub_queries = state.get("sub_queries", [state["query"]])
    current_idx = state.get("current_sub_query_index", 0)

    if current_idx >= len(sub_queries):
        logger.info("All sub-queries processed")
        return state

    current_query = sub_queries[current_idx]
    if state.get("hyde_document"):
        # Use HyDE document for search
        current_query = state["hyde_document"]

    logger.info(
        f"Searching abstracts for sub-query {current_idx + 1}/{len(sub_queries)}: {current_query[:50]}..."
    )

    try:
        results_str = search_abstracts.invoke({"query": current_query, "k": 10})

        # Parse results
        # The tool returns formatted string, we need to extract structured data
        # For now, store the raw results
        if "No papers found" not in results_str:
            # Extract papers from formatted output
            # Format: "[1] Title\n    doc_id: xxx\n    Abstract: ..."
            candidate_papers = state.get("candidate_papers", [])
            # TODO: Parse results_str to extract paper metadata
            # For now, just track that we got results
            logger.info(f"Found abstracts, result length: {len(results_str)}")

        state["current_sub_query_index"] = current_idx + 1

    except Exception as e:
        logger.error(f"Abstract search failed: {e}")

    return state


def check_more_subqueries_node(
    state: SearcherState,
) -> Command[Literal["search_abstracts", "load_pdfs"]]:
    """Check if there are more sub-queries to process."""
    sub_queries = state.get("sub_queries", [])
    current_idx = state.get("current_sub_query_index", 0)

    if current_idx < len(sub_queries):
        logger.info(f"Processing next sub-query ({current_idx + 1}/{len(sub_queries)})")
        return Command(goto="search_abstracts", update=state)
    else:
        logger.info("All sub-queries completed, moving to PDF loading")
        return Command(goto="load_pdfs", update=state)


def load_pdfs_node(state: SearcherState) -> SearcherState:
    """Load PDFs for candidate papers."""
    # In the original Agent, this is done selectively based on search results
    # For now, use the loaded_doc_ids if set, otherwise skip

    doc_ids = state.get("loaded_doc_ids", [])

    if not doc_ids:
        logger.info("No specific doc_ids to load, skipping PDF loading")
        return state

    logger.info(f"Loading {len(doc_ids)} PDFs")

    try:
        result = load_paper_pdfs.invoke({"doc_ids": doc_ids})
        logger.info(f"PDF loading result: {result[:100]}...")
    except Exception as e:
        logger.error(f"PDF loading failed: {e}")

    return state


def search_content_node(state: SearcherState) -> SearcherState:
    """Search within loaded PDFs."""
    query = state["query"]
    doc_ids = state.get("loaded_doc_ids", [])

    logger.info(f"Searching content in papers for: {query}")

    try:
        results_str = search_paper_content.invoke(
            {
                "query": query,
                "doc_ids": doc_ids,
                "k": 5,
            }
        )

        # Store results
        current_results = state.get("search_results", [])
        # TODO: Parse results_str properly
        state["search_results"] = current_results

        logger.info(f"Content search completed, results: {len(results_str)}")

    except Exception as e:
        logger.error(f"Content search failed: {e}")

    return state


def evaluate_progress_node(
    state: SearcherState,
) -> Command[Literal["search_content", "rerank"]]:
    """Evaluate if retrieval is sufficient or needs more rounds."""
    query = state["query"]
    round_num = state.get("retrieval_round", 1)

    # Prepare summary of current results
    results_summary = (
        f"Round {round_num}: {len(state.get('search_results', []))} results"
    )

    logger.info(f"Evaluating retrieval progress (round {round_num})")

    try:
        evaluation_json = evaluate_retrieval_progress.invoke(
            {
                "original_query": query,
                "current_results_summary": results_summary,
                "round_number": round_num,
            }
        )

        evaluation = json.loads(evaluation_json)

        state["is_sufficient"] = evaluation.get("is_sufficient", False)
        state["coverage_score"] = evaluation.get("coverage_score", 0.0)

        should_continue = evaluation.get("should_continue", False)

        logger.info(
            f"Evaluation: sufficient={state['is_sufficient']}, continue={should_continue}"
        )

        if should_continue and round_num < 4:
            state["retrieval_round"] = round_num + 1
            return Command(goto="search_content", update=state)
        else:
            return Command(goto="rerank", update=state)

    except Exception as e:
        logger.error(f"Progress evaluation failed: {e}")
        # Default: proceed to reranking
        return Command(goto="rerank", update=state)


def rerank_node(state: SearcherState) -> SearcherState:
    """Rerank all retrieved results."""
    query = state["query"]
    results = state.get("search_results", [])

    if not results:
        logger.warning("No results to rerank")
        state["reranked_results"] = []
        return state

    logger.info(f"Reranking {len(results)} results")

    try:
        results_json = json.dumps(results[:15], ensure_ascii=False)
        reranked_json = rerank_results.invoke(
            {
                "original_query": query,
                "results_json": results_json,
            }
        )

        reranked = json.loads(reranked_json)
        state["reranked_results"] = reranked

        logger.info(f"Reranking completed: {len(reranked)} papers")

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        state["reranked_results"] = results

    return state


def generate_answer_node(state: SearcherState) -> SearcherState:
    """Generate final answer from reranked results."""
    query = state["query"]
    results = state.get("reranked_results", [])

    logger.info(f"Generating answer from {len(results)} papers")

    # For now, create a simple formatted answer
    # In a full implementation, this would use an LLM to synthesize the answer

    if not results:
        answer = f"No relevant papers found for query: {query}"
    else:
        # Format top results
        answer_parts = [
            f"Based on the search for '{query}', here are the relevant papers:\n"
        ]

        for i, paper in enumerate(results[:5], 1):
            title = paper.get("title", "Untitled")
            abstract = paper.get("abstract", "No abstract")[:200]
            score = paper.get("llm_relevance_score", "N/A")

            answer_parts.append(f"\n{i}. {title} (Relevance: {score})")
            answer_parts.append(f"   {abstract}...")

        answer = "\n".join(answer_parts)

    state["answer"] = answer
    state["status"] = "success"

    logger.info(f"Answer generated: {len(answer)} chars")

    return state


# Build the searcher subgraph
builder = StateGraph(SearcherState)

# Add nodes
builder.add_node("analyze_query", analyze_query_node)
builder.add_node("generate_hyde", generate_hyde_node)
builder.add_node("search_abstracts", search_abstracts_node)
builder.add_node("check_more_subqueries", check_more_subqueries_node)
builder.add_node("load_pdfs", load_pdfs_node)
builder.add_node("search_content", search_content_node)
builder.add_node("evaluate_progress", evaluate_progress_node)
builder.add_node("rerank", rerank_node)
builder.add_node("generate_answer", generate_answer_node)

# Add edges
builder.add_edge(START, "analyze_query")
builder.add_edge("analyze_query", "generate_hyde")

# Conditional: check if more sub-queries
builder.add_conditional_edges(
    "search_abstracts",
    lambda state: "continue"
    if state.get("current_sub_query_index", 0) < len(state.get("sub_queries", []))
    else "done",
    {"continue": "search_abstracts", "done": "load_pdfs"},
)

# Actually, the loop is handled by check_more_subqueries_node
# Let me fix this
builder.add_edge("generate_hyde", "search_abstracts")
builder.add_edge("search_abstracts", "check_more_subqueries")

# Simplified flow for now
builder.add_edge("check_more_subqueries", "load_pdfs")
builder.add_edge("load_pdfs", "search_content")
builder.add_edge("search_content", "evaluate_progress")

# The loop: evaluate_progress can go back to search_content or to rerank
# This is already handled in the node with Command

builder.add_edge("evaluate_progress", "rerank")
builder.add_edge("rerank", "generate_answer")
builder.add_edge("generate_answer", END)

# Compile
searcher_subgraph = builder.compile()
