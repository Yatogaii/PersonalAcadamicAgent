"""Collector Node for LangGraph.

This node invokes the collector subgraph to execute the full paper collection workflow.
"""

from langgraph.graph import MessagesState
from langgraph.types import Command
from typing import Literal
from logging_config import logger

from graphs.collector_graph import collector_subgraph, CollectorState


def collector_node(state: MessagesState) -> Command[Literal["__end__"]]:
    """Execute collector subgraph to collect papers from conference.

    Extracts parameters from the original query and invokes the collector workflow.

    Args:
        state: Graph state with intent and original_query from coordinator

    Returns:
        Command to end with collection results
    """
    logger.info("Collector node invoked")

    # Extract parameters from state
    query = state.get("original_query", "")

    # TODO: Better parameter extraction (could use LLM to parse query)
    # For now, use simple heuristic or require explicit parameters
    # Parse "conference year round" pattern from query

    # Default values
    conference = ""
    year = 0
    round_val = "unspecified"

    # Try to extract from query (simple parsing)
    words = query.split()
    for i, word in enumerate(words):
        # Look for conference name (common ones)
        if word.lower() in ["usenix", "ieee", "acm", "neurips", "icml", "iclr"]:
            conference = word.lower()
        # Look for year (4-digit number)
        if word.isdigit() and len(word) == 4:
            year = int(word)
        # Look for round keywords
        if word.lower() in ["spring", "summer", "fall", "winter", "all"]:
            round_val = word.lower()

    logger.info(
        f"Collector parameters: conference={conference}, year={year}, round={round_val}"
    )

    # Prepare initial state for subgraph
    collector_state: CollectorState = {
        "conference_name": conference,
        "year": year,
        "round": round_val,
        "discovered_rounds": [],
        "search_results": [],
        "parsed_paths": [],
        "papers_collected": 0,
        "status": "pending",
        "message": "",
    }

    try:
        # Invoke the collector subgraph
        result = collector_subgraph.invoke(collector_state)

        # Format result for coordinator
        final_result = {
            "status": result.get("status", "unknown"),
            "message": result.get("message", "No message"),
            "conference": result.get("conference_name", conference),
            "year": result.get("year", year),
            "round": result.get("round", round_val),
            "papers_collected": result.get("papers_collected", 0),
            "parsed_paths": [str(p) for p in result.get("parsed_paths", [])],
        }

        logger.info(f"Collector completed: {final_result['message']}")

        return Command(
            goto="__end__",
            update={
                "messages": [{"role": "assistant", "content": final_result["message"]}],
                "collector_result": final_result,
            },
        )

    except Exception as e:
        logger.error(f"Collector subgraph failed: {e}")
        error_result = {
            "status": "error",
            "message": f"Collection failed: {str(e)}",
            "conference": conference,
            "year": year,
            "round": round_val,
            "papers_collected": 0,
        }

        return Command(
            goto="__end__",
            update={
                "messages": [{"role": "assistant", "content": error_result["message"]}],
                "collector_result": error_result,
            },
        )
