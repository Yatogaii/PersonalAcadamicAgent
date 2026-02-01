"""Collector Node for LangGraph.

This node wraps the paper collection logic from the original Collector Agent.
Will be fully implemented in Phase 3.
"""

from langgraph.graph import MessagesState
from langgraph.types import Command
from typing import Literal
from logging_config import logger


def collector_node(state: MessagesState) -> Command[Literal["__end__"]]:
    """Placeholder collector node.

    TODO: Implement full collector logic from agents/collector.py
    - Search DDG for conference URLs
    - Parse HTML to extract paper metadata
    - Enrich papers with details
    - Save to JSON and insert into RAG

    Args:
        state: Graph state with intent and original_query

    Returns:
        Command to end the subgraph
    """
    logger.info("Collector node invoked (placeholder)")

    # Extract parameters from state
    intent = state.get("intent", "unknown")
    query = state.get("original_query", "")

    # TODO: Parse conference_name, year, round from query
    # For now, return a placeholder response
    result = {
        "status": "placeholder",
        "message": f"Collector would process: {query}",
        "conference": "TBD",
        "year": "TBD",
        "round": "TBD",
        "papers_collected": 0,
    }

    return Command(
        goto="__end__",
        update={
            "messages": [{"role": "assistant", "content": str(result)}],
            "collector_result": result,
        },
    )
