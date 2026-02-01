"""Coordinator Graph for PaperCollector.

This module defines the main orchestration graph using LangGraph 1.0.
It replaces the implicit agent-based routing with explicit graph structure.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from logging_config import logger

# Import state definitions
from graphs.states import CoordinatorState

# Import the node implementations
from graphs.nodes.collector_node import collector_node
from graphs.nodes.searcher_node import searcher_node


def coordinator_router(
    state: CoordinatorState,
) -> Command[Literal["collector", "searcher", END]]:
    """Route user requests to appropriate subgraph based on intent.

    This replaces the implicit tool-calling in the original Agent.
    Uses simple keyword matching for now; can be upgraded to LLM-based routing.

    Args:
        state: Current graph state with message history

    Returns:
        Command with goto target and state updates
    """
    messages = state.get("messages", [])
    if not messages:
        logger.warning("No messages in state, ending graph")
        return Command(
            goto=END, update={"error": "No messages provided", "goto": "end"}
        )

    # Get the last user message
    last_message = messages[-1]
    content = ""

    if hasattr(last_message, "content"):
        content = last_message.content
    elif isinstance(last_message, dict):
        content = last_message.get("content", "")
    else:
        content = str(last_message)

    content_lower = content.lower()

    # Simple intent classification based on keywords
    # Keywords indicating paper collection intent
    collect_keywords = [
        "收集",
        "collect",
        "conference",
        "会议",
        "论文",
        "paper",
        "fetch",
        "download",
        "获取",
        "下载",
        "采集",
    ]

    # Keywords indicating search/RAG intent
    search_keywords = [
        "搜索",
        "search",
        "查询",
        "query",
        "查找",
        "find",
        "rag",
        "检索",
        "retrieve",
        "question",
        "问题",
    ]

    # Check for collection intent
    is_collect = any(kw in content_lower for kw in collect_keywords)
    # Check for search intent
    is_search = any(kw in content_lower for kw in search_keywords)

    logger.info(
        f"Coordinator routing: content='{content[:50]}...', collect={is_collect}, search={is_search}"
    )

    # Route based on detected intent
    if is_collect and not is_search:
        logger.info("Routing to collector subgraph")
        return Command(
            goto="collector",
            update={
                "intent": "collect",
                "original_query": content,
                "goto": "collector",
            },
        )
    elif is_search or not is_collect:
        logger.info("Routing to searcher subgraph")
        return Command(
            goto="searcher",
            update={"intent": "search", "original_query": content, "goto": "searcher"},
        )
    else:
        # Ambiguous or fallback to search
        logger.info("Ambiguous intent, defaulting to searcher")
        return Command(
            goto="searcher",
            update={"intent": "search", "original_query": content, "goto": "searcher"},
        )


# Build the coordinator graph
workflow = StateGraph(CoordinatorState)

# Add nodes
workflow.add_node("coordinator", coordinator_router)
workflow.add_node("collector", collector_node)
workflow.add_node("searcher", searcher_node)

# Add edges
workflow.add_edge(START, "coordinator")

# Conditional routing from coordinator
workflow.add_conditional_edges(
    "coordinator",
    lambda state: state.get("goto", "end"),
    {"collector": "collector", "searcher": "searcher", "end": END},
)

# Both collector and searcher end at END
workflow.add_edge("collector", END)
workflow.add_edge("searcher", END)

# Compile the graph
coordinator_graph = workflow.compile()

# Export for external use
__all__ = ["coordinator_graph", "coordinator_router", "CoordinatorState"]
