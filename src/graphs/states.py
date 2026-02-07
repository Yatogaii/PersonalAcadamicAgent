"""State definitions for graphs.

Centralizes all State TypedDicts to avoid circular imports.
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class CoordinatorState(TypedDict):
    """State for the coordinator graph.

    Extends basic message state with routing information.
    """

    messages: Annotated[list, add_messages]  # Standard message reducer
    intent: str  # "collect" or "search"
    original_query: str  # The user's original query
    goto: str  # Routing target
    error: str  # Error message if any
