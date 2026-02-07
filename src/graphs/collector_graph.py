"""Collector Subgraph for paper collection workflow.

Converts the original Collector Agent (agents/collector.py) to explicit LangGraph structure.
Each tool call becomes an explicit node with conditional edges.
"""

from typing import TypedDict, List, Optional, Literal
from pathlib import Path
import json

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from logging_config import logger

# Reuse existing tools
from agents.collector import (
    search_by_ddg,
    get_parsed_html,
    get_existing_rounds_from_db,
    enrich_papers_with_details,
)
from rag.retriever import get_rag_client_by_provider
from settings import settings


class CollectorState(TypedDict):
    """State for collector subgraph."""

    # Input
    conference_name: str
    year: int
    round: str  # "unspecified" means auto-discover

    # Intermediate results
    discovered_rounds: List[str]
    search_results: List[dict]
    parsed_paths: List[Path]

    # Final result
    papers_collected: int
    status: str
    message: str


def extract_parameters_node(state: CollectorState) -> CollectorState:
    """Extract conference parameters from the query/state.

    This replaces the implicit parameter extraction in the original Agent.
    """
    # Parameters should already be set in state by coordinator
    conference = state.get("conference_name", "")
    year = state.get("year", 0)
    round_val = state.get("round", "unspecified")

    logger.info(f"Collector: conference={conference}, year={year}, round={round_val}")

    return {
        **state,
        "conference_name": conference,
        "year": year,
        "round": round_val,
    }


def check_existing_rounds_node(
    state: CollectorState,
) -> Command[Literal["discover_rounds", "parse_papers"]]:
    """Check which rounds already exist in database.

    If round is 'unspecified', discover all rounds.
    Otherwise, check if the specific round exists.
    """
    conference = state["conference_name"]
    year = state["year"]
    round_val = state["round"]

    if not conference or not year:
        logger.error("Missing conference name or year")
        return Command(
            goto="__end__",
            update={
                **state,
                "status": "error",
                "message": "Missing conference name or year",
            },
        )

    if round_val == "unspecified":
        # Need to discover rounds
        logger.info(
            f"Round unspecified, discovering available rounds for {conference} {year}"
        )
        return Command(goto="discover_rounds", update=state)
    else:
        # Check if specific round exists
        existing = get_existing_rounds_from_db.invoke(
            {"conference": conference, "year": year}
        )
        if round_val in existing:
            logger.success(f"Round {round_val} already exists, skipping")
            return Command(
                goto="__end__",
                update={
                    **state,
                    "status": "skipped",
                    "message": f"Round {round_val} already exists",
                },
            )
        return Command(
            goto="parse_papers", update={**state, "discovered_rounds": [round_val]}
        )


def discover_rounds_node(state: CollectorState) -> CollectorState:
    """Discover available rounds for the conference.

    Uses DDG search to find conference website and determine rounds.
    """
    conference = state["conference_name"]
    year = state["year"]

    # Search for conference information
    query = f"{conference} {year} accepted papers"
    search_results = search_by_ddg.invoke({"topic": query})

    # TODO: Parse search results to discover rounds
    # For now, assume single round if unspecified
    discovered = ["all"]  # Placeholder

    logger.info(f"Discovered rounds: {discovered}")

    return {
        **state,
        "discovered_rounds": discovered,
        "search_results": search_results if isinstance(search_results, list) else [],
    }


def parse_papers_node(state: CollectorState) -> CollectorState:
    """Parse HTML for each discovered round.

    Calls get_parsed_html for each round and collects JSON paths.
    """
    conference = state["conference_name"]
    year = state["year"]
    rounds = state.get("discovered_rounds", [])

    parsed_paths = []

    for round_val in rounds:
        try:
            # This will search DDG, find URL, parse HTML
            # The tool handles: search -> parse -> save JSON
            result = get_parsed_html.invoke(
                {
                    "url": "",  # Tool will search for URL
                    "conference": conference,
                    "year": year,
                    "round": round_val,
                }
            )

            # Parse result to get JSON path
            if "Json Path:" in result:
                path_str = result.split("Json Path:")[1].strip()
                parsed_paths.append(Path(path_str))
                logger.success(f"Parsed round {round_val}: {path_str}")
            else:
                logger.warning(
                    f"No JSON path in result for round {round_val}: {result}"
                )

        except Exception as e:
            logger.error(f"Failed to parse round {round_val}: {e}")

    return {
        **state,
        "parsed_paths": parsed_paths,
    }


def enrich_papers_node(state: CollectorState) -> CollectorState:
    """Enrich papers with details (PDF URLs, abstracts).

    Calls enrich_papers_with_details for each parsed JSON file.
    """
    conference = state["conference_name"]
    paths = state.get("parsed_paths", [])

    for path in paths:
        try:
            result = enrich_papers_with_details.invoke(
                {
                    "json_path": str(path),
                    "conference": conference,
                }
            )
            logger.info(f"Enriched papers in {path}: {result}")
        except Exception as e:
            logger.error(f"Failed to enrich {path}: {e}")

    return state


def save_to_db_node(state: CollectorState) -> CollectorState:
    """Save collected papers to RAG database.

    Reads JSON files and inserts documents into vector store.
    """
    conference = state["conference_name"]
    year = state["year"]
    paths = state.get("parsed_paths", [])

    rag_client = get_rag_client_by_provider(settings.rag_provider)
    papers_collected = 0

    for path in paths:
        try:
            # Extract round from filename (format: {conference}_{yy}_{round})
            actual_round = state.get("round", "all")
            try:
                parts = path.stem.split("_")
                if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) == 2:
                    actual_round = parts[-1]
            except:
                pass

            with open(path, "r", encoding="utf-8") as f:
                papers = json.load(f)
                for paper in papers:
                    title = paper.get("title", "")
                    abstract = paper.get("abstract", "")
                    url = paper.get("url", "")
                    pdf_url = paper.get("pdf_url", "")

                    rag_client.insert_document(
                        title=title,
                        abstract=abstract,
                        url=url,
                        pdf_url=pdf_url,
                        conference_name=conference,
                        conference_year=year,
                        conference_round=actual_round,
                    )
                    papers_collected += 1

            logger.success(f"Inserted {len(papers)} papers from {path.name}")

        except Exception as e:
            logger.error(f"Failed to save {path} to DB: {e}")

    return {
        **state,
        "papers_collected": papers_collected,
        "status": "success" if papers_collected > 0 else "no_papers",
        "message": f"Collected {papers_collected} papers",
    }


# Build the collector subgraph
builder = StateGraph(CollectorState)

# Add nodes
builder.add_node("extract_params", extract_parameters_node)
builder.add_node("check_existing", check_existing_rounds_node)
builder.add_node("discover_rounds", discover_rounds_node)
builder.add_node("parse_papers", parse_papers_node)
builder.add_node("enrich_papers", enrich_papers_node)
builder.add_node("save_to_db", save_to_db_node)

# Add edges
builder.add_edge(START, "extract_params")
builder.add_edge("extract_params", "check_existing")

# Conditional: discover rounds or go directly to parsing
builder.add_edge("discover_rounds", "parse_papers")

# Linear flow after parsing
builder.add_edge("parse_papers", "enrich_papers")
builder.add_edge("enrich_papers", "save_to_db")
builder.add_edge("save_to_db", END)

# Compile
collector_subgraph = builder.compile()
