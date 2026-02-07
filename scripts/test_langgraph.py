"""Test script for LangGraph workflows.

Run this to verify that all graphs are working correctly.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from logging_config import logger, setup_logging

# Load env first
load_dotenv()

# Setup minimal logging
setup_logging(level="INFO")


def test_coordinator_routing():
    """Test coordinator routing logic."""
    logger.info("=" * 60)
    logger.info("Test 1: Coordinator Routing")
    logger.info("=" * 60)

    from graphs import coordinator_graph

    test_cases = [
        ("Collect papers from usenix 2025", "collector"),
        ("Search for AI papers", "searcher"),
        ("Download ieee conference papers", "collector"),
        ("What is transformer architecture?", "searcher"),
    ]

    for query, expected_intent in test_cases:
        try:
            initial_state = {"messages": [{"role": "user", "content": query}]}
            result = coordinator_graph.invoke(initial_state)

            # Check result
            actual_intent = result.get("intent", "unknown")
            status = "✓" if actual_intent == expected_intent else "✗"

            logger.info(
                f"{status} Query: '{query[:40]}...' -> Intent: {actual_intent} (expected: {expected_intent})"
            )

        except Exception as e:
            logger.error(f"✗ Query failed: {query[:40]}... Error: {e}")

    logger.info("")


def test_collector_structure():
    """Test collector graph structure."""
    logger.info("=" * 60)
    logger.info("Test 2: Collector Graph Structure")
    logger.info("=" * 60)

    from graphs import collector_subgraph, CollectorState

    try:
        # Create test state
        test_state: CollectorState = {
            "conference_name": "test_conf",
            "year": 2025,
            "round": "all",
            "discovered_rounds": [],
            "search_results": [],
            "parsed_paths": [],
            "papers_collected": 0,
            "status": "pending",
            "message": "",
        }

        # Get graph info
        logger.info(
            f"Collector subgraph nodes: {list(collector_subgraph.nodes.keys())}"
        )
        logger.info("✓ Collector subgraph compiled successfully")

    except Exception as e:
        logger.error(f"✗ Collector test failed: {e}")

    logger.info("")


def test_searcher_structure():
    """Test searcher graph structure."""
    logger.info("=" * 60)
    logger.info("Test 3: Searcher Graph Structure")
    logger.info("=" * 60)

    from graphs import searcher_subgraph, SearcherState

    try:
        # Create test state
        test_state: SearcherState = {
            "query": "test query",
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

        # Get graph info
        logger.info(f"Searcher subgraph nodes: {list(searcher_subgraph.nodes.keys())}")
        logger.info("✓ Searcher subgraph compiled successfully")

    except Exception as e:
        logger.error(f"✗ Searcher test failed: {e}")

    logger.info("")


def test_visualization():
    """Test graph visualization (if dependencies available)."""
    logger.info("=" * 60)
    logger.info("Test 4: Graph Visualization")
    logger.info("=" * 60)

    try:
        from graphs import coordinator_graph

        # Try to generate mermaid diagram
        mermaid_code = coordinator_graph.get_graph().draw_mermaid()

        # Save to file
        output_file = "coordinator_graph.mmd"
        with open(output_file, "w") as f:
            f.write(mermaid_code)

        logger.info(f"✓ Generated Mermaid diagram: {output_file}")
        logger.info("  View at: https://mermaid.live")

    except Exception as e:
        logger.error(f"✗ Visualization test failed: {e}")

    logger.info("")


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("LangGraph Migration Tests")
    logger.info("=" * 60 + "\n")

    test_coordinator_routing()
    test_collector_structure()
    test_searcher_structure()
    test_visualization()

    logger.info("=" * 60)
    logger.info("Tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
