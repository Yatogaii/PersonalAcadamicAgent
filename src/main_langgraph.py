"""LangGraph-based entry point for PaperCollector.

This is an alternative to main.py that uses LangGraph for workflow orchestration
instead of LangChain Agents.

Usage:
    python main_langgraph.py

Features:
    - Explicit graph-based workflow (nodes + edges)
    - Type-safe state management
    - Observable execution flow
    - Same tools and models as original
"""

import os
import sys
import json
import pprint

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
from logging_config import logger, setup_logging, PLAIN_LOG_FORMAT
from graphs import coordinator_graph
from settings import settings


def format_result(result: dict) -> str:
    """Format graph result for display."""
    if not result:
        return "No result"

    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = str(last_message)
        return content

    # Check for collector/searcher results
    collector_result = result.get("collector_result")
    searcher_result = result.get("searcher_result")

    if collector_result:
        return f"[Collector] {collector_result.get('message', 'No message')}"

    if searcher_result:
        return f"[Searcher] {searcher_result.get('answer', 'No answer')}"

    return pprint.pformat(result, width=100)


def run_workflow(user_input: str) -> str:
    """Run the LangGraph workflow for a user query.

    Args:
        user_input: User's query string

    Returns:
        Formatted result string
    """
    logger.info(f"Running LangGraph workflow for: {user_input}")

    try:
        # Prepare initial state
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
        }

        # Invoke the graph
        result = coordinator_graph.invoke(initial_state)

        # Format and return result
        formatted = format_result(result)
        logger.info(f"Workflow completed successfully")
        return formatted

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return f"Error: {str(e)}"


def run_interactive():
    """Run interactive CLI mode."""
    print("=" * 60)
    print("PaperCollector - LangGraph Mode")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            print("\nProcessing...")
            result = run_workflow(user_input)
            print(f"\nAssistant: {result}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Setup logging
    setup_logging(
        level="INFO",
        log_format=PLAIN_LOG_FORMAT,
    )

    # Check command line arguments
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        result = run_workflow(query)
        print(result)
    else:
        # Interactive mode
        run_interactive()


if __name__ == "__main__":
    main()
