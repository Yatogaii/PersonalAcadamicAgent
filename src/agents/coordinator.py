from langchain.agents import create_agent
from models import init_kimi_k2
from prompts.template import apply_prompt_template
from langchain.messages import AIMessage
from langchain.tools import tool

from agents.collector import invoke_collector
from agents.searcher import Searcher
from settings import settings

from utils import extract_text_from_message_content
import json
from logging_config import logger

@tool
def need_clarification():
    """Clarification is currently under development, unavaliable for now."""
    return

@tool
def handoff_to_collector(conference_name: str, year: int, round: str) -> list[str]:
    """
    Param:
        conference_name: The name of the conference to collect papers from.
        year: The year of the conference.
        round: The round of the conference (e.g., "all", "spring", "summer", "fall"). If the conference has only one round, use "all".
    Return:
        The random 10 paper titles and abstract collected.
    """
    logger.success(f"Invoking Collector Agent for conference: {conference_name}, year: {year}, round: {round}")
    res = []

    json_paths = invoke_collector(conference_name, year, round)
    if len(json_paths) == 0:
        return ["No new papers collected, maybe indicating all papers already exist in the database."]
    logger.success(f"Collector Agent finished. Processing collected data from JSON files. Paths: {json_paths}")

    for json_path in json_paths:
        if not json_path.exists():
            raise RuntimeError(f"Collector agent failed, no json file generated for conference {conference_name} {year} {round}.")
        with open(json_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            for paper in papers[:10]:  # Get top 10 papers
                title = paper.get('title', 'No Title')
                abstract = paper.get('abstract', 'No Abstract')
                res.append({"title": title, "abstract": abstract})
    return res 

@tool
def handoff_to_RAG(query: str):
    """
    Param:
        query: The search query string.
    Return:
        Retrieved RAG hits formatted for the coordinator to answer with citations.
    """
    logger.info(f"Invoking RAG Searcher for query: {query}")
    try:
        searcher = Searcher()
        hits = searcher.search(query)
        formatted = searcher.format_hits(hits)
        logger.success(f"RAG search completed with {len(hits)} hits: {formatted}")
        return {
            "query": query,
            "hits": hits,
            "formatted_context": formatted
        }
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return {
            "error": f"RAG search failed: {e}"
        }

# Coordinator Agent
# If need clarification, then return.
# If need search new paper, handoff to paper collector agent.
# If search for topic, just search from the milvus.
def invoke_coordinator(user_input:str, enable_clarification:bool) -> dict:
    res = {}
    main_agent = create_agent(
        #model="google_genai:gemini-flash-lite-latest",
        #model=init_chat_model_from_modelscope("Qwen/Qwen3-VL-235B-A22B-Instruct"),
        model=init_kimi_k2(),
        tools=[handoff_to_collector, handoff_to_RAG, need_clarification],
    )
    
    msgs = apply_prompt_template("coordinator")

    if not enable_clarification:
        msgs.append({"role": "system", "content": "clarification is now DISABLED."})

    msgs.append({"role": "user", "content": user_input})
    
    # 运行 Agent
    result = main_agent.invoke(
        {"messages": msgs}
    )

    messages = result.get("messages", [])

    return messages[-1].content
