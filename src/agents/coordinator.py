from langchain.agents import create_agent
from models import init_kimi_k2
from prompts.template import apply_prompt_template
from langchain.messages import AIMessage
from langchain.tools import tool

from agents.collector import invoke_collector

import json

@tool
def need_clarification():
    return

@tool
def handoff_to_collector(conference_name: str, year: int, round: str="all") -> list[str]:
    """
    Param:
        conference_name: The name of the conference to collect papers from.
        year: The year of the conference.
        round: The round of the conference (e.g., "all", "spring", "summer", "fall"). If the conference has only one round, use "all".
    Return:
        The top 10 paper titles and abstract collected.
    """
    res = []

    json_path = invoke_collector(conference_name, year, round)

    if json_path:
        with open(json_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
            for paper in papers[:10]:  # Get top 10 papers
                title = paper.get('title', 'No Title')
                abstract = paper.get('abstract', 'No Abstract')
                res.append({"title": title, "abstract": abstract})
    else:
        raise RuntimeError("Collector agent failed.")

    return res 
@tool
def handoff_to_searcher():
    """
    Searcher is currently under development, unavaliable for now.
    """
    return

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
        tools=[handoff_to_collector, handoff_to_searcher, need_clarification],
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

    final_answer = "未找到答案"
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final_answer = msg.content
            break

    return res
