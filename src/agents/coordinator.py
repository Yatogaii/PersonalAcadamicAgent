from langchain.agents import create_agent
from models import init_kimi_k2
from tools.common_tools import search_by_ddg, get_parsed_html
from prompts.template import apply_prompt_template
from langchain.messages import AIMessage
from langchain.tools import tool

@tool
def need_clarification():
    return

@tool
def handoff_to_collector():
    return

@tool
def handoff_to_searcher():
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
