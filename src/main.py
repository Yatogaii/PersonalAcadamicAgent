import os
from agents.coordinator import invoke_coordinator
from prompts.template import apply_prompt_template

from dotenv import load_dotenv

import logging
import pprint

try:
    load_dotenv()
except Exception as e:
    exit(0)
    
logging.basicConfig(level=logging.INFO)

logging.debug(f"KIMI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
    
enable_clarification = False
MAX_CLARIFICATION_ROUNDS = 3

def workflow(user_input:str):
    # Parse user-input
    user_input = user_input.strip()

    # Invoke Coordinator Agent with clarification
    is_clarification_complete = False
    while not is_clarification_complete and enable_clarification:
        final_state = invoke_coordinator(user_input, enable_clarification)

        # Summary current clarification Messages
        if final_state["need_clarification"] and final_state["clarification_times"]> MAX_CLARIFICATION_ROUNDS:
            pass

    # Get an summary for all clarification rounds.
    # clarification_complete will be True iff clarification is enabled and is completed.
    if is_clarification_complete:
        pass

    res = invoke_coordinator(user_input, False)
    pprint.pprint(res)


if __name__ == "__main__":
    user_input = 'Show me all accepted paper in USENIX Security 2024 Summer'
    workflow(user_input)

