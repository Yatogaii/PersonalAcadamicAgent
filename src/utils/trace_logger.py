"""
Trace logger utilities for fine-tuning data collection.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


# -----------------------------------------------------------------------------
# System prompt addon: force Chain-of-Thought (CoT) before tool calls
# -----------------------------------------------------------------------------

COT_INSTRUCTION = """You MUST follow these rules strictly:

1) Before calling any tool / function, you MUST output a <thought>...</thought> block FIRST.
     - The <thought> content MUST include:
         (a) analysis of the current Observation / context
         (b) the next search strategy (what you will search / grep / find, and why)
     - Only AFTER the </thought> closing tag may you issue a tool call.

2) If you do not need a tool call, still output a <thought>...</thought> block describing your analysis and plan.

3) Final answer format:
     - The original instructions require the final output to be JSON-only.
     - Because you now output <thought>, you MUST put your FINAL JSON result inside a fenced code block:
         ```json
         { ... }
         ```
     - Outside the final ```json code block, do NOT include any extra text (except the <thought> block earlier).
"""


def append_jsonl(path: Union[str, Path], obj: dict) -> None:
    """
    Append a JSON object as a single line to a JSONL file.
    
    Args:
        path: Path to the JSONL file
        obj: Dictionary to append
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def truncate_text(text: Any, max_chars: int = 4000) -> str:
    """
    Truncate text to a maximum number of characters.
    
    Args:
        text: Text to truncate (will be converted to string)
        max_chars: Maximum number of characters
        
    Returns:
        Truncated text with ellipsis if truncated
    """
    if text is None:
        return ""
    
    text_str = str(text)
    if len(text_str) <= max_chars:
        return text_str
    
    return text_str[:max_chars] + "... [truncated]"


def safe_serialize(obj: Any, max_depth: int = 10, max_str_len: int = 4000) -> Any:
    """
    Safely serialize an object to JSON-compatible format.
    Handles common non-serializable types and truncates long strings.
    
    Args:
        obj: Object to serialize
        max_depth: Maximum recursion depth
        max_str_len: Maximum string length before truncation
        
    Returns:
        JSON-serializable version of the object
    """
    if max_depth <= 0:
        return "<max_depth_reached>"
    
    # Handle None
    if obj is None:
        return None
    
    # Handle primitives
    if isinstance(obj, (bool, int, float)):
        return obj
    
    # Handle strings
    if isinstance(obj, str):
        return truncate_text(obj, max_str_len)
    
    # Handle datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Handle bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')[:max_str_len]
        except UnicodeDecodeError:
            return f"<bytes:{len(obj)}>"
    
    # Handle dict-like objects
    if isinstance(obj, dict):
        return {
            str(k): safe_serialize(v, max_depth - 1, max_str_len)
            for k, v in obj.items()
        }
    
    # Handle list-like objects
    if isinstance(obj, (list, tuple)):
        return [
            safe_serialize(item, max_depth - 1, max_str_len)
            for item in obj
        ]
    
    # Handle objects with __dict__
    if hasattr(obj, '__dict__'):
        return safe_serialize(obj.__dict__, max_depth - 1, max_str_len)
    
    # Handle objects with dict() method (like Pydantic models)
    if hasattr(obj, 'dict'):
        try:
            return safe_serialize(obj.dict(), max_depth - 1, max_str_len)
        except Exception:
            pass
    
    # Handle objects with model_dump method (Pydantic v2)
    if hasattr(obj, 'model_dump'):
        try:
            return safe_serialize(obj.model_dump(), max_depth - 1, max_str_len)
        except Exception:
            pass
    
    # Fallback to string representation
    return truncate_text(str(obj), max_str_len)


# -----------------------------------------------------------------------------
# TrajectoryCollector: OpenAI Chat-format traces for GRPO data generation
# -----------------------------------------------------------------------------


class TrajectoryCollector:
    """Collects agent trajectories in OpenAI Chat format for GRPO training.

    Stored messages follow OpenAI Chat schema:
    - system/user/assistant: {role, content}
    - assistant with tool calls: adds {tool_calls: [...]}
    - tool responses: {role: "tool", tool_call_id, name, content}
    """

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt or ""
        self.current_trace: List[Dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.current_trace = []
        if self.system_prompt:
            self.current_trace.append({"role": "system", "content": self.system_prompt})

    def get_trace(self) -> List[Dict[str, Any]]:
        return self.current_trace

    def add_user_message(self, content: str) -> None:
        self.current_trace.append({"role": "user", "content": content or ""})

    def add_assistant_message(self, content: str) -> None:
        self.current_trace.append({"role": "assistant", "content": content or ""})

    def _new_tool_call_id(self) -> str:
        # Matches common OpenAI-style ids used in examples/tests.
        return f"call_{uuid.uuid4().hex}"  # 32 hex chars

    def log_agent_step(
        self,
        thought_content: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
    ) -> str:
        """Log an assistant step containing CoT (and optionally a tool call).

        Args:
            thought_content: Model thought content (should include <thought>...</thought>)
            tool_name: Optional tool/function name
            tool_args: Optional tool/function arguments dict

        Returns:
            tool_call_id if a tool call is recorded, otherwise "".
        """

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": thought_content or "",
        }

        if tool_name:
            tool_call_id = self._new_tool_call_id()
            tool_args = tool_args or {}
            assistant_msg["tool_calls"] = [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": str(tool_name),
                        "arguments": json.dumps(safe_serialize(tool_args), ensure_ascii=False),
                    },
                }
            ]
            self.current_trace.append(assistant_msg)
            return tool_call_id

        # Thought-only
        self.current_trace.append(assistant_msg)
        return ""

    def add_tool_response(self, tool_call_id: str, tool_name: str, content: str) -> None:
        """Add a tool response message linked to a prior assistant tool call."""

        self.current_trace.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": str(tool_name),
                "content": content or "",
            }
        )

    def save_for_grpo(self, original_question: str, final_selector: Union[str, dict], file_path: str) -> None:
        """Save a successful trajectory as one JSONL line for GRPO.

        The saved structure separates model input prompt and verification ground truth.
        """

        output_path = os.environ.get("GRPO_OUTPUT_PATH", "data/grpo_training_data.jsonl")

        # Normalize selector to dict when possible
        target_selector: Any = final_selector
        if isinstance(final_selector, str):
            try:
                target_selector = json.loads(final_selector)
            except Exception:
                # keep as string
                target_selector = final_selector

        # prompt: system + the original question (even if trace has more turns)
        system_content = ""
        if self.current_trace and self.current_trace[0].get("role") == "system":
            system_content = self.current_trace[0].get("content", "")
        else:
            system_content = self.system_prompt

        prompt_msgs = [
            {"role": "system", "content": system_content or ""},
            {"role": "user", "content": original_question or ""},
        ]

        has_cot = any(
            (m.get("role") == "assistant")
            and isinstance(m.get("content"), str)
            and ("<thought>" in m.get("content", ""))
            for m in self.current_trace
        )

        record = {
            "prompt": prompt_msgs,
            "messages": self.current_trace,
            "ground_truth": {
                "target_selector": target_selector,
                "file_context": file_path,
            },
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "trace_length": len(self.current_trace),
                "has_cot": bool(has_cot),
            },
        }

        append_jsonl(output_path, record)
