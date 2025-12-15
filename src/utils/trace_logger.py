"""
Trace logger utilities for fine-tuning data collection.
"""

import json
import os
from pathlib import Path
from typing import Any, Union
from datetime import datetime


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
