"""
Utils package for trace logging and selector verification.

This package wraps the original utils.py module and adds new functionality.
When importing from 'utils', you get both the original functions and new ones.
"""

# The trick: Python found utils/ package, but we need utils.py module functions.
# We'll import them by reading the parent utils.py file directly.

import importlib.util
import sys
from pathlib import Path

# Load the utils.py module from parent directory
utils_py_path = Path(__file__).parent.parent / "utils.py"
spec = importlib.util.spec_from_file_location("_utils_module", utils_py_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load utils.py module from: {utils_py_path}")
_utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_utils_module)

# Re-export all public functions from utils.py
get_html = _utils_module.get_html
get_parsed_content_by_selector = _utils_module.get_parsed_content_by_selector
extract_text_from_message_content = _utils_module.extract_text_from_message_content
extract_json_from_codeblock = _utils_module.extract_json_from_codeblock
get_details_from_html = _utils_module.get_details_from_html

# Import from new utils submodules
from .trace_logger import append_jsonl, truncate_text, safe_serialize, COT_INSTRUCTION, TrajectoryCollector
from .selector_verifier import verify_selectors

__all__ = [
    # From original utils.py
    'get_html',
    'get_parsed_content_by_selector',
    'extract_text_from_message_content',
    'extract_json_from_codeblock',
    'get_details_from_html',
    # From new modules
    'append_jsonl',
    'truncate_text', 
    'safe_serialize',
    'COT_INSTRUCTION',
    'TrajectoryCollector',
    'verify_selectors'
]
