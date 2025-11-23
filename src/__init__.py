"""
Top-level package initializer for `src` to ensure absolute imports like
`from src.settings import settings` work when running scripts/tests from the
project root.

This file is intentionally small and only exists to make `src` a proper
python package (PEP 420 namespace packages aside). If you prefer not to add
this file, run tests with PYTHONPATH set (see README for examples):

    export PYTHONPATH="$PWD:$PYTHONPATH"
    python tests/test_milvus_search.py

"""

__all__ = []
