"""CLI: trigger html_parse_agent via get_parser_by_llm to collect GRPO trajectories.

The underlying agent handles CoT injection and TrajectoryCollector logging whenever
GRPO_OUTPUT_PATH/GRPO_COLLECT is set, so the CLI simply prepares the environment and
invokes the shared helper for each URL.

Usage:
    python scripts/grpo_collect_selectors_cli.py --url "https://example.com" --output data/grpo_training_data.jsonl

Notes:
- Output path is controlled via env var GRPO_OUTPUT_PATH to keep TrajectoryCollector
  signature minimal for downstream training pipelines.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _get_logger():
    """Best-effort access to the project's logger; falls back to prints."""

    try:
        from logging_config import logger  # type: ignore

        return logger
    except Exception:
        return None


# Ensure we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from logging_config import setup_logging
except Exception:
    setup_logging = None  # type: ignore

if callable(setup_logging):
    setup_logging()


def _best_effort_extract_json(text: str) -> Tuple[str, Any]:
    """Return (json_str, parsed_obj_or_None) without raising."""

    if not isinstance(text, str):
        text = str(text)

    # 1) Prefer explicit fenced JSON block
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            return candidate, json.loads(candidate)
        except Exception:
            return candidate, None

    # 2) Any fenced code block
    m = re.search(r"```\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return candidate, json.loads(candidate)
        except Exception:
            return candidate, None

    # 3) Fallback: first {...} or [...] span
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj : end_obj + 1].strip()
        try:
            return candidate, json.loads(candidate)
        except Exception:
            return candidate, None

    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidate = text[start_arr : end_arr + 1].strip()
        try:
            return candidate, json.loads(candidate)
        except Exception:
            return candidate, None

    return "", None


def run(url: str, output_path: str) -> int:
    os.environ["GRPO_OUTPUT_PATH"] = output_path

    logger = _get_logger()
    # Always print progress so users see activity regardless of logger setup.
    print(f"[GRPO-CLI] Run url={url}", flush=True)
    if logger:
        logger.info(f"[GRPO-CLI] Run url={url}")

    from agents.html_parse_agent import get_parser_by_llm
    from utils.selector_verifier import verify_selectors

    # Run the parser agent using the shared helper (handles prompts, CoT, logging).
    try:
        final_selector_raw = get_parser_by_llm(url, "list")
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        if logger:
            logger.exception("get_parser_by_llm failed")
        return 2

    selector_dict = None
    if isinstance(final_selector_raw, dict):
        selector_dict = final_selector_raw
        final_selector_str = json.dumps(selector_dict, ensure_ascii=False)
    else:
        final_selector_str = str(final_selector_raw or "")
        if final_selector_str:
            try:
                selector_dict = json.loads(final_selector_str)
            except Exception:
                extracted_str, extracted_obj = _best_effort_extract_json(final_selector_str)
                if extracted_str:
                    final_selector_str = extracted_str
                selector_dict = extracted_obj

    # Verify (best-effort)
    html_path = Path("htmls") / "tmp.html"
    verify_ok = False
    if selector_dict and html_path.exists():
        verify_result = verify_selectors(str(html_path), selector_dict)
        verify_ok = bool(verify_result.get("ok"))

    # Print final selector (if any) and return status code
    if final_selector_str:
        print(final_selector_str)
    else:
        print(json.dumps({"error": "no_selector_extracted"}, ensure_ascii=False))

    return 0 if (selector_dict is not None and verify_ok) else 2


def iter_url_list(path: str) -> Iterable[str]:
    """Yield URLs from a text file (one URL per line).

    Ignores empty lines and lines starting with '#'.
    """

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            yield s


def main() -> None:
    parser = argparse.ArgumentParser(description="Run html_parse_agent and save successful GRPO trajectories.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Target webpage URL")
    group.add_argument(
        "--url-list",
        dest="url_list",
        help="Path to a file containing URLs (one per line)",
    )
    parser.add_argument(
        "--output",
        default="data/grpo_training_data.jsonl",
        help="Output JSONL path (also exported as GRPO_OUTPUT_PATH)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="How many times to run each URL (batch mode)",
    )

    args = parser.parse_args()

    logger = _get_logger()

    if args.url:
        raise SystemExit(run(args.url, args.output))

    # Batch mode
    repeats = max(1, int(args.repeats))
    failures = 0
    total = 0

    urls = list(iter_url_list(args.url_list))
    print(f"[GRPO-CLI] Loaded {len(urls)} urls from {args.url_list}; repeats={repeats}; output={args.output}", flush=True)
    if logger:
        logger.info(f"[GRPO-CLI] Loaded {len(urls)} urls from {args.url_list}; repeats={repeats}; output={args.output}")

    if not urls:
        print(f"[GRPO-CLI] No URLs found in: {args.url_list}", flush=True)
        if logger:
            logger.warning(f"[GRPO-CLI] No URLs found in: {args.url_list}")
        raise SystemExit(0)

    for url_idx, url in enumerate(urls, start=1):
        for rep in range(1, repeats + 1):
            total += 1
            print(f"[GRPO-CLI] ({url_idx}/{len(urls)}) repeat {rep}/{repeats}: {url}", flush=True)
            if logger:
                logger.info(f"[GRPO-CLI] ({url_idx}/{len(urls)}) repeat {rep}/{repeats}: {url}")
            code = run(url, args.output)
            if code != 0:
                failures += 1

    # Exit non-zero if any runs failed
    raise SystemExit(0 if failures == 0 else 2)


if __name__ == "__main__":
    main()
