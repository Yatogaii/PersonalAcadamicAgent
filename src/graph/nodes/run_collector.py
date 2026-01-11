import json
from pathlib import Path
from typing import Any, Dict, List

from agents.collector import invoke_collector
from logging_config import logger

from graph.state import GraphState


def _build_sample(paths: List[Path]) -> List[Dict[str, str]]:
    sample: List[Dict[str, str]] = []
    for json_path in paths:
        if not json_path.exists():
            continue
        try:
            papers = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"Failed to read collector output {json_path}: {exc}")
            continue
        for paper in papers[:10]:
            sample.append(
                {
                    "title": paper.get("title", "No Title"),
                    "abstract": paper.get("abstract", "No Abstract"),
                }
            )
    return sample


def run_collector(state: GraphState) -> Dict[str, Any]:
    task = state.get("task") or {}
    params = task.get("params") or {}
    conference = (params.get("conference") or "").strip()
    year = params.get("year")
    round_value = (params.get("round") or "unspecified").strip() or "unspecified"

    if not conference or not year:
        return {"result": {"error": "Missing conference or year for collection."}}

    try:
        paths = invoke_collector(conference, int(year), round_value)
        if not paths:
            return {
                "result": {
                    "conference": conference,
                    "year": year,
                    "round": round_value,
                    "paths": [],
                    "message": "No new papers collected; they may already exist in the database.",
                }
            }

        sample = _build_sample(paths)
        return {
            "result": {
                "conference": conference,
                "year": year,
                "round": round_value,
                "paths": [str(p) for p in paths],
                "sample": sample,
            }
        }
    except Exception as exc:
        logger.error(f"run_collector failed: {exc}")
        errors = list(state.get("errors", []))
        errors.append(f"run_collector_error: {exc}")
        return {"result": {"error": f"Collector failed: {exc}"}, "errors": errors}
