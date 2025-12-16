"""Selector verification without ground truth.

This module provides a lightweight heuristic verifier for CSS selector JSON.
It is used by `src/agents/html_parse_agent.py` and the GRPO CLI.

Return shape:
{
  "ok": bool,
  "score": float,
  "metrics": {...},
  "diagnostics": {...}
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup


ITEM_SELECTOR_KEYS = {
    "item_selector",
    "paper_item_selector",
    "paper_selector",
    "list_item_selector",
    "papers",
    "paper_list",
}


def _load_selector_json(selector_json: Union[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[str]]:
    if isinstance(selector_json, dict):
        return selector_json, None
    if isinstance(selector_json, str):
        try:
            return json.loads(selector_json), None
        except Exception as e:
            return {}, f"selector_json_parse_error: {e}"
    return {}, "selector_json_invalid_type"


def _as_path(html_path: str) -> Path:
    p = Path(html_path)
    if p.is_absolute():
        return p
    # If caller already passed htmls/..., keep it; else prefer htmls/<name>
    if str(p).startswith("htmls" + os.sep) or str(p) == "htmls":
        return p
    return Path("htmls") / p


def _css_from_element(elem) -> str:
    """Best-effort CSS selector for an element."""
    if not getattr(elem, "name", None):
        return "body"

    classes = elem.get("class") if hasattr(elem, "get") else None
    if classes:
        # Prefer the first class to keep selector compact
        cls = classes[0]
        if cls:
            return f".{cls}"
    elem_id = elem.get("id") if hasattr(elem, "get") else None
    if elem_id:
        return f"#{elem_id}"
    return str(elem.name)


def _auto_detect_item_selector(soup: BeautifulSoup, title_selector: Optional[str]) -> Tuple[str, List[str]]:
    """Auto-detect an item selector using title selector parents.

    Returns: (selected_item_selector, candidates)
    """

    candidates: List[str] = []

    if title_selector:
        title_elems = soup.select(title_selector)
        if title_elems:
            # Count parent candidates up to 3 levels
            counts: Dict[str, int] = {}
            for t in title_elems[:200]:
                parent = getattr(t, "parent", None)
                for _ in range(3):
                    if not parent or not getattr(parent, "name", None):
                        break
                    sel = _css_from_element(parent)
                    counts[sel] = counts.get(sel, 0) + 1
                    parent = getattr(parent, "parent", None)

            # Keep reasonable candidates
            scored: List[Tuple[float, str, int]] = []
            n_titles = len(title_elems)
            for sel, cnt in counts.items():
                if 5 <= cnt <= 500:
                    # Prefer counts close to number of titles
                    closeness = 1.0 - (abs(cnt - n_titles) / max(n_titles, 1))
                    scored.append((closeness, sel, cnt))

            scored.sort(reverse=True, key=lambda x: (x[0], x[2]))
            for _, sel, _ in scored[:10]:
                if sel not in candidates:
                    candidates.append(sel)

    if not candidates:
        candidates.append("body")

    return candidates[0], candidates


def _extract_field_selectors(sel: Dict[str, Any]) -> Dict[str, str]:
    fields: Dict[str, str] = {}

    def pick(keys: List[str]) -> Optional[str]:
        for k in keys:
            v = sel.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    title = pick(["title", "paper_title", "paperTitle"])
    abstract = pick(["abstract", "paper_abstract", "summary"])
    pdf = pick(["pdf_link", "pdf", "pdf_url", "pdfUrl"])
    authors = pick(["authors", "author", "paper_authors"])
    link = pick(["link", "url", "paper_url", "paper_link"])

    if title:
        fields["title"] = title
    if abstract:
        fields["abstract"] = abstract
    if pdf:
        fields["pdf_link"] = pdf
    if authors:
        fields["authors"] = authors
    if link:
        fields["link"] = link

    return fields


def _text_or_empty(elem) -> str:
    if not elem:
        return ""
    try:
        return " ".join(elem.get_text(" ", strip=True).split())
    except Exception:
        return ""


def _href_or_empty(elem) -> str:
    if not elem:
        return ""
    try:
        href = elem.get("href")
        return str(href) if href else ""
    except Exception:
        return ""


def verify_selectors(html_path: str, selector_json: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Verify selectors by heuristic extraction and scoring (no ground truth)."""

    sel, schema_err = _load_selector_json(selector_json)

    diagnostics: Dict[str, Any] = {
        "schema_validation_error": schema_err,
        "item_selector_candidates": [],
        "selected_item_selector": None,
        "field_selectors": {},
        "extraction_errors": [],
        "warnings": [],
        "ok_criteria": {"min_items": 5, "min_title_hit_rate": 0.6},
    }

    metrics: Dict[str, Any] = {
        "n_items": 0,
        "title_hit_rate": 0.0,
        "pdf_hit_rate": 0.0,
        "abstract_hit_rate": 0.0,
        "authors_hit_rate": 0.0,
        "link_hit_rate": 0.0,
        "duplicate_title_rate": 0.0,
        "empty_text_rate": 0.0,
        "avg_title_len": 0.0,
        "avg_abstract_len": 0.0,
        "score_breakdown": {},
    }

    p = _as_path(html_path)
    if not p.exists():
        diagnostics["warnings"].append(f"html_not_found: {str(p)}")
        return {"ok": False, "score": 0.0, "metrics": metrics, "diagnostics": diagnostics}

    html = p.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Item selector
    explicit_item_selector = None
    for k in ITEM_SELECTOR_KEYS:
        v = sel.get(k)
        if isinstance(v, str) and v.strip():
            explicit_item_selector = v.strip()
            break

    title_selector = sel.get("title") if isinstance(sel.get("title"), str) else None

    if explicit_item_selector:
        item_selector = explicit_item_selector
        candidates = [explicit_item_selector]
    else:
        item_selector, candidates = _auto_detect_item_selector(soup, title_selector)

    diagnostics["item_selector_candidates"] = candidates
    diagnostics["selected_item_selector"] = item_selector

    # Fields
    field_selectors = _extract_field_selectors(sel)
    diagnostics["field_selectors"] = field_selectors

    # Extract items
    try:
        items = soup.select(item_selector)[:300]
    except Exception as e:
        diagnostics["extraction_errors"].append(f"item_selector_error: {e}")
        items = []

    n_items = len(items)
    metrics["n_items"] = n_items

    if n_items == 0:
        return {"ok": False, "score": 0.0, "metrics": metrics, "diagnostics": diagnostics}

    # Per-item extraction
    titles: List[str] = []
    abstracts: List[str] = []
    title_hits = pdf_hits = abstract_hits = authors_hits = link_hits = 0

    for item in items:
        try:
            # title
            if "title" in field_selectors:
                t_elem = item.select_one(field_selectors["title"]) if field_selectors["title"] else None
                t = _text_or_empty(t_elem)
                if t:
                    title_hits += 1
                    titles.append(t)
                else:
                    titles.append("")

            # abstract
            if "abstract" in field_selectors:
                a_elem = item.select_one(field_selectors["abstract"]) if field_selectors["abstract"] else None
                a = _text_or_empty(a_elem)
                if a:
                    abstract_hits += 1
                    abstracts.append(a)
                else:
                    abstracts.append("")

            # pdf link
            if "pdf_link" in field_selectors:
                p_elem = item.select_one(field_selectors["pdf_link"]) if field_selectors["pdf_link"] else None
                href = _href_or_empty(p_elem)
                if href:
                    pdf_hits += 1

            # authors
            if "authors" in field_selectors:
                au_elem = item.select_one(field_selectors["authors"]) if field_selectors["authors"] else None
                au = _text_or_empty(au_elem)
                if au:
                    authors_hits += 1

            # link
            if "link" in field_selectors:
                l_elem = item.select_one(field_selectors["link"]) if field_selectors["link"] else None
                href = _href_or_empty(l_elem)
                if href:
                    link_hits += 1

        except Exception as e:
            diagnostics["extraction_errors"].append(str(e))

    def rate(hits: int) -> float:
        return float(hits) / float(n_items) if n_items else 0.0

    metrics["title_hit_rate"] = rate(title_hits) if "title" in field_selectors else 0.0
    metrics["abstract_hit_rate"] = rate(abstract_hits) if "abstract" in field_selectors else 0.0
    metrics["pdf_hit_rate"] = rate(pdf_hits) if "pdf_link" in field_selectors else 0.0
    metrics["authors_hit_rate"] = rate(authors_hits) if "authors" in field_selectors else 0.0
    metrics["link_hit_rate"] = rate(link_hits) if "link" in field_selectors else 0.0

    # Quality metrics
    non_empty_titles = [t for t in titles if t]
    if non_empty_titles:
        uniq = len(set(non_empty_titles))
        metrics["duplicate_title_rate"] = 1.0 - (uniq / float(len(non_empty_titles)))
        metrics["avg_title_len"] = sum(len(t) for t in non_empty_titles) / float(len(non_empty_titles))
    else:
        metrics["duplicate_title_rate"] = 0.0
        metrics["avg_title_len"] = 0.0

    non_empty_abstracts = [a for a in abstracts if a]
    if non_empty_abstracts:
        metrics["avg_abstract_len"] = sum(len(a) for a in non_empty_abstracts) / float(len(non_empty_abstracts))
    else:
        metrics["avg_abstract_len"] = 0.0

    metrics["empty_text_rate"] = 1.0 - (len(non_empty_titles) / float(len(titles))) if titles else 1.0

    # Scoring
    # 1) Item count score (30)
    if n_items < 10:
        item_score = 30.0 * (n_items / 10.0)
    elif n_items <= 200:
        item_score = 30.0
    else:
        # Gradual penalty after 200, reach 0 at 500
        item_score = 30.0 * max(0.0, 1.0 - ((n_items - 200.0) / 300.0))

    # 2) Field extraction score (50)
    field_score = 0.0
    field_score += 30.0 * metrics["title_hit_rate"]
    field_score += 15.0 * metrics["pdf_hit_rate"]
    field_score += 5.0 * metrics["abstract_hit_rate"]

    # 3) Quality score (20 with penalties)
    quality = 20.0
    quality -= min(20.0, 20.0 * metrics["duplicate_title_rate"])
    quality -= min(20.0, 20.0 * metrics["empty_text_rate"])
    if metrics["avg_title_len"] and (metrics["avg_title_len"] < 5 or metrics["avg_title_len"] > 200):
        quality -= 5.0
    quality = max(0.0, min(20.0, quality))

    total = max(0.0, min(100.0, item_score + field_score + quality))

    metrics["score_breakdown"] = {
        "item_count": item_score,
        "field_extraction": field_score,
        "quality": quality,
    }

    ok = (n_items >= diagnostics["ok_criteria"]["min_items"]) and (
        metrics["title_hit_rate"] >= diagnostics["ok_criteria"]["min_title_hit_rate"]
    )

    return {"ok": bool(ok), "score": float(total), "metrics": metrics, "diagnostics": diagnostics}
