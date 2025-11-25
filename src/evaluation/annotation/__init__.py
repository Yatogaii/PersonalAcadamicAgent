"""
Annotation Module - LLM 标注

负责：
1. Paper-level 标注 (summary, keywords, research_area)
2. Section-level 标注 (method_summary, eval_summary)
"""

from .paper_annotator import PaperAnnotator
from .section_annotator import SectionAnnotator

__all__ = ["PaperAnnotator", "SectionAnnotator"]
