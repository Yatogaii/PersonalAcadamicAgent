"""
QA 生成用的 Prompt 模板
"""

# ============== Easy Questions ==============

EASY_QA_PROMPT = """
You are a test question generator for an academic paper search system.

# Paper Summaries
{paper_summaries}

# Task
Generate {count} EASY questions that can be answered by finding specific papers.

Requirements:
- Questions should contain keywords that directly appear in the paper titles/abstracts
- Each question should have exactly 1-2 relevant papers as the answer
- Questions should be natural and realistic

# Output Format (JSON array)
[
  {{
    "question": "Which paper proposes the XXX method?",
    "expected_doc_ids": ["doc_id_1"],
    "reference_answer": "The paper 'Title' proposes XXX method for ...",
    "answer_source": "abstract"
  }},
  ...
]
"""


# ============== Medium Questions ==============

MEDIUM_QA_PROMPT = """
You are a test question generator for an academic paper search system.

# Paper Summaries (with method and evaluation details)
{paper_summaries}

# Task
Generate {count} MEDIUM difficulty questions that require semantic understanding.

Requirements:
- Questions should NOT contain exact keywords from papers (require semantic matching)
- Questions should ask about methodology details or experimental results
- Each question should have 1-3 relevant papers as the answer

# Output Format (JSON array)
[
  {{
    "question": "How does the approach in XXX paper handle the challenge of ...?",
    "expected_doc_ids": ["doc_id_1"],
    "reference_answer": "The paper addresses this by ...",
    "answer_source": "method"
  }},
  ...
]
"""


# ============== Hard Questions ==============

HARD_QA_PROMPT = """
You are a test question generator for an academic paper search system.

# Paper Summaries (grouped by research area)
{paper_summaries}

# Task
Generate {count} HARD questions that require cross-paper analysis.

Requirements:
- Questions should require information from multiple papers
- Questions could be comparative, survey-style, or trend analysis
- Expected answers should reference 2-5 papers

# Output Format (JSON array)
[
  {{
    "question": "What are the main approaches to XXX problem in recent research?",
    "expected_doc_ids": ["doc_id_1", "doc_id_2", "doc_id_3"],
    "reference_answer": "There are several approaches: Paper A proposes ..., Paper B uses ...",
    "answer_source": "multiple",
    "is_multi_paper": true
  }},
  ...
]
"""


# ============== 辅助函数 ==============

def format_paper_summaries_for_easy(annotations: list, batch_size: int = 30) -> str:
    """格式化论文摘要用于 Easy 问题生成"""
    lines = []
    for ann in annotations[:batch_size]:
        lines.append(f"- doc_id: {ann.doc_id}")
        lines.append(f"  title: {ann.title}")
        lines.append(f"  summary: {ann.summary}")
        lines.append(f"  keywords: {', '.join(ann.keywords)}")
        lines.append("")
    return "\n".join(lines)


def format_paper_summaries_for_medium(annotations: list, batch_size: int = 20) -> str:
    """格式化论文摘要用于 Medium 问题生成（包含 section 信息）"""
    lines = []
    for ann in annotations[:batch_size]:
        lines.append(f"- doc_id: {ann.doc_id}")
        lines.append(f"  title: {ann.title}")
        lines.append(f"  summary: {ann.summary}")
        if ann.method_summary:
            lines.append(f"  method: {ann.method_summary}")
        if ann.eval_summary:
            lines.append(f"  evaluation: {ann.eval_summary}")
        lines.append("")
    return "\n".join(lines)


def format_paper_summaries_for_hard(annotations_by_area: dict, batch_size: int = 30) -> str:
    """格式化论文摘要用于 Hard 问题生成（按领域分组）"""
    lines = []
    for area, annotations in annotations_by_area.items():
        lines.append(f"## Research Area: {area}")
        for ann in annotations[:batch_size // len(annotations_by_area)]:
            lines.append(f"- [{ann.doc_id}] {ann.title}: {ann.summary}")
        lines.append("")
    return "\n".join(lines)
