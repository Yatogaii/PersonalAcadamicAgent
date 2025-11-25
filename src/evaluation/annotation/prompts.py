"""
标注用的 Prompt 模板
"""

# ============== Paper-level 标注 ==============

PAPER_ANNOTATION_PROMPT = """
You are an academic paper analyzer. Given a paper's title and abstract, generate a structured annotation.

# Input
Title: {title}
Abstract: {abstract}

# Task
Generate the following in JSON format:
1. summary: A 1-2 sentence summary of the paper's core contribution (in English)
2. keywords: 3-5 key technical terms or concepts (in English)
3. research_area: The primary research area (choose from: security, privacy, systems, networking, ML/AI, software engineering, other)

# Output Format (JSON only, no markdown)
{{
  "summary": "...",
  "keywords": ["...", "...", "..."],
  "research_area": "..."
}}
"""


# ============== Section-level 标注 ==============

METHOD_ANNOTATION_PROMPT = """
You are an academic paper analyzer. Given the Method/Design section of a paper, generate a structured summary.

# Paper Title
{title}

# Method Section Content
{content}

# Task
Generate the following in JSON format:
1. summary: A 2-3 sentence summary of the core methodology/approach (in English)
2. keywords: 3-5 key technical terms specific to the method (in English)

# Output Format (JSON only, no markdown)
{{
  "summary": "...",
  "keywords": ["...", "...", "..."]
}}
"""


EVALUATION_ANNOTATION_PROMPT = """
You are an academic paper analyzer. Given the Evaluation/Experiment section of a paper, generate a structured summary.

# Paper Title
{title}

# Evaluation Section Content
{content}

# Task
Generate the following in JSON format:
1. summary: A 2-3 sentence summary of key experimental results and findings (in English)
2. keywords: 3-5 key terms related to the evaluation (metrics, datasets, baselines, etc.)

# Output Format (JSON only, no markdown)
{{
  "summary": "...",
  "keywords": ["...", "...", "..."]
}}
"""


# ============== 辅助函数 ==============

def build_paper_annotation_prompt(title: str, abstract: str) -> str:
    """构建 Paper-level 标注 prompt"""
    return PAPER_ANNOTATION_PROMPT.format(title=title, abstract=abstract)


def build_method_annotation_prompt(title: str, content: str) -> str:
    """构建 Method section 标注 prompt"""
    # 截断过长内容
    max_length = 3000
    if len(content) > max_length:
        content = content[:max_length] + "\n... [truncated]"
    return METHOD_ANNOTATION_PROMPT.format(title=title, content=content)


def build_evaluation_annotation_prompt(title: str, content: str) -> str:
    """构建 Evaluation section 标注 prompt"""
    max_length = 3000
    if len(content) > max_length:
        content = content[:max_length] + "\n... [truncated]"
    return EVALUATION_ANNOTATION_PROMPT.format(title=title, content=content)
