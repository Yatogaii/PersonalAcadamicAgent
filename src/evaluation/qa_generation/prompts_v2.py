"""
QA 生成用的 Prompt 模板 (V2)

改进版：支持多层次问题生成
- Level 1 (Easy): 单论文精确题
- Level 2 (Medium): 单论文推理题
- Level 3 (Hard): 跨论文比较题
- Level 4 (Expert): 领域综述题
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.qa_generation.qa_generator import ChunkInfo
    from evaluation.qa_generation.paper_clustering import PaperCluster


# ============== Level 1: 单论文精确题 (Easy) ==============

LEVEL1_EASY_PROMPT = """
You are generating test questions for an academic paper retrieval system.

# Paper Information
{paper_summaries}

# Task
Generate {count} EASY questions that test basic paper retrieval.

## Requirements:
1. Questions should ask about WHAT a specific paper proposes/studies
2. Answers should be directly found in the abstract
3. Questions should contain some keywords from the paper (but not the exact title)
4. Each question has exactly ONE correct paper

## Examples:
- "What paper proposes using trusted hardware for bot detection?"
- "Which research studies VPN security mental models?"

# Output Format (JSON array)
[
  {{
    "question": "Which paper studies ...",
    "expected_doc_ids": ["doc_id_1"],
    "expected_chunk_ids": [0, 1],
    "reference_answer": "The paper 'Title' proposes...",
    "answer_source": "abstract"
  }}
]

Output ONLY valid JSON array. No markdown, no explanation.
"""


# ============== Level 2: 单论文推理题 (Medium) ==============

LEVEL2_MEDIUM_PROMPT = """
You are generating test questions for an academic paper retrieval system.

# Paper Information (with methodology details)
{paper_summaries}

# Task
Generate {count} MEDIUM questions that require understanding paper methodology.

## Requirements:
1. Questions should ask HOW something is achieved, not WHAT paper does it
2. Answers require reading the methodology/evaluation sections
3. Do NOT include paper titles or obvious keywords in questions
4. Questions should be about:
   - Implementation details
   - Experimental setup
   - Evaluation metrics
   - Limitations or trade-offs

## Good Examples:
- "How can side-channel attacks be detected with bounded Type-1 error rates?"
- "What techniques enable secure gradient aggregation in distributed learning?"

## Bad Examples (avoid):
- "What does the DMA paper propose?" (mentions paper)
- "How does CACTI work?" (uses exact name)

# Output Format (JSON array)
[
  {{
    "question": "How can ... be achieved?",
    "expected_doc_ids": ["doc_id_1"],
    "expected_chunk_ids": [10, 11, 12],
    "reference_answer": "The approach uses...",
    "answer_source": "method"
  }}
]

Output ONLY valid JSON array. No markdown, no explanation.
"""


# ============== Level 3: 跨论文比较题 (Hard) ==============

LEVEL3_COMPARISON_PROMPT = """
You are generating COMPARISON questions that require analyzing multiple related papers.

# Paper Cluster: {cluster_theme}
These papers share common research focus:

{cluster_papers}

# Task
Generate {count} questions that COMPARE or CONTRAST these related papers.

## Requirements:
1. Questions must require reading 2-3 papers to answer
2. Focus on COMPARING:
   - Different approaches to the same problem
   - Trade-offs between methods
   - Strengths and weaknesses
   - Complementary techniques

3. NEVER mention paper titles, doc_ids, or author names in questions
4. Questions should be abstract enough that keywords alone won't find the answer

## Good Examples:
- "How do different approaches to detecting timing vulnerabilities compare in terms of accuracy vs. performance?"
- "What are the trade-offs between hardware-based and software-based defenses against side-channel attacks?"

## Bad Examples (avoid):
- "Compare paper A and paper B" (mentions papers)
- "What are side-channel attacks?" (too simple, single paper can answer)

# Output Format (JSON array)
[
  {{
    "question": "How do different approaches to X compare in terms of Y?",
    "expected_doc_ids": ["doc_id_1", "doc_id_2"],
    "expected_chunk_ids": null,
    "reference_answer": "Paper 1 uses approach A which..., while Paper 2 uses approach B which...",
    "answer_source": "multiple",
    "is_multi_paper": true
  }}
]

Output ONLY valid JSON array. No markdown, no explanation.
"""


# ============== Level 4: 领域综述题 (Expert) ==============

LEVEL4_SURVEY_PROMPT = """
You are generating SURVEY questions that require synthesizing knowledge across a research area.

# Research Area Overview
{area_overview}

# Available Papers
{paper_list}

# Task
Generate {count} SURVEY-style questions about this research area.

## Requirements:
1. Questions should ask about TRENDS, PATTERNS, or LANDSCAPE of the field
2. Answers require synthesizing information from 3+ papers
3. Questions should be what a researcher reviewing the field would ask
4. Focus on:
   - Common challenges across papers
   - Evolution of techniques
   - Open problems and future directions
   - Methodological patterns

## Good Examples:
- "What are the main approaches researchers use to evaluate security tool usability?"
- "How has the field addressed the trade-off between security and performance?"
- "What common experimental methodologies are used in adversarial ML research?"

## Bad Examples:
- "What papers discuss X?" (too simple)
- "Summarize paper X" (single paper)

# Output Format (JSON array)
[
  {{
    "question": "What are the common methodological approaches in X research?",
    "expected_doc_ids": ["id1", "id2", "id3", "id4"],
    "expected_chunk_ids": null,
    "reference_answer": "Researchers commonly use: 1) ... 2) ... 3) ...",
    "answer_source": "multiple",
    "is_multi_paper": true
  }}
]

Output ONLY valid JSON array. No markdown, no explanation.
"""


# ============== 格式化函数 ==============

def format_for_level1(
    all_chunks: dict[str, list["ChunkInfo"]], 
    max_papers: int = 30
) -> str:
    """格式化用于 Level 1 (Easy) 问题"""
    lines = []
    papers = list(all_chunks.items())[:max_papers]
    
    for doc_id, chunks in papers:
        if not chunks:
            continue
        
        title = chunks[0].title
        abstract_chunks = [c for c in chunks if c.section_category == 0]
        abstract = abstract_chunks[0].chunk_text[:500] if abstract_chunks else ""
        
        lines.append(f"doc_id: {doc_id}")
        lines.append(f"title: {title}")
        lines.append(f"abstract: {abstract}")
        lines.append("")
    
    return "\n".join(lines)


def format_for_level2(
    all_chunks: dict[str, list["ChunkInfo"]], 
    max_papers: int = 15
) -> str:
    """格式化用于 Level 2 (Medium) 问题"""
    lines = []
    papers = list(all_chunks.items())[:max_papers]
    
    for doc_id, chunks in papers:
        if not chunks:
            continue
        
        title = chunks[0].title
        
        # Abstract
        abstract_chunks = [c for c in chunks if c.section_category == 0]
        abstract = abstract_chunks[0].chunk_text[:300] if abstract_chunks else ""
        
        # Method section (category 2)
        method_chunks = [c for c in chunks if c.section_category == 2]
        method_texts = []
        method_ids = []
        for mc in method_chunks[:3]:
            method_texts.append(mc.chunk_text[:300])
            method_ids.append(mc.chunk_index)
        
        # Evaluation section (category 4)
        eval_chunks = [c for c in chunks if c.section_category == 4]
        eval_text = eval_chunks[0].chunk_text[:200] if eval_chunks else ""
        
        lines.append(f"doc_id: {doc_id}")
        lines.append(f"title: {title}")
        lines.append(f"abstract: {abstract}")
        if method_texts:
            lines.append(f"methodology: {' '.join(method_texts)[:600]}")
            lines.append(f"methodology_chunk_ids: {method_ids}")
        if eval_text:
            lines.append(f"evaluation: {eval_text}")
        lines.append("")
    
    return "\n".join(lines)


def format_cluster_for_level3(
    cluster: "PaperCluster",
    all_chunks: dict[str, list["ChunkInfo"]]
) -> str:
    """格式化聚类内论文用于 Level 3 (Comparison) 问题"""
    lines = []
    
    for doc_id in cluster.paper_ids:
        chunks = all_chunks.get(doc_id, [])
        if not chunks:
            continue
        
        title = chunks[0].title
        
        # Abstract
        abstract_chunks = [c for c in chunks if c.section_category == 0]
        abstract = abstract_chunks[0].chunk_text[:400] if abstract_chunks else ""
        
        # Method summary
        method_chunks = [c for c in chunks if c.section_category == 2]
        method = method_chunks[0].chunk_text[:300] if method_chunks else ""
        
        lines.append(f"[{doc_id}]")
        lines.append(f"Title: {title}")
        lines.append(f"Abstract: {abstract}")
        if method:
            lines.append(f"Approach: {method}")
        lines.append("")
    
    return "\n".join(lines)


def format_for_level4(
    all_chunks: dict[str, list["ChunkInfo"]],
    clusters: list["PaperCluster"] = None
) -> tuple[str, str]:
    """格式化用于 Level 4 (Survey) 问题
    
    Returns:
        (area_overview, paper_list)
    """
    # 生成领域概述
    all_titles = []
    all_abstracts = []
    
    for doc_id, chunks in all_chunks.items():
        if not chunks:
            continue
        title = chunks[0].title
        all_titles.append(title)
        
        abstract_chunks = [c for c in chunks if c.section_category == 0]
        if abstract_chunks:
            all_abstracts.append(abstract_chunks[0].chunk_text[:200])
    
    # 简单的领域概述
    area_overview = f"""
This collection contains {len(all_chunks)} security research papers covering topics including:
- System security and vulnerabilities
- Privacy and machine learning
- Network security and attacks
- User studies and security behavior

Common themes: vulnerability detection, defense mechanisms, evaluation methodologies
"""
    
    # 论文列表
    paper_lines = []
    for doc_id, chunks in list(all_chunks.items())[:20]:
        if not chunks:
            continue
        title = chunks[0].title
        abstract_chunks = [c for c in chunks if c.section_category == 0]
        summary = abstract_chunks[0].chunk_text[:150] if abstract_chunks else ""
        paper_lines.append(f"[{doc_id}] {title}")
        paper_lines.append(f"    {summary}")
        paper_lines.append("")
    
    return area_overview, "\n".join(paper_lines)


# ============== 兼容旧版 ==============

# 保持旧函数名的兼容性
EASY_QA_PROMPT = LEVEL1_EASY_PROMPT
MEDIUM_QA_PROMPT = LEVEL2_MEDIUM_PROMPT
HARD_QA_PROMPT = LEVEL3_COMPARISON_PROMPT

format_chunks_for_easy = format_for_level1
format_chunks_for_medium = format_for_level2

def format_chunks_for_hard(all_chunks, max_papers=30):
    """兼容旧版"""
    return format_for_level1(all_chunks, max_papers)
