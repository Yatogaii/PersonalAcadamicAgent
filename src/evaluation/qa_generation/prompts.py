"""
QA 生成用的 Prompt 模板
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.qa_generation.qa_generator import ChunkInfo

# ============== Easy Questions ==============

EASY_QA_PROMPT = """
You are a test question generator for an academic paper search system.

# Paper Information
{paper_summaries}

# Task
Generate {count} EASY questions that can be answered by finding specific papers.

Requirements:
- Questions should contain keywords that directly appear in the paper titles/abstracts
- Each question should have exactly 1-2 relevant papers as the answer
- Questions should be natural and realistic
- Use the doc_id values exactly as provided

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

Output ONLY the JSON array, no other text.
"""


# ============== Medium Questions ==============

MEDIUM_QA_PROMPT = """
You are a test question generator for an academic paper search system.

# Paper Information (with method details)
{paper_summaries}

# Task
Generate {count} MEDIUM difficulty questions that require semantic understanding.

## Requirements:
1. **Questions should NOT contain exact keywords from paper titles**
   - Use paraphrased or conceptual descriptions instead
   - BAD: "How does the CACTI paper handle CAPTCHA?"
   - GOOD: "How can trusted execution environments help reduce bot detection friction?"

2. **Questions should ask about methodology details or experimental results**
   - Focus on HOW something is done, not WHAT paper does it

3. **Do NOT reference doc_ids or paper titles in questions**
   - Use doc_id values ONLY in expected_doc_ids field

4. Each question should have 1-3 relevant papers as the answer

# Output Format (JSON array)
[
  {{
    "question": "How can hardware-based security features reduce the burden of human verification systems?",
    "expected_doc_ids": ["doc_id_1"],
    "expected_chunk_ids": [5, 6, 7],
    "reference_answer": "The paper addresses this by ...",
    "answer_source": "method"
  }},
  ...
]

Output ONLY the JSON array, no other text.
"""


# ============== Hard Questions ==============

HARD_QA_PROMPT = """
You are a test question generator for an academic paper search system.

# Paper Information (multiple papers)
{paper_summaries}

# Task
Generate {count} HARD questions that require cross-paper analysis and semantic understanding.

## CRITICAL Requirements:
1. **NEVER mention paper titles, doc_ids, or specific paper identifiers in questions**
   - BAD: "How does the VPN paper compare to the CAPTCHA paper?"
   - BAD: "What does [doc_id_xxx] propose?"
   - GOOD: "What are the common user study methodologies in security research?"

2. **Questions must be ABSTRACT and THEMATIC** - focus on:
   - Research trends or patterns across papers
   - Common challenges and solution approaches
   - Comparative methodology analysis
   - Synthesizing findings from multiple sources

3. **Minimize searchable keywords** - questions should require semantic understanding:
   - BAD: "What are VPN security mental model challenges?" (too many specific terms)
   - GOOD: "How do researchers study end-user security perceptions?"

4. Expected answers should reference 2-5 papers
5. Use the doc_id values ONLY in expected_doc_ids field, never in the question itself

# Output Format (JSON array)
[
  {{
    "question": "What methodologies do researchers use to evaluate user understanding of security tools?",
    "expected_doc_ids": ["doc_id_1", "doc_id_2", "doc_id_3"],
    "reference_answer": "Researchers commonly use surveys, interviews, and think-aloud protocols...",
    "answer_source": "multiple",
    "is_multi_paper": true
  }},
  ...
]

Output ONLY the JSON array, no other text.
"""


# ============== 格式化函数 ==============

def format_chunks_for_easy(
    all_chunks: dict[str, list["ChunkInfo"]], 
    max_papers: int = 30
) -> str:
    """格式化 chunks 用于 Easy 问题生成（只用 abstract）"""
    lines = []
    papers = list(all_chunks.items())[:max_papers]
    
    for doc_id, chunks in papers:
        if not chunks:
            continue
        
        title = chunks[0].title
        # 找 abstract（section_category == 0）
        abstract_chunks = [c for c in chunks if c.section_category == 0]
        abstract = abstract_chunks[0].chunk_text[:500] if abstract_chunks else ""
        
        lines.append(f"- doc_id: {doc_id}")
        lines.append(f"  title: {title}")
        lines.append(f"  abstract: {abstract}")
        lines.append("")
    
    return "\n".join(lines)


def format_chunks_for_medium(
    all_chunks: dict[str, list["ChunkInfo"]], 
    max_papers: int = 20
) -> str:
    """格式化 chunks 用于 Medium 问题生成（包含 method section）"""
    lines = []
    papers = list(all_chunks.items())[:max_papers]
    
    for doc_id, chunks in papers:
        if not chunks:
            continue
        
        title = chunks[0].title
        
        # 找 abstract（section_category == 0）
        abstract_chunks = [c for c in chunks if c.section_category == 0]
        abstract = abstract_chunks[0].chunk_text[:300] if abstract_chunks else ""
        
        # 找 method section（section_category == 2）
        method_chunks = [c for c in chunks if c.section_category == 2]
        method_text = ""
        if method_chunks:
            method_text = " ".join([c.chunk_text[:200] for c in method_chunks[:2]])
        
        # 找 evaluation section（section_category == 4）
        eval_chunks = [c for c in chunks if c.section_category == 4]
        eval_text = ""
        if eval_chunks:
            eval_text = eval_chunks[0].chunk_text[:200]
        
        lines.append(f"- doc_id: {doc_id}")
        lines.append(f"  title: {title}")
        lines.append(f"  abstract: {abstract}")
        if method_text:
            lines.append(f"  method: {method_text[:400]}")
        if eval_text:
            lines.append(f"  evaluation: {eval_text}")
        
        # 添加 chunk_ids 供参考
        method_chunk_ids = [c.chunk_index for c in method_chunks[:5]]
        if method_chunk_ids:
            lines.append(f"  method_chunk_ids: {method_chunk_ids}")
        
        lines.append("")
    
    return "\n".join(lines)


def format_chunks_for_hard(
    all_chunks: dict[str, list["ChunkInfo"]], 
    max_papers: int = 30
) -> str:
    """格式化 chunks 用于 Hard 问题生成（多论文综述）"""
    lines = []
    papers = list(all_chunks.items())[:max_papers]
    
    lines.append("# Available Papers for Cross-Paper Questions")
    lines.append("")
    
    for doc_id, chunks in papers:
        if not chunks:
            continue
        
        title = chunks[0].title
        
        # 只用 abstract 的简要描述
        abstract_chunks = [c for c in chunks if c.section_category == 0]
        abstract = abstract_chunks[0].chunk_text[:200] if abstract_chunks else ""
        
        lines.append(f"- [{doc_id}] {title}")
        lines.append(f"  Summary: {abstract}")
        lines.append("")
    
    return "\n".join(lines)


# ============== 旧版兼容函数（可删除） ==============

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
