from typing import Any, Dict, List
import json

from logging_config import logger
from rag.retriever import get_rag_client_by_provider
from rag.pdf_loader import PDFLoader, LoadStatus
from settings import settings
from models import get_llm_by_usage

# Global RAG client and PDF loader for tools
_rag_client = None
_pdf_loader = None

def _get_rag_client():
    global _rag_client
    if _rag_client is None:
        _rag_client = get_rag_client_by_provider(settings.rag_provider)
    return _rag_client

def _get_pdf_loader():
    global _pdf_loader
    if _pdf_loader is None:
        if settings.chunk_strategy == "contextual":
            logger.info("Initializing PDFLoader with contextual chunking strategy.")
            _pdf_loader = PDFLoader(_get_rag_client(), llm_client=get_llm_by_usage('contextual'))
        else:
            _pdf_loader = PDFLoader(_get_rag_client())
    return _pdf_loader

# Global LLM for agentic tools
_agentic_llm = None

def _get_agentic_llm():
    global _agentic_llm
    if _agentic_llm is None:
        _agentic_llm = get_llm_by_usage('agentic')
    return _agentic_llm


# ============== Phase 1: Agentic Retrieval Tools ==============

def analyze_query(query: str) -> Dict[str, Any]:
    """
    [Phase 1] 分析用户查询，生成检索策略和多个子查询。
    这是 Agentic Retrieval 的第一步，必须在检索前调用！
    
    Args:
        query: 用户的原始查询
    
    Returns:
        Dict with analysis results:
        - query_type: 查询类型（comparison/definition/survey/technical_detail）
        - key_concepts: 关键概念列表
        - sub_queries: 子查询列表（按优先级排序）
        - estimated_complexity: 复杂度（high/medium/low）
        - should_use_hyde: 是否应该使用 HyDE
    """
    logger.info(f"Analyzing query: {query}")
    
    llm = _get_agentic_llm()
    
    prompt = f"""You are a research query analyzer. Analyze the following user query and generate a retrieval strategy.

User Query: "{query}"

Your task:
1. Identify the query type (comparison, definition, survey, technical_detail, or other)
2. Extract key concepts and terminology
3. Generate 2-4 sub-queries that progressively explore different aspects
   - Start with broad/general queries
   - Progress to specific/detailed queries
4. Estimate query complexity (high/medium/low)
5. Decide if HyDE (hypothetical document generation) would help

Guidelines for sub-queries:
- For comparisons: separate queries for each entity, then comparison
- For surveys: broad overview first, then specific techniques/methods
- For technical details: background first, then specific mechanisms
- Each sub-query should be self-contained and searchable

Respond ONLY with a valid JSON object (no markdown, no explanations):
{{
  "query_type": "comparison|definition|survey|technical_detail|other",
  "key_concepts": ["concept1", "concept2"],
  "sub_queries": ["query1", "query2", "query3"],
  "estimated_complexity": "high|medium|low",
  "should_use_hyde": true|false,
  "reasoning": "brief explanation of strategy"
}}"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith('```'):
            # Remove markdown code block markers
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            content = content.replace('```json', '').replace('```', '').strip()
        
        # Validate JSON
        analysis = json.loads(content)
        
        logger.info(
            "Query analysis done: type=%s complexity=%s sub_queries=%d hyde=%s"
            % (
                analysis.get("query_type"),
                analysis.get("estimated_complexity"),
                len(analysis.get("sub_queries", [])),
                analysis.get("should_use_hyde"),
            )
        )
        
        return analysis
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        # Fallback: return simple analysis
        fallback = {
            "query_type": "other",
            "key_concepts": [query],
            "sub_queries": [query],
            "estimated_complexity": "medium",
            "should_use_hyde": False,
            "reasoning": f"Analysis failed, using original query. Error: {str(e)}"
        }
        return fallback


def generate_hypothetical_answer(query: str) -> str:
    """
    [Optional - HyDE] 生成假想的理想答案文档，用于改善检索质量。
    适用于抽象/高层次的查询。生成的文档会被用于向量检索。
    
    Args:
        query: 子查询或原始查询
    
    Returns:
        假想的答案文档文本（会被 embedding 后用于检索）
    """
    logger.info(f"Generating hypothetical answer for: {query}")
    
    llm = _get_agentic_llm()
    
    prompt = f"""You are an expert researcher. Generate a hypothetical answer to the following query.

Query: "{query}"

Write a detailed, well-structured answer (2-3 paragraphs) as if you were writing an abstract or introduction section of a research paper that perfectly answers this query.

Include:
- Key technical terms and concepts
- Relevant methodologies or approaches
- Expected findings or conclusions
- References to common techniques or frameworks

Do NOT include citations like [1] or [2]. Just write the content.

Your hypothetical answer:"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"Generated hypothetical document ({len(content)} chars)")
        return content.strip()
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        logger.warning("Falling back to original query")
        return query  # Fallback to original query


def evaluate_retrieval_progress(
    original_query: str,
    current_results_summary: str,
    round_number: int,
) -> Dict[str, Any]:
    """
    [Self-Reflection] 评估当前检索结果是否充分，决定是否需要继续检索。
    
    Args:
        original_query: 用户的原始查询
        current_results_summary: 当前已检索结果的摘要（论文标题列表）
        round_number: 当前是第几轮检索（1-based）
    
    Returns:
        Dict with evaluation results:
        - is_sufficient: 是否已充分
        - coverage_score: 覆盖度评分 (0.0-1.0)
        - missing_aspects: 缺失的方面
        - should_continue: 是否应该继续检索
        - next_focus: 下一步应该关注什么
    """
    logger.info(f"Evaluating retrieval progress (round {round_number})")
    
    llm = _get_agentic_llm()
    
    prompt = f"""You are evaluating the sufficiency of retrieved research papers.

Original Query: "{original_query}"

Current Round: {round_number}

Retrieved Papers So Far:
{current_results_summary}

Your task:
1. Assess if the retrieved papers adequately cover the query
2. Identify any missing aspects or gaps
3. Decide if more retrieval rounds are needed
4. If continuing, suggest what to focus on next

Guidelines:
- Round 1-2: Usually continue unless results are perfect
- Round 3+: Only continue if critical information is missing
- Max 4 rounds recommended to avoid diminishing returns

Respond ONLY with a valid JSON object (no markdown, no explanations):
{{
  "is_sufficient": true|false,
  "coverage_score": 0.0-1.0,
  "missing_aspects": ["aspect1", "aspect2"],
  "should_continue": true|false,
  "next_focus": "description of what to search next",
  "reasoning": "brief explanation"
}}"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON
        content = content.strip()
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            content = content.replace('```json', '').replace('```', '').strip()
        
        evaluation = json.loads(content)
        
        logger.info(
            "Evaluation done: coverage=%.2f sufficient=%s continue=%s"
            % (
                evaluation.get("coverage_score", 0.0),
                evaluation.get("is_sufficient"),
                evaluation.get("should_continue"),
            )
        )
        if evaluation.get("missing_aspects"):
            logger.info(f"Missing aspects: {', '.join(evaluation.get('missing_aspects', []))}")
        if evaluation.get("next_focus"):
            logger.info(f"Next focus: {evaluation.get('next_focus')}")
        
        return evaluation
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        # Fallback: stop after round 3
        fallback = {
            "is_sufficient": round_number >= 3,
            "coverage_score": 0.5,
            "missing_aspects": [],
            "should_continue": round_number < 3,
            "next_focus": "Continue with remaining sub-queries",
            "reasoning": f"Evaluation failed, using heuristic. Error: {str(e)}"
        }
        return fallback


def rerank_results(original_query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [Final Step] 使用 LLM 对检索结果进行相关性评分和重排序。
    
    Args:
        original_query: 用户的原始查询
        results: 检索结果列表（包含 title, abstract, doc_id）
    
    Returns:
        Reranked results with relevance scores
    """
    logger.info(f"Reranking results for query: {original_query}")
    
    llm = _get_agentic_llm()
    
    if not results:
        logger.warning("No results to rerank")
        return results
    
    logger.info(f"Reranking {len(results)} papers")
    
    # Prepare results for LLM
    results_for_llm = []
    for i, r in enumerate(results[:15], 1):  # Limit to top 15 for efficiency
        results_for_llm.append({
            "index": i,
            "title": r.get("title", "Untitled"),
            "abstract": r.get("abstract", "")[:400],  # Truncate for token efficiency
            "doc_id": r.get("doc_id", "")
        })
    
    prompt = f"""You are a research paper relevance evaluator. Rate the relevance of each paper to the query.

Query: "{original_query}"

Papers:
{json.dumps(results_for_llm, ensure_ascii=False, indent=2)}

Your task:
For each paper, assign a relevance score from 0-10:
- 9-10: Highly relevant, directly addresses the query
- 7-8: Relevant, covers important aspects
- 5-6: Somewhat relevant, tangentially related
- 3-4: Marginally relevant
- 0-2: Not relevant or off-topic

Respond ONLY with a valid JSON array (no markdown, no explanations):
[
  {{"index": 1, "score": 8.5, "reason": "brief explanation"}},
  {{"index": 2, "score": 7.0, "reason": "brief explanation"}},
  ...
]"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract JSON
        content = content.strip()
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            content = content.replace('```json', '').replace('```', '').strip()
        
        scores = json.loads(content)
        
        # Apply scores to results
        score_map = {s["index"]: s["score"] for s in scores if "index" in s and "score" in s}
        
        for i, r in enumerate(results[:15], 1):
            if i in score_map:
                r["llm_relevance_score"] = score_map[i]
            else:
                r["llm_relevance_score"] = 5.0  # Default middle score
        
        # Sort by LLM score
        reranked = sorted(results[:15], key=lambda x: x.get("llm_relevance_score", 0), reverse=True)
        
        # Filter out low scores (< 4.0)
        reranked = [r for r in reranked if r.get("llm_relevance_score", 0) >= 4.0]
        
        logger.info(f"Reranking completed: {len(reranked)} kept from {len(results)}")
        
        return reranked
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        logger.warning("Returning original results without reranking")
        return results  # Return original results


# ============== Phase 2: Abstract Search ==============

def search_abstracts(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    [Phase 2] 搜索论文摘要，找出相关论文。
    这是搜索的第一步，返回候选论文列表。
    
    Args:
        query: 搜索关键词或自然语言查询
        k: 返回论文数量 (默认 5)
    
    Returns:
        候选论文列表，包含 title, abstract, doc_id
    """
    logger.info(f"Searching abstracts (top {k}) for: {query}")
    
    client = _get_rag_client()
    results = client.search_abstracts(query, k)
    
    if not results:
        logger.warning("No papers found matching the query")
        return []
    
    logger.info(f"Found {len(results)} papers")
    
    normalized: List[Dict[str, Any]] = []
    for r in results:
        normalized.append(
            {
                "title": r.get("title", ""),
                "abstract": r.get("abstract", ""),
                "doc_id": r.get("doc_id", ""),
                "url": r.get("url", ""),
                "pdf_url": r.get("pdf_url", ""),
                "conference_name": r.get("conference_name", ""),
                "conference_year": r.get("conference_year", ""),
                "score": r.get("score", 0.0),
                "section_category": r.get("section_category", 0),
                "parent_section": r.get("parent_section", ""),
                "page_number": r.get("page_number", 0),
            }
        )

    return normalized


# ============== Phase 3: Lazy Load PDF ==============

def load_paper_pdfs(doc_ids: List[str]) -> Dict[str, Any]:
    """
    [Phase 3] 加载指定论文的 PDF 内容到数据库。
    在使用 search_paper_content 之前必须调用此工具！
    会自动跳过已加载的论文。
    
    Args:
        doc_ids: 要加载的论文 doc_id 列表（从 search_abstracts 获取）
    
    Returns:
        Dict with per-doc load results and summary counts
    
    注意：
        - 一次建议加载 3-5 篇论文，避免等待过长
        - 加载过程需要下载和解析 PDF，可能需要一些时间
    """
    loader = _get_pdf_loader()
    results = loader.load_papers(doc_ids)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    serialized: Dict[str, Any] = {}
    
    for doc_id, result in results.items():
        serialized[doc_id] = {
            "status": result.status.value,
            "message": result.message,
            "chunks_count": result.chunks_count,
        }
        
        if result.status == LoadStatus.SUCCESS:
            success_count += 1
        elif result.status == LoadStatus.ALREADY_EXISTS:
            skip_count += 1
        else:
            fail_count += 1
    
    return {
        "summary": {
            "loaded": success_count,
            "skipped": skip_count,
            "failed": fail_count,
        },
        "results": serialized,
    }


# ============== Phase 4: Deep Search ==============

def search_paper_content(
    query: str,
    doc_ids: List[str] | None = None,
    category: int = -1,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    [Phase 4] 在已加载的论文中搜索具体内容。
    注意：必须先用 load_paper_pdfs 加载论文！
    
    Args:
        query: 搜索查询
        doc_ids: 要搜索的论文 doc_id 列表（留空则搜索所有已加载的论文）
        category: 章节类型过滤，-1 表示全部
            0 = Abstract (摘要)
            1 = Introduction (背景、动机)
            2 = Method (技术细节、算法、架构)
            3 = Evaluation (实验、结果、性能数据)
            4 = Conclusion (结论)
            6 = Related Work (注意：描述的是其他论文！)
        k: 返回结果数量 (默认 5)
    
    Returns:
        匹配的文本片段及其元数据列表
    """
    client = _get_rag_client()
    
    doc_ids = doc_ids or []

    # 如果指定了 doc_ids，需要逐个搜索并合并结果
    # TODO: 优化为批量搜索
    if doc_ids:
        all_results = []
        for doc_id in doc_ids:
            results = client.search_by_section(
                query, 
                doc_id=doc_id, 
                section_category=category if category >= 0 else None, 
                k=k
            )
            all_results.extend(results)
        # 按相关性排序（假设有 score 字段）
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = all_results[:k]
    else:
        results = client.search_by_section(
            query, 
            doc_id=None, 
            section_category=category if category >= 0 else None, 
            k=k
        )
    
    if not results:
        return []

    normalized: List[Dict[str, Any]] = []
    for r in results:
        normalized.append(
            {
                "title": r.get("title", ""),
                "abstract": r.get("text", ""),
                "doc_id": r.get("doc_id", ""),
                "chunk_id": r.get("chunk_id", -1),
                "section_category": r.get("section_category", 0),
                "parent_section": r.get("parent_section", ""),
                "page_number": r.get("page_number", 0),
                "score": r.get("score", 0.0),
            }
        )

    return normalized


# ============== Context Tools ==============

def get_context_window(doc_id: str, chunk_id: int, window: int = 1) -> str:
    """
    获取指定 chunk 周围的上下文文本。
    当检索到的片段不完整或被截断时使用。
    
    Args:
        doc_id: 论文的 doc_id
        chunk_id: chunk_id（从 search_paper_content 结果获取）
        window: 前后各包含多少个 chunk (默认 1)
    
    Returns:
        扩展的上下文文本
    """
    client = _get_rag_client()
    context = client.get_context_window(doc_id, chunk_id, window)
    
    if not context:
        return "Could not retrieve context for this chunk."
    
    return f"[Context Window for doc_id={doc_id}, chunk_id={chunk_id}]\n\n{context}"


# ============== Searcher Class ==============

class Searcher:
    """RAG searcher that supports both simple and agentic modes.

    - Simple mode: Direct vector search (only abstracts)
    - Agentic mode: LLM agent with Lazy Load PDF workflow
    
    Lazy Load Workflow:
    1. search_abstracts - 找候选论文
    2. load_paper_pdfs - 按需加载 PDF
    3. search_paper_content - 搜索正文
    4. get_context_window - 获取更多上下文
    """

    def __init__(self) -> None:
        self.rag_client = get_rag_client_by_provider(settings.rag_provider)
        self.top_k = settings.milvus_top_k

    def _assign_ids(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for idx, hit in enumerate(hits, 1):
            if "id" not in hit:
                hit["id"] = idx
        return hits

    def _agentic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Run the LangGraph searcher subgraph."""
        logger.info(f"Agentic search started for: {query}")
        from graph.subgraphs.searcher import run_searcher_subgraph

        return run_searcher_subgraph(query, k=k)

    def search(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        """Query vector store and return normalized hits with ids."""
        k = k or self.top_k
        
        if settings.enable_agentic_rag:
            try:
                hits = self._agentic_search(query, k)
                return self._assign_ids(hits)
            except Exception as exc:
                logger.error(f"Agentic search failed, falling back to simple search: {exc}")
            
        # Simple mode: direct vector search
        raw_hits = self.rag_client.query_relevant_documents(query)
        hits: List[Dict[str, Any]] = []
        for idx, hit in enumerate(raw_hits[:k]):
            hits.append(
                {
                    "id": idx + 1,
                    "title": hit.get("title", ""),
                    "abstract": hit.get("abstract", ""),
                    "url": hit.get("url", ""),
                    "doc_id": hit.get("doc_id", ""),
                    "score": hit.get("score", 0.0),
                    "conference_name": hit.get("conference_name", ""),
                    "conference_year": hit.get("conference_year", ""),
                    "conference_round": hit.get("conference_round", ""),
                    "section_category": hit.get("section_category", 0),
                    "parent_section": hit.get("parent_section", ""),
                    "page_number": hit.get("page_number", 0),
                }
            )
        logger.info(f"Searcher retrieved {len(hits)} hits for query: {query}")
        return hits

    def format_hits(self, hits: List[Dict[str, Any]], max_len: int = 600) -> str:
        """Helper for coordinator: render hits into concise numbered blocks."""
        if not hits:
            return "No relevant documents found."

        for idx, hit in enumerate(hits, 1):
            if "id" not in hit:
                hit["id"] = idx
        
        from parser.pdf_parser import SectionCategory
        
        blocks = []
        for h in hits:
            meta_parts = []
            if h.get("conference_name"):
                meta_parts.append(str(h["conference_name"]))
            if h.get("conference_year"):
                meta_parts.append(str(h["conference_year"]))
            
            # Add structure info to metadata
            cat_id = h.get("section_category", 0)
            try:
                cat_name = SectionCategory(cat_id).name
            except:
                cat_name = "UNKNOWN"
            
            if cat_name != "ABSTRACT":
                meta_parts.append(f"Section: {cat_name}")
            
            parent = h.get("parent_section")
            if parent:
                meta_parts.append(f"Parent: {parent}")
                
            page = h.get("page_number")
            if page and page > 0:
                meta_parts.append(f"Page: {page}")

            meta = " | ".join(meta_parts)
            abstract = h.get("abstract", "")
            if len(abstract) > max_len:
                abstract = abstract[:max_len] + "..."
            blocks.append(
                f"[{h['id']}] {h.get('title') or 'Untitled'}\n"
                f"{meta + '\\n' if meta else ''}"
                f"URL: {h.get('url') or 'N/A'}\n"
                f"Content: {abstract}"
            )
        return "\n\n".join(blocks)


def invoke_searcher(query: str, k: int | None = None) -> List[Dict[str, Any]]:
    """Convenience wrapper used by coordinator/tools. Returns hits only."""
    searcher = Searcher()
    return searcher.search(query, k)
