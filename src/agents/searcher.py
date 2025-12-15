from typing import Any, Dict, List
import json

from logging_config import logger
from rag.retriever import get_rag_client_by_provider
from rag.pdf_loader import PDFLoader, LoadStatus
from settings import settings
from models import get_llm_by_usage
from prompts.template import apply_prompt_template
from langchain.tools import tool
from langchain.agents import create_agent

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

@tool
def analyze_query(query: str) -> str:
    """
    [Phase 1] 分析用户查询，生成检索策略和多个子查询。
    这是 Agentic Retrieval 的第一步，必须在检索前调用！
    
    Args:
        query: 用户的原始查询
    
    Returns:
        JSON格式的分析结果，包含：
        - query_type: 查询类型（comparison/definition/survey/technical_detail）
        - key_concepts: 关键概念列表
        - sub_queries: 子查询列表（按优先级排序）
        - estimated_complexity: 复杂度（high/medium/low）
        - should_use_hyde: 是否应该使用 HyDE
    """
    logger.info("="*80)
    logger.info("🎯 [PHASE 1: QUERY ANALYSIS] Starting query analysis...")
    logger.info(f"📝 Original Query: {query}")
    logger.info("="*80)
    
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
        
        logger.success("✅ Query analysis completed successfully!")
        logger.info(f"   📊 Query Type: {analysis.get('query_type')}")
        logger.info(f"   🔥 Complexity: {analysis.get('estimated_complexity')}")
        logger.info(f"   🔑 Key Concepts: {', '.join(analysis.get('key_concepts', []))}")
        logger.info(f"   📋 Generated {len(analysis.get('sub_queries', []))} sub-queries:")
        for i, sq in enumerate(analysis.get('sub_queries', []), 1):
            logger.info(f"      {i}. {sq}")
        logger.info(f"   🚀 Use HyDE: {analysis.get('should_use_hyde')}")
        logger.info(f"   💡 Reasoning: {analysis.get('reasoning', 'N/A')}")
        logger.info("="*80)
        
        return json.dumps(analysis, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"❌ Query analysis failed: {e}")
        # Fallback: return simple analysis
        fallback = {
            "query_type": "other",
            "key_concepts": [query],
            "sub_queries": [query],
            "estimated_complexity": "medium",
            "should_use_hyde": False,
            "reasoning": f"Analysis failed, using original query. Error: {str(e)}"
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2)


@tool
def generate_hypothetical_answer(query: str) -> str:
    """
    [Optional - HyDE] 生成假想的理想答案文档，用于改善检索质量。
    适用于抽象/高层次的查询。生成的文档会被用于向量检索。
    
    Args:
        query: 子查询或原始查询
    
    Returns:
        假想的答案文档文本（会被 embedding 后用于检索）
    """
    logger.info("="*80)
    logger.info("🔮 [HyDE] Generating hypothetical answer document...")
    logger.info(f"📝 Query: {query}")
    logger.info("="*80)
    
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
        
        logger.success(f"✅ Generated hypothetical document ({len(content)} chars)")
        logger.info(f"📄 Preview: {content[:200]}...")
        logger.info("="*80)
        return content.strip()
    except Exception as e:
        logger.error(f"❌ HyDE generation failed: {e}")
        logger.warning("⚠️  Falling back to original query")
        return query  # Fallback to original query


@tool
def evaluate_retrieval_progress(original_query: str, current_results_summary: str, round_number: int) -> str:
    """
    [Self-Reflection] 评估当前检索结果是否充分，决定是否需要继续检索。
    
    Args:
        original_query: 用户的原始查询
        current_results_summary: 当前已检索结果的摘要（论文标题列表）
        round_number: 当前是第几轮检索（1-based）
    
    Returns:
        JSON格式的评估结果，包含：
        - is_sufficient: 是否已充分
        - coverage_score: 覆盖度评分 (0.0-1.0)
        - missing_aspects: 缺失的方面
        - should_continue: 是否应该继续检索
        - next_focus: 下一步应该关注什么
    """
    logger.info("="*80)
    logger.info(f"🔍 [SELF-REFLECTION] Evaluating retrieval progress - Round {round_number}")
    logger.info(f"📝 Original Query: {original_query}")
    logger.info(f"📊 Current Results Summary:")
    logger.info(current_results_summary[:500] + "..." if len(current_results_summary) > 500 else current_results_summary)
    logger.info("="*80)
    
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
        
        logger.success(f"✅ Evaluation completed - Round {round_number}")
        logger.info(f"   📊 Coverage Score: {evaluation.get('coverage_score'):.2f}/1.0")
        logger.info(f"   ✔️  Is Sufficient: {evaluation.get('is_sufficient')}")
        logger.info(f"   ➡️  Should Continue: {evaluation.get('should_continue')}")
        if evaluation.get('missing_aspects'):
            logger.warning(f"   ⚠️  Missing Aspects: {', '.join(evaluation.get('missing_aspects', []))}")
        if evaluation.get('next_focus'):
            logger.info(f"   🎯 Next Focus: {evaluation.get('next_focus')}")
        logger.info(f"   💭 Reasoning: {evaluation.get('reasoning', 'N/A')}")
        logger.info("="*80)
        
        return json.dumps(evaluation, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        # Fallback: stop after round 3
        fallback = {
            "is_sufficient": round_number >= 3,
            "coverage_score": 0.5,
            "missing_aspects": [],
            "should_continue": round_number < 3,
            "next_focus": "Continue with remaining sub-queries",
            "reasoning": f"Evaluation failed, using heuristic. Error: {str(e)}"
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2)


@tool
def rerank_results(original_query: str, results_json: str) -> str:
    """
    [Final Step] 使用 LLM 对检索结果进行相关性评分和重排序。
    
    Args:
        original_query: 用户的原始查询
        results_json: 检索结果的 JSON 字符串（包含 title, abstract, doc_id）
    
    Returns:
        重排序后的结果（JSON 格式），每个结果包含相关性分数
    """
    logger.info("="*80)
    logger.info("🏆 [PHASE 3: RERANKING] Starting LLM-based reranking...")
    logger.info(f"📝 Query: {original_query}")
    
    llm = _get_agentic_llm()
    
    try:
        results = json.loads(results_json)
    except:
        logger.error("❌ Failed to parse results JSON")
        return results_json  # Return as-is if parsing fails
    
    if not results:
        logger.warning("⚠️  No results to rerank")
        return results_json
    
    logger.info(f"📊 Input: {len(results)} papers to rerank")
    logger.info("="*80)
    
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
        
        logger.success(f"✅ Reranking completed!")
        logger.info(f"   📊 Final Results: {len(reranked)} papers (filtered from {len(results)})")
        logger.info(f"   🏆 Top 5 Papers by Relevance:")
        for i, r in enumerate(reranked[:5], 1):
            score = r.get("llm_relevance_score", 0)
            title = r.get("title", "Untitled")[:60]
            logger.info(f"      {i}. [{score:.1f}/10] {title}...")
        logger.info("="*80)
        
        return json.dumps(reranked, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"❌ Reranking failed: {e}")
        logger.warning("⚠️  Returning original results without reranking")
        return results_json  # Return original results


# ============== Phase 2: Abstract Search ==============

@tool
def search_abstracts(query: str, k: int = 5) -> str:
    """
    [Phase 2] 搜索论文摘要，找出相关论文。
    这是搜索的第一步，返回候选论文列表。
    
    Args:
        query: 搜索关键词或自然语言查询
        k: 返回论文数量 (默认 5)
    
    Returns:
        候选论文列表，包含 title, abstract 预览, doc_id
    """
    logger.info("="*80)
    logger.info("🔎 [PHASE 2: RETRIEVAL] Searching abstracts...")
    logger.info(f"📝 Query: {query}")
    logger.info(f"📊 Requested: top {k} papers")
    logger.info("="*80)
    
    client = _get_rag_client()
    results = client.search_abstracts(query, k)
    
    if not results:
        logger.warning("⚠️  No papers found matching the query")
        return "No papers found matching the query."
    
    logger.success(f"✅ Found {len(results)} papers")
    logger.info("📄 Top 3 Results:")
    for i, r in enumerate(results[:3], 1):
        title = r.get('title', 'Untitled')[:60]
        logger.info(f"   {i}. {title}... (doc_id: {r.get('doc_id', 'N/A')[:8]}...)")
    logger.info("="*80)
    
    output = []
    for i, r in enumerate(results, 1):
        abstract = r.get("abstract", "")[:300] + "..." if len(r.get("abstract", "")) > 300 else r.get("abstract", "")
        output.append(
            f"[{i}] {r.get('title', 'Untitled')}\n"
            f"    doc_id: {r.get('doc_id', 'N/A')}\n"
            f"    Abstract: {abstract}"
        )
    
    return "\n\n".join(output)


# ============== Phase 3: Lazy Load PDF ==============

@tool
def load_paper_pdfs(doc_ids: List[str]) -> str:
    """
    [Phase 3] 加载指定论文的 PDF 内容到数据库。
    在使用 search_paper_content 之前必须调用此工具！
    会自动跳过已加载的论文。
    
    Args:
        doc_ids: 要加载的论文 doc_id 列表（从 search_abstracts 获取）
    
    Returns:
        加载状态报告
    
    注意：
        - 一次建议加载 3-5 篇论文，避免等待过长
        - 加载过程需要下载和解析 PDF，可能需要一些时间
    """
    loader = _get_pdf_loader()
    results = loader.load_papers(doc_ids)
    
    # 格式化输出
    output = ["PDF Loading Results:"]
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for doc_id, result in results.items():
        status_icon = {
            LoadStatus.SUCCESS: "✓",
            LoadStatus.ALREADY_EXISTS: "○",
            LoadStatus.DOWNLOAD_FAILED: "✗",
            LoadStatus.PARSE_FAILED: "✗",
            LoadStatus.NO_PDF_URL: "✗",
            LoadStatus.NOT_FOUND: "✗",
        }.get(result.status, "?")
        
        output.append(f"  {status_icon} {doc_id}: {result.message}")
        
        if result.status == LoadStatus.SUCCESS:
            success_count += 1
        elif result.status == LoadStatus.ALREADY_EXISTS:
            skip_count += 1
        else:
            fail_count += 1
    
    output.append(f"\nSummary: {success_count} loaded, {skip_count} skipped, {fail_count} failed")
    
    if success_count + skip_count > 0:
        output.append("\nYou can now use search_paper_content to search within these papers.")
    
    return "\n".join(output)


# ============== Phase 4: Deep Search ==============

@tool  
def search_paper_content(query: str, doc_ids: List[str] = [], category: int = -1, k: int = 5) -> str:
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
        匹配的文本片段及其元数据
    """
    client = _get_rag_client()
    
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
        return "No matching content found. Make sure you have loaded the papers first using load_paper_pdfs."
    
    from parser.pdf_parser import SectionCategory
    
    output = []
    for i, r in enumerate(results, 1):
        cat_id = r.get("section_category", 0)
        try:
            cat_name = SectionCategory(cat_id).name
        except:
            cat_name = "UNKNOWN"
        
        text = r.get("text", "")[:500] + "..." if len(r.get("text", "")) > 500 else r.get("text", "")
        
        output.append(
            f"[{i}] doc_id: {r.get('doc_id', 'N/A')} | chunk_id: {r.get('chunk_id', 'N/A')}\n"
            f"    Section: {cat_name} | Parent: {r.get('parent_section', 'N/A')}\n"
            f"    Text: {text}"
        )
    
    return "\n\n".join(output)


# ============== Context Tools ==============

@tool
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
        
        if settings.enable_agentic_rag:
            self.llm = get_llm_by_usage('agentic')
            self._setup_agent()

    def _setup_agent(self):
        """Setup the LangChain agent with Agentic Retrieval tools."""
        self.tools = [
            # Phase 1: Agentic Retrieval
            analyze_query,
            generate_hypothetical_answer,
            evaluate_retrieval_progress,
            rerank_results,
            # Phase 2: Abstract search
            search_abstracts,
            # Phase 3: Lazy load PDF
            load_paper_pdfs,
            # Phase 4: Deep search
            search_paper_content,
            # Context tools
            get_context_window,
        ]
        
        # Load prompt as system message
        prompt_msgs = apply_prompt_template("agentic_searcher")
        self.system_prompt = prompt_msgs[0]["content"]

    def _agentic_search(self, query: str) -> Dict[str, Any]:
        """Run the agentic search loop."""
        logger.info("\n" + "🚀"*40)
        logger.info("🤖 AGENTIC RAG PIPELINE STARTED")
        logger.info(f"📝 User Query: {query}")
        logger.info("🚀"*40 + "\n")
        
        agent = create_agent(
            model=self.llm, 
            tools=self.tools,
        )
        
        # Build messages with system prompt and user query
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            result = agent.invoke({"messages": msgs}, config={"recursion_limit": 100})
            messages = result.get("messages", [])
            # Get the last AI message as the answer
            answer = ""
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    answer = msg.content
                    break
            
            logger.info("\n" + "✅"*40)
            logger.info("🎉 AGENTIC RAG PIPELINE COMPLETED")
            logger.info(f"📊 Total Messages: {len(messages)}")
            logger.info(f"📝 Answer Length: {len(answer)} chars")
            logger.info("✅"*40 + "\n")
            
            return {
                "answer": answer,
                "intermediate_steps": messages
            }
        except Exception as e:
            logger.error(f"❌ Agentic search failed: {e}")
            logger.error("="*80)
            return {"answer": f"Search failed: {e}", "intermediate_steps": []}

    def search(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        """Query vector store and return normalized hits with ids."""
        k = k or self.top_k
        
        if settings.enable_agentic_rag:
            # Agentic mode: return the agent's analysis
            result = self._agentic_search(query)
            # For compatibility, wrap the answer in a hit-like structure
            return [{
                "id": 1,
                "title": "Agentic Search Result",
                "abstract": result["answer"],
                "url": "",
                "doc_id": "agentic",
                "score": 1.0,
                "conference_name": "",
                "conference_year": "",
                "conference_round": "",
                "section_category": 0,
                "parent_section": "",
                "page_number": 0,
            }]
            
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
        
        # Check if this is an agentic result
        if len(hits) == 1 and hits[0].get("doc_id") == "agentic":
            return hits[0].get("abstract", "No answer generated.")
        
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
