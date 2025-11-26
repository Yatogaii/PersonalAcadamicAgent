from typing import Any, Dict, List

from logging_config import logger
from rag.retriever import get_rag_client_by_provider
from rag.pdf_loader import PDFLoader, LoadStatus
from settings import settings
from models import init_kimi_k2, init_ollama_model
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
            _pdf_loader = PDFLoader(_get_rag_client(), llm_client=init_ollama_model())
        else:
            _pdf_loader = PDFLoader(_get_rag_client())
    return _pdf_loader


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
    client = _get_rag_client()
    results = client.search_abstracts(query, k)
    
    if not results:
        return "No papers found matching the query."
    
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
            self.llm = init_kimi_k2()
            self._setup_agent()

    def _setup_agent(self):
        """Setup the LangChain agent with Lazy Load tools."""
        self.tools = [
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
            result = agent.invoke({"messages": msgs})
            messages = result.get("messages", [])
            # Get the last AI message as the answer
            answer = ""
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    answer = msg.content
                    break
            
            return {
                "answer": answer,
                "intermediate_steps": messages
            }
        except Exception as e:
            logger.error(f"Agentic search failed: {e}")
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
