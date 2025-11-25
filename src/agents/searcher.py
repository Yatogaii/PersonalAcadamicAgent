from typing import Any, Dict, List

from logging_config import logger
from rag.retriever import get_rag_client_by_provider
from settings import settings
from models import init_kimi_k2
from prompts.template import apply_prompt_template
from langchain.tools import tool
from langchain.agents import create_agent

# Global RAG client for tools
_rag_client = None

def _get_rag_client():
    global _rag_client
    if _rag_client is None:
        _rag_client = get_rag_client_by_provider(settings.rag_provider)
    return _rag_client

# ============== Agentic RAG Tools ==============

@tool
def search_abstracts(query: str, k: int = 5) -> str:
    """
    Search paper abstracts to identify relevant papers.
    Use this FIRST to get an overview of relevant papers in the database.
    
    Args:
        query: The search query (keywords or natural language)
        k: Number of papers to return (default 5)
    
    Returns:
        List of papers with title, abstract preview, and doc_id
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


@tool  
def search_by_section(query: str, doc_id: str = "", category: int = -1, k: int = 5) -> str:
    """
    Search within specific sections or papers for detailed information.
    
    Args:
        query: The search query
        doc_id: Limit search to a specific paper (use doc_id from search_abstracts). Leave empty for all papers.
        category: Filter by section type. Use -1 for all sections.
            0 = Abstract
            1 = Introduction (background, motivation)
            2 = Method (technical details, algorithms, architecture)
            3 = Evaluation (experiments, results, performance numbers)
            4 = Conclusion
            6 = Related Work (describes OTHER papers!)
        k: Number of chunks to return (default 5)
    
    Returns:
        List of text chunks with metadata
    """
    client = _get_rag_client()
    
    # Handle empty string as None
    actual_doc_id = doc_id if doc_id else None
    actual_category = category if category >= 0 else None
    
    results = client.search_by_section(query, actual_doc_id, actual_category, k)
    
    if not results:
        return "No matching sections found."
    
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


@tool
def get_context_window(doc_id: str, chunk_id: int, window: int = 1) -> str:
    """
    Get surrounding text around a specific chunk for more context.
    Use when a retrieved snippet seems incomplete or truncated.
    
    Args:
        doc_id: The paper's doc_id
        chunk_id: The chunk_id from search_by_section results
        window: Number of chunks before and after to include (default 1)
    
    Returns:
        Extended text including surrounding chunks
    """
    client = _get_rag_client()
    context = client.get_context_window(doc_id, chunk_id, window)
    
    if not context:
        return "Could not retrieve context for this chunk."
    
    return f"[Context Window for doc_id={doc_id}, chunk_id={chunk_id}]\n\n{context}"


@tool
def get_paper_introduction(doc_id: str) -> str:
    """
    Get the Introduction section of a specific paper.
    Use to understand the paper's background, motivation, and problem statement.
    
    Args:
        doc_id: The paper's doc_id
    
    Returns:
        The Introduction text (truncated if too long)
    """
    client = _get_rag_client()
    intro = client.get_paper_introduction(doc_id)
    
    if not intro:
        return f"No Introduction found for doc_id={doc_id}"
    
    return f"[Introduction for doc_id={doc_id}]\n\n{intro}"


# ============== Searcher Class ==============

class Searcher:
    """RAG searcher that supports both simple and agentic modes.

    - Simple mode: Direct vector search
    - Agentic mode: LLM agent with multiple tools for iterative retrieval
    """

    def __init__(self) -> None:
        self.rag_client = get_rag_client_by_provider(settings.rag_provider)
        self.top_k = settings.milvus_top_k
        
        if settings.enable_agentic_rag:
            self.llm = init_kimi_k2()
            self._setup_agent()

    def _setup_agent(self):
        """Setup the LangChain agent with tools."""
        self.tools = [
            search_abstracts,
            search_by_section,
            get_context_window,
            get_paper_introduction,
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
