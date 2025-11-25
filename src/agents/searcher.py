from typing import Any, Dict, List

from logging_config import logger
from rag.retriever import get_rag_client_by_provider
from settings import settings


class Searcher:
    """RAG searcher that only retrieves and normalizes hits.

    Generation is handled by the coordinator to keep output format consistent.
    """

    def __init__(self) -> None:
        self.rag_client = get_rag_client_by_provider(settings.rag_provider)
        self.top_k = settings.milvus_top_k

    def search(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        """Query vector store and return normalized hits with ids."""
        k = k or self.top_k
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
