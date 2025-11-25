from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    metadata: dict
    score:float

class RAG(ABC):
    @abstractmethod
    def query_relevant_documents(self, query: str):
        raise NotImplementedError

    @abstractmethod
    def insert_document(self, title: str, abstract: str, url: str='', pdf_url: str='', conference_name: str='', conference_year: int=0, conference_round: str='all'):
        raise NotImplementedError

    @abstractmethod
    def list_resources(self) -> list[str]:
        raise NotImplementedError
    
    @abstractmethod
    def check_conference_exists(self, conference_name: str, year: int, round: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_existing_rounds(self, conference_name: str, year: int) -> list[str]:
        """Return a list of rounds that exist in the DB for the given conference and year."""
        raise NotImplementedError

    @abstractmethod
    def get_conference_papers(self, conference_name: str, year: int, round: str, limit: int = 10) -> list[Chunk]:
        """Return up to `limit` papers for a given conference/year/round.

        Implementations should return a list of Chunk objects where:
        - content: usually abstract or main content
        - metadata: should at least include title and url if available
        - score: can be 0 or any placeholder when not ranked
        """
        raise NotImplementedError

    @abstractmethod
    def insert_paper_chunks(self, doc_id: str, chunks: list[dict], paper_title: str = ""):
        """
        Inserts parsed chunks into the vector store.
        chunks: List of dicts with structure metadata (chunk_index, section_category, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def get_context_window(self, doc_id: str, center_chunk_index: int, window_size: int = 1) -> str:
        """
        Retrieves the context window around a specific chunk.
        Returns the concatenated text of chunks in [center - window, center + window].
        """
        raise NotImplementedError

_rag_clients: dict[str, RAG] = {}

def get_rag_client_by_provider(provider: str) -> RAG:
    if provider not in _rag_clients:
        if provider == 'milvus':
            from .milvus import MilvusProvider
            _rag_clients[provider] = MilvusProvider()
        elif provider == 'pgvector':
            from .pgvector import PGVectorProvider
            _rag_clients[provider] = PGVectorProvider()
        else:
            raise ValueError(f"Unsupported RAG provider: {provider}")
    return _rag_clients[provider]