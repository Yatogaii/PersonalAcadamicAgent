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
    def insert_document(self, title: str, abstract: str, url: str=''):
        raise NotImplementedError

    @abstractmethod
    def list_resources(self) -> list[str]:
        raise NotImplementedError