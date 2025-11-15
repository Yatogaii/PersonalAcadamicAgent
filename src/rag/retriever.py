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
        pass

    @abstractmethod
    def insert_documents(self, title: str, abstract: str, url: str=''):
        pass

    @abstractmethod
    def list_resources(self):
        pass