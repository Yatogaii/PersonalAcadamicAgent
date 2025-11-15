from pymilvus import MilvusClient, FieldSchema, DataType
import os
from rag.retriever import RAG

from langchain.embeddings import init_embeddings

"""
Milvus(lite) Implementation for RAG.
"""
class MilvusProvider(RAG):
    def __init__(self) -> None:
        super().__init__()
        # --- Connection / collection configuration ---
        self.uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
        self.user: str = os.getenv("MILVUS_USER", "default")
        self.password: str = os.getenv("MILVUS_PASSWORD", "123456")
        self.collection_name: str = os.getenv("MILVUS_COLLECTION", "documents")

        # --- Search configuration ---
        top_k_raw = os.getenv("MILVUS_TOP_K", "10")
        self.top_k: int = int(top_k_raw) if top_k_raw.isdigit() else 10

        # --- Vector field names ---
        self.vector_field: str = os.getenv("MILVUS_VECTOR_FIELD", "embedding")
        self.id_field: str = os.getenv("MILVUS_ID_FIELD", "id")
        self.abstract_field: str = os.getenv("MILVUS_ABSTRACT_FIELD", "abstract")
        self.title_field: str = os.getenv("MILVUS_TITLE_FIELD", "title")
        self.url_field: str = os.getenv("MILVUS_URL_FIELD", "url")
        self.metadata_field: str = os.getenv("MILVUS_METADATA_FIELD", "metadata")

        # --- Embedding model configuration ---
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.embedding_model_base_url = os.getenv("EMBEDDING_MODEL_BASE_URL", "https://huggingface.co/")
        self.embedding_model_api_key = os.getenv("EMBEDDING_MODEL_API_KEY", "")
        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME", "documents")
        self.dim = int(os.getenv("EMBEDDING_DIM", "2048"))

        # --- Milvus Client (Lazy Load) ---
        self.client = self._get_client()
        
    def _create_schema(self):
        # Define schema for Milvus collection
        schema = MilvusClient.create_schema(fields=[
            FieldSchema(name=self.id_field, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=self.abstract_field, dtype=DataType.VARCHAR),
            FieldSchema(name=self.title_field, dtype=DataType.VARCHAR),
            FieldSchema(name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ])
        return schema

    def _get_client(self):
        if not self.uri.startswith("http"):
            client = MilvusClient(uri=self.uri)
            self._ensure_collection_exists(self.collection_name)
        else:
            raise RuntimeError("Milvus HTTP URI is not supported in this implementation.")
        return client

    def _ensure_collection_exists(self, collection_name):
        if not self.client.has_collection(collection_name):
            # Create Collection Schema
            schema = self._create_schema()

            # Create Index
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=self.id_field,
                index_type="AUTOINDEX",
            )

            index_params.add_index(
                field_name=self.vector_field,
                index_type="IVF_FLAT",
                metric_type="L2",
                params = {"nlist": 1024},
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )

    def query_relevant_documents(self, query: str):
        return super().query_relevant_documents(query)

        
    # Insert one paper to database.
    def insert_documents(self, title: str, abstract: str, url: str = ''):
        return super().insert_documents(title, abstract, url)