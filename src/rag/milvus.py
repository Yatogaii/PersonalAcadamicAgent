from pymilvus import MilvusClient, FieldSchema, DataType
import os
from rag.retriever import RAG, Chunk
from uuid import uuid4

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
        self.id_field: str = os.getenv("MILVUS_ID_FIELD", "id")
        self.doc_id_field: str = os.getenv("MILVUS_DOC_ID_FIELD", "doc_id")
        self.title_field: str = os.getenv("MILVUS_TITLE_FIELD", "title")
        self.abstract_field: str = os.getenv("MILVUS_ABSTRACT_FIELD", "abstract")
        self.content_field: str = os.getenv("MILVUS_CONTENT_FIELD", "content")
        self.url_field: str = os.getenv("MILVUS_URL_FIELD", "url")
        self.chunk_id_field: str = os.getenv("MILVUS_CHUNK_ID_FIELD", "chunk_id")
        self.vector_field: str = os.getenv("MILVUS_VECTOR_FIELD", "vectors")

        # --- Vector index configuration ---
        self.vector_index_metric_type: str = os.getenv("MILVUS_VECTOR_INDEX_METRIC_TYPE", "L2")

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
            FieldSchema(name=self.doc_id_field, dtype=DataType.VARCHAR),
            FieldSchema(name=self.title_field, dtype=DataType.VARCHAR),
            FieldSchema(name=self.abstract_field, dtype=DataType.VARCHAR),
            FieldSchema(name=self.content_field, dtype=DataType.VARCHAR, nullable=True),
            FieldSchema(name=self.url_field, dtype=DataType.VARCHAR, nullable=True),
            FieldSchema(name=self.chunk_id_field, dtype=DataType.INT64, nullable=True),
            FieldSchema(name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.dim, nullable=True),
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
                metric_type=self.vector_index_metric_type,
                index_name="vector_index",
                params = {"nlist": 1024},
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )

    def query_relevant_documents(self, query: str):
        return super().query_relevant_documents(query)

        
    # Initially insert one paper to database.
    def insert_document(self, title: str, abstract: str, url: str = ''):
        '''
        We insert pdf vector to milvus lazily.
        For the first time we saw a pdf, we just insert title, abstract, url_of_pdf to database.
        '''
        data = {
            "doc_id": uuid4(),
            "title": title,
            "abstract": abstract,
            "url": url,
        }

        self.client.insert(collection_name=self.collection_name, data=data)
    
    def insert_document_content(self, doc_id, title, abstract, content): 
        raise RuntimeError("Chunk strategy not implement yet!")