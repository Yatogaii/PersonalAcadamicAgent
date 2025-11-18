from pymilvus import MilvusClient, FieldSchema, DataType
from rag.retriever import RAG, Chunk
from rag.feature_extractor import FeatureExtractor
from uuid import uuid4
from settings import settings

from langchain.embeddings import init_embeddings

"""
Milvus(lite) Implementation for RAG.
"""
class MilvusProvider(RAG):
    def __init__(self) -> None:
        super().__init__()
        # --- Connection / collection configuration ---
        self.uri: str = settings.milvus_uri
        self.user: str = settings.milvus_user
        self.password: str = settings.milvus_password
        self.collection_name: str = settings.milvus_collection

        # --- Search configuration ---
        self.top_k: int = settings.milvus_top_k

        # --- Vector field names ---
        self.id_field: str = settings.milvus_id_field
        self.doc_id_field: str = settings.milvus_doc_id_field
        self.doc_vector_field: str = settings.milvus_doc_vector_field
        self.title_field: str = settings.milvus_title_field
        self.abstract_field: str = settings.milvus_abstract_field
        self.content_field: str = settings.milvus_content_field
        self.url_field: str = settings.milvus_url_field
        self.chunk_id_field: str = settings.milvus_chunk_id_field
        self.vector_field: str = settings.milvus_vector_field
        self.conference_name_field: str = settings.milvus_conference_name_field
        self.conference_year_field: str = settings.milvus_conference_year_field
        self.conference_round_field: str = settings.milvus_conference_round_field

        # --- Vector index configuration ---
        self.vector_index_metric_type: str = settings.milvus_vector_index_metric_type

        # --- Embedding model configuration ---
        self.embedding_model = settings.embedding_model
        self.embedding_model_base_url = settings.embedding_model_base_url
        self.embedding_model_api_key = settings.embedding_model_api_key
        self.dim = settings.embedding_dim

        # --- Milvus Client ---
        self.client = self._get_client()

        # --- Feature Extractor ---
        self.embedding_client = FeatureExtractor(
            provider="huggingface",
            api_key=settings.HF_TOKEN,
            model=settings.embedding_model
        )
        
    def _create_schema(self):
        # Define schema for Milvus collection
        schema = MilvusClient.create_schema(fields=[
            FieldSchema(name=self.id_field, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=self.doc_id_field, dtype=DataType.VARCHAR),
            FieldSchema(name=self.doc_vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.dim, nullable=True),
            FieldSchema(name=self.title_field, dtype=DataType.VARCHAR),
            FieldSchema(name=self.abstract_field, dtype=DataType.VARCHAR),
            FieldSchema(name=self.content_field, dtype=DataType.VARCHAR, nullable=True),
            FieldSchema(name=self.url_field, dtype=DataType.VARCHAR, nullable=True),
            FieldSchema(name=self.chunk_id_field, dtype=DataType.INT64, nullable=True),
            FieldSchema(name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.dim, nullable=True),
            FieldSchema(name=self.conference_name_field, dtype=DataType.VARCHAR),
            FieldSchema(name=self.conference_year_field, dtype=DataType.INT64),
            FieldSchema(name=self.conference_round_field, dtype=DataType.VARCHAR),
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
            
            index_params.add_index(
                field_name=self.conference_name_field,
                index_type="bitmap",
                metric_type=self.vector_index_metric_type,
                index_name="conference_name_index",
            )

            index_params.add_index(
                field_name=self.conference_year_field,
                index_type="bitmap",
                metric_type=self.vector_index_metric_type,
                index_name="conference_year_index",
            )

            index_params.add_index(
                field_name=self.conference_round_field,
                index_type="bitmap",
                metric_type=self.vector_index_metric_type,
                index_name="conference_round_index",
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )

    def query_relevant_documents(self, query: str):
        raise NotImplementedError("MilvusProvider.query_relevant_documents not implemented yet.")

        
    # Initially insert one paper to database.
    def insert_document(self, title: str, abstract: str, url: str = ''):
        '''
        We insert pdf vector to milvus lazily.
        For the first time we saw a pdf, we just insert title, abstract, url_of_pdf to database.
        '''
        doc_vector  = self.embedding_client.embed_query(f"Title: {title}\nAbstract: {abstract}")
        data = {
            "doc_id": uuid4(),
            "doc_vectors": doc_vector,
            "title": title,
            "abstract": abstract,
            "url": url,
        }

        self.client.insert(collection_name=self.collection_name, data=data)
    
    def insert_document_content(self, doc_id, title, abstract, content): 
        raise RuntimeError("Chunk strategy not implement yet!")

    def list_resources(self) -> list[str]:
        return ["Milvus Collection: " + self.collection_name]

    def check_conference_exists(self, conference_name: str, year: int, round: str) -> bool:
        # Check if documents from the specified conference, year, and round exist
        query = f'{self.conference_name_field} == "{conference_name}" && {self.conference_year_field} == {year} && {self.conference_round_field} == "{round}"'
        results = self.client.query(
            collection_name=self.collection_name,
            expr=query,
            output_fields=[self.id_field],
            limit=1
        )
        return len(results) > 0