from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
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

        # Doc-level collection (abstract/title embedding)
        self.doc_collection: str = settings.milvus_doc_collection
        # Chunk-level collection (future use for content chunks)
        self.chunk_collection: str = settings.milvus_chunk_collection

        # --- Search configuration ---
        self.top_k: int = settings.milvus_top_k
        # --- Field names for doc-level collection ---
        self.id_field: str = settings.milvus_id_field
        self.doc_id_field: str = settings.milvus_doc_id_field
        self.doc_vector_field: str = settings.milvus_doc_vector_field
        self.title_field: str = settings.milvus_title_field
        self.abstract_field: str = settings.milvus_abstract_field
        self.url_field: str = settings.milvus_url_field
        self.conference_name_field: str = settings.milvus_conference_name_field
        self.conference_year_field: str = settings.milvus_conference_year_field
        self.conference_round_field: str = settings.milvus_conference_round_field

        # --- Field names for chunk-level collection (future use) ---
        self.content_field: str = settings.milvus_content_field
        self.chunk_id_field: str = settings.milvus_chunk_id_field
        self.vector_field: str = settings.milvus_vector_field

        # --- Vector index configuration ---
        self.vector_index_metric_type: str = settings.milvus_vector_index_metric_type

        # --- Embedding model configuration ---
        self.embedding_model = settings.embedding_model
        self.embedding_model_base_url = settings.embedding_model_base_url
        self.embedding_model_api_key = settings.embedding_model_api_key
        self.dim = settings.embedding_dim

        # --- Milvus Client ---
        self._get_client()

        # --- Feature Extractor ---
        self.embedding_client = FeatureExtractor(
            provider=settings.embedding_provider,
            api_key=settings.HF_TOKEN,
            model=settings.embedding_model
        )
        
    def _create_doc_schema(self) -> CollectionSchema:
        """Schema for doc-level collection: one row per paper (title+abstract embedding)."""
        return CollectionSchema(
            fields=[
                FieldSchema(name=self.id_field, dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name=self.doc_id_field, dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name=self.doc_vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name=self.title_field, dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name=self.abstract_field, dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name=self.url_field, dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name=self.conference_name_field, dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name=self.conference_year_field, dtype=DataType.INT64),
                FieldSchema(name=self.conference_round_field, dtype=DataType.VARCHAR, max_length=64),
            ]
        )

    def _create_chunk_schema(self) -> CollectionSchema:
        """Schema for chunk-level collection: future use for content chunks.

        暂时只定义好结构，后续再实现 insert/query 逻辑。
        """
        return CollectionSchema(
            fields=[
                FieldSchema(name=self.id_field, dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name=self.doc_id_field, dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name=self.chunk_id_field, dtype=DataType.INT64),
                FieldSchema(name=self.content_field, dtype=DataType.VARCHAR, max_length=16384),
                FieldSchema(name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
        )

    def _get_client(self):
        if not self.uri.startswith("http"):
            self.client = MilvusClient(uri=self.uri)
            # Ensure both doc-level and chunk-level collections exist
            self._ensure_doc_collection_exists(self.doc_collection)
            self._ensure_chunk_collection_exists(self.chunk_collection)
        else:
            raise RuntimeError("Milvus HTTP URI is not supported in this implementation.")
        return self.client

    def _ensure_doc_collection_exists(self, collection_name: str) -> None:
        """Create doc-level collection and indexes if missing."""
        if not self.client.has_collection(collection_name):
            schema = self._create_doc_schema()

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=self.doc_vector_field,
                index_type="FLAT",
                metric_type=self.vector_index_metric_type,
                index_name="doc_vector_index",
                params={"nlist": 1024},
            )
            index_params.add_index(
                field_name=self.conference_name_field,
                index_type="FLAT",
                index_name="conference_name_index",
            )
            index_params.add_index(
                field_name=self.conference_year_field,
                index_type="FLAT",
                index_name="conference_year_index",
            )
            index_params.add_index(
                field_name=self.conference_round_field,
                index_type="FLAT",
                index_name="conference_round_index",
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

    def _ensure_chunk_collection_exists(self, collection_name: str) -> None:
        """Create chunk-level collection and indexes if missing.

        目前只建表和索引，具体写入/查询逻辑以后再实现。
        """
        if not self.client.has_collection(collection_name):
            schema = self._create_chunk_schema()

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=self.vector_field,
                index_type="FLAT",
                metric_type=self.vector_index_metric_type,
                index_name="chunk_vector_index",
                params={"nlist": 1024},
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

    def query_relevant_documents(self, query: str):
        raise NotImplementedError("MilvusProvider.query_relevant_documents not implemented yet.")

        
    # Initially insert one paper to database.
    def insert_document(self, title: str, abstract: str, url: str = '', conference_name: str='', conference_year: int=0, conference_round: str='all'):
        '''
        We insert pdf vector to milvus lazily.
        For the first time we saw a pdf, we just insert title, abstract, url_of_pdf to database.
        '''
        doc_vector  = self.embedding_client.embed_query(f"Title: {title}\nAbstract: {abstract}")
        data = {
            "doc_id": str(uuid4()),
            "doc_vectors": doc_vector,
            "title": title,
            "abstract": abstract,
            "url": url,
            "conference_name": conference_name,
            "conference_year": conference_year,
            "conference_round": conference_round,
        }

        self.client.insert(collection_name=self.doc_collection, data=data)
    
    def insert_document_content(self, doc_id, title, abstract, content): 
        raise RuntimeError("Chunk strategy not implement yet!")

    def list_resources(self) -> list[str]:
        return [
            "Milvus Doc Collection: " + self.doc_collection,
            "Milvus Chunk Collection: " + self.chunk_collection,
        ]

    def check_conference_exists(self, conference_name: str, year: int, round: str) -> bool:
        # Check if documents from the specified conference, year, and round exist
        query = f'{self.conference_name_field} == "{conference_name}" && {self.conference_year_field} == {year} && {self.conference_round_field} == "{round}"'
        results = self.client.query(
            collection_name=self.doc_collection,
            filter=query,
            output_fields=[self.id_field],
            limit=1
        )
        return len(results) > 0

    def get_conference_papers(self, conference_name: str, year: int, round: str, limit: int = 10) -> list[Chunk]:
        """Fetch up to `limit` papers for a given conference/year/round from Milvus.

        This assumes that documents were inserted with title, abstract and url fields populated,
        and that they are tagged with conference_name/year/round fields.
        """
        query = f'{self.conference_name_field} == "{conference_name}" && {self.conference_year_field} == {year} && {self.conference_round_field} == "{round}"'
        results = self.client.query(
            collection_name=self.doc_collection,
            filter=query,
            output_fields=[
                self.title_field,
                self.abstract_field,
                self.url_field,
            ],
            limit=limit,
        )

        chunks: list[Chunk] = []
        for r in results:
            title = r.get(self.title_field, "")
            abstract = r.get(self.abstract_field, "")
            url = r.get(self.url_field, "")
            content = f"Title: {title}\nAbstract: {abstract}"
            metadata = {
                "title": title,
                "abstract": abstract,
                "url": url,
                "conference_name": conference_name,
                "conference_year": year,
                "conference_round": round,
            }
            chunks.append(Chunk(content=content, metadata=metadata, score=0.0))

        return chunks