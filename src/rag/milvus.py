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
        self.token: str = settings.milvus_token
        self.collection: str = settings.milvus_collection

        # --- Search configuration ---
        self.top_k: int = settings.milvus_top_k
        
        # --- Field names ---
        self.id_field: str = settings.milvus_id_field
        self.doc_id_field: str = settings.milvus_doc_id_field
        self.vector_field: str = settings.milvus_vector_field
        self.text_field: str = settings.milvus_text_field
        self.title_field: str = settings.milvus_title_field
        self.url_field: str = settings.milvus_url_field
        self.pdf_url_field: str = settings.milvus_pdf_url_field
        self.conference_name_field: str = settings.milvus_conference_name_field
        self.conference_year_field: str = settings.milvus_conference_year_field
        self.conference_round_field: str = settings.milvus_conference_round_field
        self.chunk_id_field: str = settings.milvus_chunk_id_field

        # --- Vector index configuration ---
        self.vector_index_metric_type: str = settings.milvus_vector_index_metric_type
        self.index_type: str = settings.milvus_index_type
        self.index_params: dict = settings.milvus_index_params
        self.search_params: dict = settings.milvus_search_params

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
        
    def _create_schema(self) -> CollectionSchema:
        """Schema for unified collection (papers + chunks)."""
        return CollectionSchema(
            fields=[
                FieldSchema(name=self.id_field, dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name=self.doc_id_field, dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name=self.vector_field, dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name=self.text_field, dtype=DataType.VARCHAR, max_length=16384), # Abstract or Content
                
                # Optional fields (nullable=True)
                FieldSchema(name=self.title_field, dtype=DataType.VARCHAR, max_length=512, nullable=True),
                FieldSchema(name=self.url_field, dtype=DataType.VARCHAR, max_length=2048, nullable=True),
                FieldSchema(name=self.pdf_url_field, dtype=DataType.VARCHAR, max_length=2048, nullable=True),
                FieldSchema(name=self.conference_name_field, dtype=DataType.VARCHAR, max_length=256, nullable=True),
                FieldSchema(name=self.conference_year_field, dtype=DataType.INT64, nullable=True),
                FieldSchema(name=self.conference_round_field, dtype=DataType.VARCHAR, max_length=64, nullable=True),
                FieldSchema(name=self.chunk_id_field, dtype=DataType.INT64, nullable=True),
            ]
        )

    def _get_client(self):
        self.client = MilvusClient(uri=self.uri, token=self.token)
        self._ensure_collection_exists(self.collection)
        return self.client

    def _ensure_collection_exists(self, collection_name: str) -> None:
        """Create collection and indexes if missing."""
        if not self.client.has_collection(collection_name):
            schema = self._create_schema()

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=self.vector_field,
                index_type=self.index_type,
                metric_type=self.vector_index_metric_type,
                index_name="vector_index",
                params=self.index_params,
            )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

    def query_relevant_documents(self, query: str):
        res = [] 
        # Just return an basic search result for now.
        # Use two level search
        milvus_res = self.client.search(
                                        collection_name=self.collection,
                                        data=[self.embedding_client.embed_query(query)],
                                        search_params=self.search_params,
                                        output_fields=[
                                            self.title_field,
                                            self.text_field,
                                            self.doc_id_field,
                                            self.url_field,
                                            self.pdf_url_field,
                                            self.conference_name_field,
                                            self.conference_year_field,
                                            self.conference_round_field,
                                        ])
        for each_entity in milvus_res[0]:
            res.append({
                "title": each_entity[self.title_field],
                "abstract": each_entity[self.text_field],
                "doc_id": each_entity[self.doc_id_field],
                "url": each_entity.get(self.url_field, ""),
                "pdf_url": each_entity.get(self.pdf_url_field, ""),
                "conference_name": each_entity[self.conference_name_field],
                "conference_year": each_entity[self.conference_year_field],
                "conference_round": each_entity[self.conference_round_field],
            })

        return res

    # Initially insert one paper to database.
    def insert_document(self, title: str, abstract: str, url: str = '', pdf_url: str = '', conference_name: str='', conference_year: int=0, conference_round: str='all'):
        '''
        We insert pdf vector to milvus lazily.
        For the first time we saw a pdf, we just insert title, abstract, url_of_pdf to database.
        '''
        doc_vector  = self.embedding_client.embed_query(f"Title: {title}\nAbstract: {abstract}")
        data = {
            self.doc_id_field: str(uuid4()),
            self.vector_field: doc_vector,
            self.title_field: title,
            self.text_field: abstract,
            self.url_field: url,
            self.pdf_url_field: pdf_url,
            self.conference_name_field: conference_name,
            self.conference_year_field: conference_year,
            self.conference_round_field: conference_round,
        }

        self.client.insert(collection_name=self.collection, data=data)
    
    def insert_document_content(self, doc_id, title, abstract, content): 
        raise RuntimeError("Chunk strategy not implement yet!")

    def list_resources(self) -> list[str]:
        return [
            "Milvus Collection: " + self.collection,
        ]

    def check_conference_exists(self, conference_name: str, year: int, round: str) -> bool:
        # Check if documents from the specified conference, year, and round exist
        query = f'{self.conference_name_field} == "{conference_name}" && {self.conference_year_field} == {year} && {self.conference_round_field} == "{round}"'
        results = self.client.query(
            collection_name=self.collection,
            filter=query,
            output_fields=[self.id_field],
            limit=1
        )
        return len(results) > 0

    def get_existing_rounds(self, conference_name: str, year: int) -> list[str]:
        """Get list of rounds that already exist for a conference and year."""
        query = f'{self.conference_name_field} == "{conference_name}" && {self.conference_year_field} == {year}'
        results = self.client.query(
            collection_name=self.collection,
            filter=query,
            output_fields=[self.conference_round_field],
            limit=1000  # Fetch enough to cover all rounds
        )
        rounds = set()
        for res in results:
            r = res.get(self.conference_round_field)
            if r:
                rounds.add(r)
        return list(rounds)

    def get_conference_papers(self, conference_name: str, year: int, round: str, limit: int = 10) -> list[Chunk]:
        """Fetch up to `limit` papers for a given conference/year/round from Milvus.

        This assumes that documents were inserted with title, abstract and url fields populated,
        and that they are tagged with conference_name/year/round fields.
        """
        query = f'{self.conference_name_field} == "{conference_name}" && {self.conference_year_field} == {year} && {self.conference_round_field} == "{round}"'
        results = self.client.query(
            collection_name=self.collection,
            filter=query,
            output_fields=[
                self.title_field,
                self.text_field,
                self.url_field,
                self.pdf_url_field,
            ],
            limit=limit,
        )

        chunks: list[Chunk] = []
        for r in results:
            title = r.get(self.title_field, "")
            abstract = r.get(self.text_field, "")
            url = r.get(self.url_field, "")
            pdf_url = r.get(self.pdf_url_field, "")
            content = f"Title: {title}\nAbstract: {abstract}"
            metadata = {
                "title": title,
                "abstract": abstract,
                "url": url,
                "pdf_url": pdf_url,
                "conference_name": conference_name,
                "conference_year": year,
                "conference_round": round,
            }
            chunks.append(Chunk(content=content, metadata=metadata, score=0.0))

        return chunks