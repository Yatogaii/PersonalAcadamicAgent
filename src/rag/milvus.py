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
        
        # --- Structure-Aware RAG Fields ---
        # self.chunk_id_field is used for the sequential index (0, 1, 2...)
        self.section_category_field: str = settings.milvus_section_category_field
        self.parent_section_field: str = settings.milvus_parent_section_field
        self.page_number_field: str = settings.milvus_page_number_field

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
                
                # Structure-Aware RAG Fields
                FieldSchema(name=self.chunk_id_field, dtype=DataType.INT64, nullable=True), # Sequential index (-1 for paper level)
                FieldSchema(name=self.section_category_field, dtype=DataType.INT64, nullable=True),
                FieldSchema(name=self.parent_section_field, dtype=DataType.VARCHAR, max_length=512, nullable=True),
                FieldSchema(name=self.page_number_field, dtype=DataType.INT64, nullable=True),
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
    def insert_document(self, title: str, abstract: str, url: str = '', pdf_url: str = '', conference_name: str='', conference_year: int=0, conference_round: str='all') -> str:
        '''
        We insert pdf vector to milvus lazily.
        For the first time we saw a pdf, we just insert title, abstract, url_of_pdf to database.
        Returns the generated doc_id.
        '''
        doc_vector  = self.embedding_client.embed_query(f"Title: {title}\nAbstract: {abstract}")
        doc_id = str(uuid4())
        data = {
            self.doc_id_field: doc_id,
            self.vector_field: doc_vector,
            self.title_field: title,
            self.text_field: abstract,
            self.url_field: url,
            self.pdf_url_field: pdf_url,
            self.conference_name_field: conference_name,
            self.conference_year_field: conference_year,
            self.conference_round_field: conference_round,
            # Set default values for structure fields for paper-level entry
            self.chunk_id_field: -1,
            self.section_category_field: 0, # Abstract category is 0
            self.parent_section_field: "",
            self.page_number_field: 1
        }

        self.client.insert(collection_name=self.collection, data=data)
        return doc_id
    
    def insert_paper_chunks(self, doc_id: str, chunks: list[dict], paper_title: str = ""):
        """
        Inserts parsed chunks into Milvus.
        chunks: List of dicts from pdf_parser.flatten_pdf_tree
        """
        if not chunks:
            return

        print(f"Inserting {len(chunks)} chunks for doc_id: {doc_id}")
        
        # Prepare data for insertion
        data_list = []
        
        # Batch embedding could be optimized here, but for now we do one by one or small batches
        # Let's assume embed_query can handle single strings. 
        # If we want batching, we should check FeatureExtractor.
        
        for chunk in chunks:
            text = chunk["text"]
            # Create a rich representation for embedding
            # "Title: {Paper Title}\nSection: {Section Title}\nContent: {Text}"
            embed_text = f"Title: {paper_title}\nSection: {chunk['section_title']}\nContent: {text}"
            vector = self.embedding_client.embed_query(embed_text)
            
            entry = {
                self.doc_id_field: doc_id,
                self.vector_field: vector,
                self.text_field: text,
                self.title_field: paper_title, # Store paper title for context
                
                # Structure fields
                self.chunk_id_field: chunk["chunk_index"],
                self.section_category_field: chunk["section_category"],
                self.parent_section_field: chunk["parent_section"],
                self.page_number_field: chunk["page_number"],
                
                # Optional fields (can be empty or inherited if we had them)
                self.url_field: "",
                self.pdf_url_field: "",
                self.conference_name_field: "", # Could pass these if needed
                self.conference_year_field: 0,
                self.conference_round_field: "",
            }
            data_list.append(entry)
            
        # Insert in batches if necessary (Milvus has limits)
        batch_size = 100
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            self.client.insert(collection_name=self.collection, data=batch)
            print(f"Inserted batch {i} to {i+len(batch)}")

    def get_context_window(self, doc_id: str, center_chunk_index: int, window_size: int = 1) -> str:
        """
        Retrieves the context window around a specific chunk.
        Returns the concatenated text of chunks in [center - window, center + window].
        """
        start_idx = max(0, center_chunk_index - window_size)
        end_idx = center_chunk_index + window_size
        
        # Query Milvus for chunks with same doc_id and index in range
        query = f'{self.doc_id_field} == "{doc_id}" && {self.chunk_id_field} >= {start_idx} && {self.chunk_id_field} <= {end_idx}'
        
        results = self.client.query(
            collection_name=self.collection,
            filter=query,
            output_fields=[self.text_field, self.chunk_id_field],
            limit=window_size * 2 + 5 # Fetch enough
        )
        
        # Sort by chunk_id
        sorted_chunks = sorted(results, key=lambda x: x[self.chunk_id_field])
        
        # Concatenate text
        context_text = "\n\n".join([c[self.text_field] for c in sorted_chunks])
        return context_text



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