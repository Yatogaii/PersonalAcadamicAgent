from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # app
    app_name: str = "MyApp"
    debug_mode: bool = False
    database_url: Optional[str] = None

    rag_provider: str = "milvus"

    # --- Milvus / vectorstore ---
    milvus_uri: str = "http://localhost:19530" # Default to standalone, can be file path for lite
    milvus_token: str = "" # For standalone authentication
    milvus_collection: str = "papers" # Unified collection
    milvus_top_k: int = 10

    milvus_id_field: str = "id"
    milvus_doc_id_field: str = "doc_id"
    milvus_vector_field: str = "vectors" # Unified vector field
    milvus_text_field: str = "text" # Unified text field (abstract or content)
    milvus_title_field: str = "title"
    milvus_url_field: str = "url"
    milvus_pdf_url_field: str = "pdf_url"
    milvus_chunk_id_field: str = "chunk_id"
    milvus_vector_index_metric_type: str = "L2"
    milvus_index_type: str = "FLAT"
    milvus_index_params: dict = {"nlist": 1024}
    milvus_search_params: dict = {"nprobe": 10}
    milvus_conference_name_field: str = "conference_name"
    milvus_conference_year_field: str = "conference_year"
    milvus_conference_round_field: str = "conference_round"
    
    # New fields for Structure-Aware RAG
    # milvus_chunk_id_field is already defined above and will be used for sequential index
    milvus_section_category_field: str = "section_category"
    milvus_parent_section_field: str = "parent_section"
    milvus_page_number_field: str = "page_number"

    # --- Postgres / PGVector ---
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "vectordb"
    postgres_table_name: str = "papers"

    # --- Embedding model ---
    embedding_provider: str = "ollama"
    embedding_model: str = "qwen3-embedding:4b"
    embedding_model_base_url: str = "https://huggingface.co/"
    embedding_model_api_key: str = ""
    embedding_dim: int = 2560    
    chunk_strategy: str = "paragraph" # Support: paragraph, fixed_size, sentence, contextual.
    chunk_fixed_size: int = 500

    # --- API keys ---
    KIMI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    HF_TOKEN: str = ""

    # --- Proxies ---
    HTTP_PROXY: Optional[str] = None
    HTTPS_PROXY: Optional[str] = None

    enable_agentic_rag: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


# create a singleton settings instance to import across the project
settings = Settings()