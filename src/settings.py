from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # app
    app_name: str = "MyApp"
    debug_mode: bool = False
    database_url: Optional[str] = None

    rag_provider: str = "milvus"

    # --- Milvus / vectorstore ---
    milvus_uri: str = "./milvus_lite.db"
    milvus_user: str = "default"
    milvus_password: str = "123456"
    milvus_doc_collection: str = "documents"
    milvus_chunk_collection: str = "documents_chunks"
    milvus_top_k: int = 10

    milvus_id_field: str = "id"
    milvus_doc_id_field: str = "doc_id"
    milvus_doc_vector_field: str = "doc_vectors"
    milvus_title_field: str = "title"
    milvus_abstract_field: str = "abstract"
    milvus_content_field: str = "content"
    milvus_url_field: str = "url"
    milvus_chunk_id_field: str = "chunk_id"
    milvus_vector_field: str = "vectors"
    milvus_vector_index_metric_type: str = "L2"
    milvus_conference_name_field: str = "conference_name"
    milvus_conference_year_field: str = "conference_year"
    milvus_conference_round_field: str = "conference_round"

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


# create a singleton settings instance to import across the project
settings = Settings()