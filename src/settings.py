from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # app
    app_name: str = "MyApp"
    debug_mode: bool = False
    database_url: Optional[str] = None

    # --- Milvus / vectorstore ---
    milvus_uri: str = "http://localhost:19530"
    milvus_user: str = "default"
    milvus_password: str = "123456"
    milvus_collection: str = "documents"
    milvus_top_k: int = 10

    milvus_id_field: str = "id"
    milvus_doc_id_field: str = "doc_id"
    milvus_title_field: str = "title"
    milvus_abstract_field: str = "abstract"
    milvus_content_field: str = "content"
    milvus_url_field: str = "url"
    milvus_chunk_id_field: str = "chunk_id"
    milvus_vector_field: str = "vectors"
    milvus_vector_index_metric_type: str = "L2"

    # --- Embedding model ---
    embedding_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_model_base_url: str = "https://huggingface.co/"
    embedding_model_api_key: str = ""
    embedding_dim: int = 2048

    # --- API keys ---
    kimi_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


# create a singleton settings instance to import across the project
settings = Settings()