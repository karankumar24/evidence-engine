"""Application configuration via Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    debug: bool = False
    openai_api_key: str = ""
    extraction_model: str = "gpt-4o-mini"
    retrieval_top_k_bm25: int = 10
    retrieval_top_k_final: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    index_dir: str = "./indexes"


settings = Settings()
