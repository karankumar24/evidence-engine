"""Application configuration via Pydantic Settings."""

from pydantic import field_validator
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
    classification_model: str = "gpt-4o-mini"
    verdict_needs_review_threshold: float = 0.7
    verdict_prompt_version: str = "v1"
    cors_origins: list[str] = ["http://127.0.0.1:8000", "http://localhost:8000"]
    sentry_dsn: str = ""

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v  # type: ignore[return-value]


settings = Settings()
