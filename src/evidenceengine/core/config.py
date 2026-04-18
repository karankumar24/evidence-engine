"""Application configuration via Pydantic Settings."""

from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    debug: bool = False
    # LLM provider is OpenRouter by default; any OpenAI-compatible endpoint works
    # (Azure, local vLLM, real OpenAI). Accepts either LLM_API_KEY (preferred) or
    # legacy OPENAI_API_KEY — LLM_API_KEY wins when both are set.
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    extraction_model: str = "gpt-4o-mini"
    retrieval_top_k_bm25: int = 10
    retrieval_top_k_final: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    index_dir: str = "./indexes"
    classification_model: str = "gpt-4o-mini"
    verdict_needs_review_threshold: float = 0.7
    verdict_prompt_version: str = "v1"
    cors_origins: Annotated[list[str], NoDecode] = ["http://127.0.0.1:8000", "http://localhost:8000"]
    sentry_dsn: str = ""

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v  # type: ignore[return-value]

    # Back-compat shims: existing code references settings.openai_api_key /
    # settings.openai_base_url. Keep those names working after the LLM_* rename.
    @property
    def openai_api_key(self) -> str:
        return self.llm_api_key

    @property
    def openai_base_url(self) -> str:
        return self.llm_base_url


settings = Settings()
