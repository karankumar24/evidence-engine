"""Application configuration via Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    debug: bool = False


settings = Settings()
