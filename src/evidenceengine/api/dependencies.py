"""FastAPI dependency providers."""

from evidenceengine.core.config import settings
from evidenceengine.core.database import get_db  # re-export for API use
from evidenceengine.storage.file_store import FileStore

__all__ = ["get_db", "get_file_store"]


def get_file_store() -> FileStore:
    """Return a FileStore instance using the configured upload directory."""
    return FileStore(settings.upload_dir)
