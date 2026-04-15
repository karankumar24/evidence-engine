"""Parser protocol — defines the interface for all document parsers."""

from typing import Protocol

from evidenceengine.ingestion.models import ParsedDocument


class DocumentParser(Protocol):
    """Protocol that all document parsers must satisfy."""

    def parse(self, filepath: str) -> ParsedDocument:
        """Parse a document file and return structured content with positional metadata."""
        ...
