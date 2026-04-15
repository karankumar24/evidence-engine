"""File type validation via magic bytes and parse-then-accept gate.

validate_file_type: Checks raw bytes for PDF or DOCX MIME type.
validate_and_parse_file: Single entry point that validates then parses (INGEST-06).

The parse-then-accept pattern: files are not accepted until they parse successfully.
Any file that fails validation or parsing returns a clear error — never silently stored.
"""

import os

import magic

from evidenceengine.ingestion.models import ParsedDocument

# Accepted MIME types
_ACCEPTED_MIME_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    # Some systems report DOCX as generic zip or Office Open XML
    "application/zip": None,  # Need additional check for DOCX
    "application/octet-stream": None,  # Fallback for some PDF detections
}

_DOCX_SIGNATURES = [
    b"PK\x03\x04",  # ZIP header (DOCX is a ZIP)
]

_PDF_SIGNATURES = [
    b"%PDF",
]


def validate_file_type(file_content: bytes) -> str:
    """Validate file type from raw bytes using python-magic.

    Args:
        file_content: Raw bytes of the file (first 2048+ bytes sufficient).

    Returns:
        "pdf" or "docx"

    Raises:
        ValueError: If the file type is not PDF or DOCX, with a clear message.
    """
    # Strategy: check raw signatures first (reliable for PDF and DOCX),
    # then fall back to python-magic for confirmation.

    # PDF signature check (most reliable)
    if file_content[:4] == b"%PDF":
        return "pdf"

    # DOCX is a ZIP with [Content_Types].xml and word/ entries
    if file_content[:4] == b"PK\x03\x04":
        # ZIP-based file — check for DOCX internal structure markers
        # These markers appear early in the ZIP central directory
        if b"[Content_Types].xml" in file_content or b"word/document" in file_content:
            return "docx"
        # Some ZIP files without DOCX markers — check with magic
        # Note: if it's a ZIP but NOT a DOCX, it will be rejected below

    # Use python-magic on full content (or large chunk) for remaining cases
    mime_type = magic.from_buffer(file_content, mime=True)

    if mime_type == "application/pdf":
        return "pdf"

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/x-zip-compressed",
    ):
        # If we're here with a ZIP it didn't match DOCX markers above — reject
        if mime_type == "application/zip":
            raise ValueError(
                "ZIP files are not supported. Only PDF and DOCX files are accepted."
            )
        return "docx"

    raise ValueError(
        f"File type '{mime_type}' is not supported. Only PDF and DOCX files are accepted."
    )


def validate_and_parse_file(
    filepath: str,
    file_content: bytes | None = None,
) -> ParsedDocument:
    """Validate and parse a file — the single parse-then-accept entry point.

    If file_content is provided, validates type from bytes. Otherwise reads from filepath.
    Determines file type, calls the appropriate parser, and returns ParsedDocument.

    Args:
        filepath: Path to the file.
        file_content: Raw bytes of the file (optional; read from filepath if None).

    Returns:
        ParsedDocument with parsed content and positional metadata.

    Raises:
        ValueError: If the file type is not supported, or if parsing fails.
        FileNotFoundError: If filepath does not exist.
    """
    from evidenceengine.ingestion.docx_parser import parse_docx
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    filename = os.path.basename(filepath)

    # Read content if not provided
    if file_content is None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "rb") as f:
            file_content = f.read()

    # Validate file type from magic bytes
    try:
        file_type = validate_file_type(file_content)
    except ValueError:
        raise  # Re-raise with original message

    # Dispatch to appropriate parser
    try:
        if file_type == "pdf":
            return parse_pdf(filepath)
        elif file_type == "docx":
            return parse_docx(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to parse {filename}: {exc}") from exc
