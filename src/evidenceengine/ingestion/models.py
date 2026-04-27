"""Dataclasses for parsed document content with positional metadata."""

from dataclasses import dataclass, field


@dataclass
class TextPosition:
    """Positional metadata for a parsed text block."""

    page: int | None  # 1-based page number; None for DOCX (no page info)
    paragraph: int  # 0-based block/paragraph index within the document
    char_start: int  # Start offset in ParsedDocument.raw_text (inclusive)
    char_end: int  # End offset in ParsedDocument.raw_text (exclusive)
    section_header: str | None  # Current section heading (last heading seen before this block)
    bbox: tuple[float, float, float, float] | None = None  # PDF only: (x0, y0, x1, y1)


@dataclass
class ParsedBlock:
    """A single text unit (paragraph, heading, table cell text, footnote) with position."""

    text: str
    position: TextPosition
    block_type: str  # "paragraph" | "heading" | "table" | "footnote"


@dataclass
class ParsedDocument:
    """The complete parsed output of a single document file."""

    filename: str
    total_pages: int | None  # None for DOCX
    blocks: list[ParsedBlock]
    markdown_text: str  # PyMuPDF4LLM markdown for PDF; raw_text for DOCX
    raw_text: str  # Concatenated block texts joined with "\n"
    tables: list[dict] = field(default_factory=list)  # Structured table data
    # Extracted paper metadata for citation anchor resolution.
    # Schema: {"title": str, "authors": [str, ...], "year": str}
    # None for DOCX or when PDF metadata is unavailable.
    paper_metadata: dict | None = None
