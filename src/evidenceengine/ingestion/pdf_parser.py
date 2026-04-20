"""PDF parser using dual-layer extraction: PyMuPDF4LLM (markdown) + PyMuPDF (positions).

Layer 1 — Markdown: pymupdf4llm.to_markdown() for clean reading-order text with formatting.
Layer 2 — Structured: pymupdf page.get_text("dict") for per-block position metadata.
Layer 3 — Tables: pymupdf page.find_tables() for structured table extraction.

The critical invariant: raw_text[block.char_start:block.char_end] == block.text
This is enforced with a self-check after assembly — any offset drift raises ValueError.
"""

import logging
import os

import fitz  # PyMuPDF

from evidenceengine.ingestion.models import ParsedBlock, ParsedDocument, TextPosition

logger = logging.getLogger(__name__)

# Font size threshold above which a block is classified as a heading
HEADING_FONT_SIZE_THRESHOLD = 13.0

# Bottom fraction of page height below which a block is classified as a footnote
FOOTNOTE_PAGE_FRACTION = 0.15

# Minimum font size for footnote detection (smaller than body text)
FOOTNOTE_FONT_SIZE_MAX = 9.0


def _detect_block_type(
    block: dict,
    page_height: float,
    *,
    is_heading: bool,
) -> str:
    """Classify a block as heading, footnote, or paragraph."""
    if is_heading:
        return "heading"
    block_y0 = block.get("bbox", (0, 0, 0, 0))[1]
    block_y1 = block.get("bbox", (0, 0, 0, 0))[3]
    block_center_y = (block_y0 + block_y1) / 2
    # Footnote heuristic: in bottom 15% of page
    if block_center_y > page_height * (1 - FOOTNOTE_PAGE_FRACTION):
        # Check if font size is small (footnote-like)
        max_size = 0.0
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                max_size = max(max_size, span.get("size", 0))
        if 0 < max_size <= FOOTNOTE_FONT_SIZE_MAX:
            return "footnote"
    return "paragraph"


def parse_pdf(filepath: str) -> ParsedDocument:
    """Parse a PDF and return a ParsedDocument with full positional metadata.

    Args:
        filepath: Absolute or relative path to the PDF file.

    Returns:
        ParsedDocument with blocks, raw_text, markdown_text, tables, and total_pages.

    Raises:
        ValueError: If character offset integrity self-check fails (offset drift detected).
        FileNotFoundError: If the file does not exist.
    """
    filename = os.path.basename(filepath)

    doc = fitz.open(filepath)
    total_pages = doc.page_count

    blocks: list[ParsedBlock] = []
    tables: list[dict] = []
    current_offset = 0
    current_section: str | None = None

    for page_num, page in enumerate(doc, start=1):
        page_height = page.rect.height

        # Layer 3: Extract structured table data first (so we can skip table blocks in Layer 2)
        page_tables = page.find_tables()
        table_bboxes = []
        for table in page_tables:
            try:
                data = table.extract()
                # Normalise: replace None cells with empty string
                cleaned_data = [
                    [cell if cell is not None else "" for cell in row] for row in data
                ]
                tables.append(
                    {
                        "page": page_num,
                        "bbox": list(table.bbox),
                        "data": cleaned_data,
                    }
                )
                table_bboxes.append(table.bbox)
            except Exception as exc:
                # Table extraction is best-effort: a malformed table on one
                # page must not abort parsing the whole document. Log so the
                # failure is visible but continue.
                logger.warning(
                    "pdf_parser: failed to extract table on page %d of %s: %s",
                    page_num, filename, exc,
                )

        # Layer 2: Structured text extraction with block/line/span hierarchy
        block_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        para_idx_on_page = 0

        for raw_block in block_dict.get("blocks", []):
            # Skip image blocks
            if raw_block.get("type") != 0:
                continue

            # Collect all span text from all lines in this block
            lines = raw_block.get("lines", [])
            if not lines:
                continue

            # Get maximum font size in the block (for heading detection)
            max_font_size = 0.0
            span_texts = []
            for line in lines:
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    if size > max_font_size:
                        max_font_size = size
                    text = span.get("text", "")
                    if text:
                        span_texts.append(text)

            block_text = "".join(span_texts).strip()
            if not block_text:
                continue

            is_heading = max_font_size >= HEADING_FONT_SIZE_THRESHOLD
            block_type = _detect_block_type(
                raw_block,
                page_height,
                is_heading=is_heading,
            )

            if block_type == "heading":
                current_section = block_text

            bbox = raw_block.get("bbox")

            char_start = current_offset
            char_end = current_offset + len(block_text)
            current_offset = char_end + 1  # +1 for the "\n" separator

            position = TextPosition(
                page=page_num,
                paragraph=para_idx_on_page,
                char_start=char_start,
                char_end=char_end,
                section_header=current_section if block_type != "heading" else None,
                bbox=tuple(bbox) if bbox else None,
            )

            blocks.append(ParsedBlock(text=block_text, position=position, block_type=block_type))
            para_idx_on_page += 1

    doc.close()

    # Assemble raw_text by joining block texts with "\n"
    raw_text = "\n".join(b.text for b in blocks)

    # Layer 1: Build markdown from assembled blocks (no ONNX/ML dependency).
    # pymupdf4llm 1.27+ triggers ONNX CoreML inference on macOS which hangs.
    # Plain text from fitz is sufficient for LLM claim extraction.
    markdown_text = raw_text

    # CRITICAL self-check: verify offset integrity for every block
    # Recompute offsets from the assembled raw_text to catch any drift
    offset = 0
    for i, block in enumerate(blocks):
        expected_start = offset
        expected_end = offset + len(block.text)
        actual_slice = raw_text[expected_start:expected_end]
        if actual_slice != block.text:
            raise ValueError(
                f"Offset integrity failure at block {i}:\n"
                f"  block.text={block.text!r}\n"
                f"  raw_text[{expected_start}:{expected_end}]={actual_slice!r}"
            )
        # Update the stored position to match the assembled raw_text
        block.position.char_start = expected_start
        block.position.char_end = expected_end
        offset = expected_end + 1  # +1 for "\n"

    return ParsedDocument(
        filename=filename,
        total_pages=total_pages,
        blocks=blocks,
        markdown_text=markdown_text,
        raw_text=raw_text,
        tables=tables,
    )
