"""DOCX parser using python-docx with computed character offsets.

Note: DOCX files do not provide page number information. All blocks have page=None.
Character offsets are computed from the assembled raw_text — same integrity guarantee
as the PDF parser: raw_text[char_start:char_end] == block.text for every block.
"""

import os

from docx import Document

from evidenceengine.ingestion.models import ParsedBlock, ParsedDocument, TextPosition


def parse_docx(filepath: str) -> ParsedDocument:
    """Parse a DOCX file and return a ParsedDocument with positional metadata.

    Args:
        filepath: Absolute or relative path to the DOCX file.

    Returns:
        ParsedDocument with blocks, raw_text, markdown_text (= raw_text), tables.
        All blocks have position.page = None (DOCX does not provide page numbers).

    Raises:
        ValueError: If character offset integrity self-check fails.
        FileNotFoundError: If the file does not exist.
    """
    filename = os.path.basename(filepath)
    doc = Document(filepath)

    blocks: list[ParsedBlock] = []
    tables: list[dict] = []
    current_section: str | None = None
    para_idx = 0
    current_offset = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue  # Skip empty paragraphs

        # Detect heading via paragraph style
        style_name = para.style.name if para.style else ""
        is_heading = style_name.startswith("Heading") or style_name.startswith("Title")

        block_type = "heading" if is_heading else "paragraph"
        if is_heading:
            current_section = text

        char_start = current_offset
        char_end = current_offset + len(text)
        current_offset = char_end + 1  # +1 for "\n" separator

        position = TextPosition(
            page=None,
            paragraph=para_idx,
            char_start=char_start,
            char_end=char_end,
            section_header=current_section if not is_heading else None,
            bbox=None,
        )
        blocks.append(ParsedBlock(text=text, position=position, block_type=block_type))
        para_idx += 1

    # Extract tables
    for table_idx, table in enumerate(doc.tables):
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)
        if table_data:
            tables.append(
                {
                    "table_index": table_idx,
                    "page": None,  # DOCX tables don't have page numbers
                    "data": table_data,
                }
            )

    # Assemble raw_text by joining block texts with "\n"
    raw_text = "\n".join(b.text for b in blocks)

    # CRITICAL self-check: recompute offsets from assembled raw_text to verify integrity
    offset = 0
    for i, block in enumerate(blocks):
        expected_start = offset
        expected_end = offset + len(block.text)
        actual_slice = raw_text[expected_start:expected_end]
        if actual_slice != block.text:
            raise ValueError(
                f"DOCX offset integrity failure at block {i}:\n"
                f"  block.text={block.text!r}\n"
                f"  raw_text[{expected_start}:{expected_end}]={actual_slice!r}"
            )
        block.position.char_start = expected_start
        block.position.char_end = expected_end
        offset = expected_end + 1  # +1 for "\n"

    return ParsedDocument(
        filename=filename,
        total_pages=None,  # DOCX does not provide page count without rendering
        blocks=blocks,
        markdown_text=raw_text,  # No markdown conversion for DOCX
        raw_text=raw_text,
        tables=tables,
    )
