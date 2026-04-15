"""Tests for the DOCX parser: paragraph indices, character offsets, headings, tables."""

import os

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def simple_docx():
    return os.path.join(FIXTURES, "simple_report.docx")


# --------------------------------------------------------------------------- #
# Test 1: parse_docx returns ParsedDocument with non-empty blocks and raw_text
# --------------------------------------------------------------------------- #
def test_parse_docx_returns_parsed_document(simple_docx):
    from evidenceengine.ingestion.docx_parser import parse_docx

    result = parse_docx(simple_docx)
    assert result is not None
    assert len(result.blocks) > 0
    assert len(result.raw_text) > 0


# --------------------------------------------------------------------------- #
# Test 2: DOCX blocks have page=None
# --------------------------------------------------------------------------- #
def test_docx_blocks_page_is_none(simple_docx):
    from evidenceengine.ingestion.docx_parser import parse_docx

    result = parse_docx(simple_docx)
    for block in result.blocks:
        assert block.position.page is None, (
            f"DOCX page should be None, got {block.position.page}"
        )


# --------------------------------------------------------------------------- #
# Test 3: DOCX blocks have valid paragraph indices and character offsets
# --------------------------------------------------------------------------- #
def test_docx_paragraph_indices(simple_docx):
    from evidenceengine.ingestion.docx_parser import parse_docx

    result = parse_docx(simple_docx)
    for block in result.blocks:
        assert block.position.paragraph >= 0
        assert block.position.char_start >= 0
        assert block.position.char_end > block.position.char_start


# --------------------------------------------------------------------------- #
# Test 4: Character offset integrity for DOCX
# --------------------------------------------------------------------------- #
def test_docx_offset_integrity(simple_docx):
    from evidenceengine.ingestion.docx_parser import parse_docx

    result = parse_docx(simple_docx)
    for i, block in enumerate(result.blocks):
        extracted = result.raw_text[block.position.char_start : block.position.char_end]
        assert extracted == block.text, (
            f"DOCX block {i} offset mismatch:\n"
            f"  block.text={block.text!r}\n"
            f"  raw_text[{block.position.char_start}:{block.position.char_end}]={extracted!r}"
        )


# --------------------------------------------------------------------------- #
# Test 5: DOCX headings detected via paragraph style
# --------------------------------------------------------------------------- #
def test_docx_headings_detected(simple_docx):
    from evidenceengine.ingestion.docx_parser import parse_docx

    result = parse_docx(simple_docx)
    heading_blocks = [b for b in result.blocks if b.block_type == "heading"]
    assert len(heading_blocks) >= 1, "At least one heading should be detected from Heading style"


# --------------------------------------------------------------------------- #
# Test 6: DOCX tables extracted as structured data
# --------------------------------------------------------------------------- #
def test_docx_tables_extracted(simple_docx):
    from evidenceengine.ingestion.docx_parser import parse_docx

    result = parse_docx(simple_docx)
    assert len(result.tables) > 0, "simple_report.docx should contain at least one table"
    for table in result.tables:
        assert "data" in table
        assert isinstance(table["data"], list)
        for row in table["data"]:
            assert isinstance(row, list)
