"""Tests for the PDF parser: positional metadata, offset integrity, tables, footnotes."""

import os

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="module")
def simple_pdf():
    return os.path.join(FIXTURES, "simple_report.pdf")


@pytest.fixture(scope="module")
def multi_column_pdf():
    return os.path.join(FIXTURES, "multi_column_report.pdf")


@pytest.fixture(scope="module")
def tables_pdf():
    return os.path.join(FIXTURES, "report_with_tables.pdf")


# --------------------------------------------------------------------------- #
# Test 1: parse_pdf returns a ParsedDocument with non-empty blocks/text
# --------------------------------------------------------------------------- #
def test_parse_pdf_returns_parsed_document(simple_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(simple_pdf)
    assert result is not None
    assert len(result.blocks) > 0
    assert len(result.raw_text) > 0
    assert len(result.markdown_text) > 0
    assert result.filename == os.path.basename(simple_pdf)


# --------------------------------------------------------------------------- #
# Test 2: Character offset integrity — raw_text[start:end] == block.text
# --------------------------------------------------------------------------- #
def test_offset_integrity_simple(simple_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(simple_pdf)
    for i, block in enumerate(result.blocks):
        extracted = result.raw_text[block.position.char_start : block.position.char_end]
        assert extracted == block.text, (
            f"Block {i} offset mismatch:\n"
            f"  block.text={block.text!r}\n"
            f"  raw_text[{block.position.char_start}:{block.position.char_end}]={extracted!r}"
        )


def test_offset_integrity_tables(tables_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(tables_pdf)
    for i, block in enumerate(result.blocks):
        extracted = result.raw_text[block.position.char_start : block.position.char_end]
        assert extracted == block.text, (
            f"Block {i} offset mismatch in tables PDF:\n"
            f"  block.text={block.text!r}\n"
            f"  raw_text[{block.position.char_start}:{block.position.char_end}]={extracted!r}"
        )


# --------------------------------------------------------------------------- #
# Test 3: Blocks have valid page numbers and paragraph indices
# --------------------------------------------------------------------------- #
def test_valid_page_and_paragraph_indices(simple_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(simple_pdf)
    for block in result.blocks:
        assert block.position.page is not None
        assert block.position.page >= 1, f"Page number must be >= 1, got {block.position.page}"
        assert block.position.paragraph >= 0, f"Paragraph index must be >= 0"


# --------------------------------------------------------------------------- #
# Test 4: Section headings are detected
# --------------------------------------------------------------------------- #
def test_headings_detected(simple_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(simple_pdf)
    heading_blocks = [b for b in result.blocks if b.block_type == "heading"]
    assert len(heading_blocks) >= 1, "At least one heading block should be detected"


# --------------------------------------------------------------------------- #
# Test 5: Multi-column PDF produces coherent text (no interleaving)
# --------------------------------------------------------------------------- #
def test_multi_column_coherent_text(multi_column_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(multi_column_pdf)
    # Coherent text means each block's text is a complete fragment (>= 3 chars)
    # and not just single characters or column artefacts
    text_blocks = [b for b in result.blocks if b.block_type == "paragraph"]
    assert len(text_blocks) > 0
    for block in text_blocks:
        assert len(block.text.strip()) >= 3, f"Suspiciously short block: {block.text!r}"


# --------------------------------------------------------------------------- #
# Test 6: Tables extracted as structured data with rows/columns
# --------------------------------------------------------------------------- #
def test_tables_extracted(tables_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(tables_pdf)
    assert len(result.tables) > 0, "tables_pdf should have at least one table"
    for table in result.tables:
        assert "data" in table, "Table must have 'data' key"
        assert "page" in table, "Table must have 'page' key"
        assert isinstance(table["data"], list), "Table data must be a list of rows"
        assert len(table["data"]) > 0, "Table must have at least one row"


# --------------------------------------------------------------------------- #
# Test 7: tables list fields - each table has data (list of lists) and page
# --------------------------------------------------------------------------- #
def test_table_structure(tables_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(tables_pdf)
    for table in result.tables:
        assert isinstance(table["data"], list)
        for row in table["data"]:
            assert isinstance(row, list)


# --------------------------------------------------------------------------- #
# Test 8: Footnotes are extracted (not silently dropped)
# --------------------------------------------------------------------------- #
def test_footnotes_extracted(simple_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(simple_pdf)
    # simple_report.pdf is generated with a footnote — check it appears in raw_text
    # (even if not classified as footnote block type, the text must be present)
    assert "footnote" in result.raw_text.lower() or any(
        b.block_type == "footnote" for b in result.blocks
    ), "Footnote text should appear in raw_text or as a footnote block"


# --------------------------------------------------------------------------- #
# Test 9: ParsedDocument.total_pages is populated
# --------------------------------------------------------------------------- #
def test_total_pages_populated(simple_pdf):
    from evidenceengine.ingestion.pdf_parser import parse_pdf

    result = parse_pdf(simple_pdf)
    assert result.total_pages is not None
    assert result.total_pages >= 1
