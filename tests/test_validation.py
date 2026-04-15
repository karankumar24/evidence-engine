"""Tests for file type validation and parse-then-accept gate."""

import io
import os

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# --------------------------------------------------------------------------- #
# Test 7: validate_file_type accepts PDF and DOCX, rejects PNG/TXT/ZIP
# --------------------------------------------------------------------------- #
def test_validate_file_type_accepts_pdf():
    from evidenceengine.ingestion.validation import validate_file_type

    # Minimal valid PDF magic bytes
    pdf_bytes = b"%PDF-1.4\n" + b"%" + bytes([0xE2, 0xE3, 0xCF, 0xD3]) + b"\n"
    result = validate_file_type(pdf_bytes)
    assert result == "pdf"


def test_validate_file_type_accepts_docx():
    from evidenceengine.ingestion.validation import validate_file_type

    # DOCX is a ZIP file — create minimal valid DOCX bytes using python-docx
    import io

    from docx import Document

    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("test")
    doc.save(buf)
    result = validate_file_type(buf.getvalue())
    assert result == "docx"


def test_validate_file_type_rejects_png():
    from evidenceengine.ingestion.validation import validate_file_type

    # PNG magic bytes
    png_bytes = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"\x00" * 100
    with pytest.raises(ValueError, match="not supported"):
        validate_file_type(png_bytes)


def test_validate_file_type_rejects_txt():
    from evidenceengine.ingestion.validation import validate_file_type

    txt_bytes = b"This is just a plain text file with no special markers.\n" * 10
    with pytest.raises(ValueError, match="not supported"):
        validate_file_type(txt_bytes)


def test_validate_file_type_rejects_zip():
    from evidenceengine.ingestion.validation import validate_file_type

    # ZIP magic bytes (not DOCX)
    zip_bytes = bytes([0x50, 0x4B, 0x03, 0x04]) + b"\x00" * 200
    with pytest.raises(ValueError):
        validate_file_type(zip_bytes)


# --------------------------------------------------------------------------- #
# Test 8: validate_and_parse_file returns ParsedDocument for valid files
# --------------------------------------------------------------------------- #
def test_validate_and_parse_file_pdf():
    from evidenceengine.ingestion.validation import validate_and_parse_file

    pdf_path = os.path.join(FIXTURES, "simple_report.pdf")
    result = validate_and_parse_file(pdf_path)
    assert result is not None
    assert len(result.blocks) > 0


def test_validate_and_parse_file_docx():
    from evidenceengine.ingestion.validation import validate_and_parse_file

    docx_path = os.path.join(FIXTURES, "simple_report.docx")
    result = validate_and_parse_file(docx_path)
    assert result is not None
    assert len(result.blocks) > 0


# --------------------------------------------------------------------------- #
# Test 9: validate_and_parse_file raises ValueError with clear message
# --------------------------------------------------------------------------- #
def test_validate_and_parse_file_raises_for_invalid(tmp_path):
    from evidenceengine.ingestion.validation import validate_and_parse_file

    # Create a fake .pdf file that's actually a text file
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"This is not a real PDF file at all.\n" * 10)

    with pytest.raises(ValueError, match="not supported|Failed to parse"):
        validate_and_parse_file(str(fake_pdf))
