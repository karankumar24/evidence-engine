"""Bug 2: Empty-file server-side guard tests.

Verifies that validate_file_type raises a clear 'No file attached' error
for zero-byte content instead of leaking 'application/x-empty'.
"""

import pytest

from evidenceengine.ingestion.validation import validate_file_type


def test_empty_bytes_raises_no_file_attached():
    with pytest.raises(ValueError, match="No file attached"):
        validate_file_type(b"")


def test_none_equivalent_raises_no_file_attached():
    # bytearray zero-length also treated as empty
    with pytest.raises(ValueError, match="No file attached"):
        validate_file_type(bytearray())


def test_empty_file_error_does_not_leak_mime_type():
    # Must NOT expose 'application/x-empty' to the user
    with pytest.raises(ValueError) as exc_info:
        validate_file_type(b"")
    assert "application/x-empty" not in str(exc_info.value)


def test_valid_pdf_still_accepted():
    # Regression: ensure the empty check doesn't break valid PDF detection
    pdf_header = b"%PDF-1.4 fake content"
    result = validate_file_type(pdf_header)
    assert result == "pdf"


def test_valid_docx_still_accepted():
    # Regression: valid DOCX (ZIP with word/document marker) still accepted
    import zipfile
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document/>")
    docx_bytes = buf.getvalue()
    result = validate_file_type(docx_bytes)
    assert result == "docx"
