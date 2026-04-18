"""Bug 1: Report drop zone contract tests.

Verifies the upload form template has correct input names, accept attributes,
and a drop handler that uses DataTransfer (not direct FileList assignment).
"""

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "src" / "evidenceengine" / "templates"


@pytest.fixture(scope="module")
def upload_html() -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    tpl = env.get_template("upload.html")
    return tpl.render(error=None, max_mb=50)


def test_report_input_name(upload_html):
    assert 'name="report"' in upload_html


def test_report_input_accept(upload_html):
    assert 'accept=".pdf,.docx' in upload_html


def test_report_input_required(upload_html):
    # The input must be marked required for native browser validation
    assert "required" in upload_html


def test_drop_handler_uses_datatransfer(upload_html):
    # Direct FileList assignment ($refs.reportInput.files = $event.dataTransfer.files)
    # silently fails in most browsers. The fix uses DataTransfer constructor.
    assert "new DataTransfer()" in upload_html, (
        "Drop handler must use 'new DataTransfer()' — direct FileList assignment is read-only"
    )


def test_drop_handler_not_direct_assignment(upload_html):
    assert "$refs.reportInput.files = $event.dataTransfer.files" not in upload_html, (
        "Direct FileList assignment silently fails — must use DataTransfer constructor instead"
    )


def test_submit_button_gated_on_report_file(upload_html):
    assert "!reportFile" in upload_html, (
        "Submit button must be disabled until reportFile is set"
    )


def test_spinner_gated_on_report_file(upload_html):
    # Spinner must only appear when both loading AND a file is attached
    assert "loading && reportFile" in upload_html, (
        "Spinner must be gated on 'loading && reportFile' to avoid showing with no file"
    )


def test_sources_input_name(upload_html):
    assert 'name="sources"' in upload_html


def test_sources_input_multiple(upload_html):
    assert "multiple" in upload_html
