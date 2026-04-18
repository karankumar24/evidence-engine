"""Bug 3: Spinner gating tests.

Verifies the submit button spinner is statically gated on both loading state
AND file presence — it must not appear when no file is attached.
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


def test_spinner_requires_loading_and_file(upload_html):
    # x-show="loading" alone is wrong — spinner fires even with no file
    assert 'x-show="loading && reportFile"' in upload_html or \
           "loading && reportFile" in upload_html, (
        "Spinner must be gated on 'loading && reportFile', not just 'loading'"
    )


def test_spinner_not_unconditional(upload_html):
    # Ensure the bare unconditional pattern isn't present
    assert 'x-show="loading"' not in upload_html, (
        "Unconditional 'x-show=\"loading\"' on spinner causes it to show with no file"
    )


def test_submit_disabled_without_file(upload_html):
    assert '!reportFile' in upload_html, (
        "Submit button must include '!reportFile' in its :disabled binding"
    )


def test_alpine_data_has_report_file_state(upload_html):
    assert "reportFile" in upload_html and "null" in upload_html, (
        "Alpine x-data must declare 'reportFile: null' to track file attachment state"
    )
