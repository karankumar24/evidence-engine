"""Assert no Google Fonts CDN references exist in templates or static CSS."""
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "src" / "evidenceengine" / "templates"
STATIC_DIR = Path(__file__).parent.parent.parent / "src" / "evidenceengine" / "static"

CDN_STRINGS = ["fonts.googleapis.com", "fonts.gstatic.com"]

TEMPLATES_TO_CHECK = [
    "base.html",
    "dashboard_index.html",
    "design_system.html",
    "errors/_error_base.html",
]


@pytest.fixture(scope="module")
def jinja_env():
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


@pytest.mark.parametrize("template_name", TEMPLATES_TO_CHECK)
def test_template_has_no_google_fonts(jinja_env, template_name):
    source = jinja_env.loader.get_source(jinja_env, template_name)[0]
    for cdn in CDN_STRINGS:
        assert cdn not in source, f"'{cdn}' found in {template_name}"


def test_fonts_css_exists_and_is_clean():
    fonts_css = STATIC_DIR / "css" / "fonts.css"
    assert fonts_css.exists(), "fonts.css not found"
    content = fonts_css.read_text()
    assert "@font-face" in content, "fonts.css has no @font-face rules"
    assert "/static/fonts/" in content, "fonts.css doesn't reference local fonts"
    for cdn in CDN_STRINGS:
        assert cdn not in content, f"'{cdn}' found in fonts.css"


def test_local_font_files_exist():
    fonts_dir = STATIC_DIR / "fonts"
    assert fonts_dir.exists(), "static/fonts/ directory missing"
    woff2_files = list(fonts_dir.glob("*.woff2"))
    assert len(woff2_files) >= 10, f"Expected ≥10 WOFF2 files, found {len(woff2_files)}"
    families = {"source-serif-4", "inter", "jetbrains-mono"}
    for family in families:
        matches = [f for f in woff2_files if family in f.name]
        assert matches, f"No WOFF2 files found for {family}"
