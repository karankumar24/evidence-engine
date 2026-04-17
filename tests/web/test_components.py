"""Render-smoke tests for Jinja2 component macros.

Each test renders a minimal template that imports a component macro,
then asserts that key data-attribute markers and content are present.
No network, no database, no server required.
"""

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "src" / "evidenceengine" / "templates"


@pytest.fixture(scope="module")
def jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def render(env: Environment, src: str) -> str:
    """Render an inline template string using the real template loader."""
    tpl = env.from_string(src)
    return tpl.render()


# ── Verdict badge ─────────────────────────────────────────────────────────────

def test_verdict_badge_supported(jinja_env):
    html = render(jinja_env, """
    {% import "components/_badge.html" as badge %}
    {{ badge.verdict_badge("supported") }}
    """)
    assert 'data-component="verdict-badge"' in html
    assert 'data-verdict="supported"' in html
    assert "Supported" in html


def test_verdict_badge_contradicted(jinja_env):
    html = render(jinja_env, """
    {% import "components/_badge.html" as badge %}
    {{ badge.verdict_badge("contradicted") }}
    """)
    assert 'data-verdict="contradicted"' in html
    assert "Contradicted" in html


def test_status_badge_completed(jinja_env):
    html = render(jinja_env, """
    {% import "components/_badge.html" as badge %}
    {{ badge.status_badge("completed") }}
    """)
    assert 'data-status="completed"' in html
    assert "completed" in html


def test_method_badge(jinja_env):
    html = render(jinja_env, """
    {% import "components/_badge.html" as badge %}
    {{ badge.method_badge("semantic") }}
    """)
    assert 'data-method="semantic"' in html


# ── Button ────────────────────────────────────────────────────────────────────

def test_button_primary(jinja_env):
    html = render(jinja_env, """
    {% import "components/_button.html" as btn %}
    {% call btn.button(variant="primary") %}Approve{% endcall %}
    """)
    assert 'data-component="button"' in html
    assert 'data-variant="primary"' in html
    assert "Approve" in html


def test_button_destructive(jinja_env):
    html = render(jinja_env, """
    {% import "components/_button.html" as btn %}
    {% call btn.button(variant="destructive") %}Reject{% endcall %}
    """)
    assert 'data-variant="destructive"' in html
    assert "Reject" in html


# ── Confidence ────────────────────────────────────────────────────────────────

def test_confidence_grade_a(jinja_env):
    html = render(jinja_env, """
    {% import "components/_confidence.html" as conf %}
    {{ conf.confidence_display(0.90) }}
    """)
    assert 'data-component="confidence"' in html
    assert 'data-grade="A"' in html
    assert "90%" in html


def test_confidence_grade_b(jinja_env):
    html = render(jinja_env, """
    {% import "components/_confidence.html" as conf %}
    {{ conf.confidence_display(0.72) }}
    """)
    assert 'data-grade="B"' in html


def test_confidence_grade_d(jinja_env):
    html = render(jinja_env, """
    {% import "components/_confidence.html" as conf %}
    {{ conf.confidence_display(0.40) }}
    """)
    assert 'data-grade="D"' in html


# ── Verdict Distribution ──────────────────────────────────────────────────────

def test_verdict_distribution_renders_segments(jinja_env):
    html = render(jinja_env, """
    {% import "components/_pill_distribution.html" as dist %}
    {% set d = {
        "contradicted":         {"count": 2,  "pct": 20},
        "needs_review":         {"count": 3,  "pct": 30},
        "insufficient_support": {"count": 2,  "pct": 20},
        "supported":            {"count": 3,  "pct": 30}
    } %}
    {{ dist.verdict_distribution(d) }}
    """)
    assert 'data-component="verdict-distribution"' in html
    assert 'data-total="10"' in html
    assert 'data-segment="supported"' in html
    assert 'data-segment="contradicted"' in html


def test_verdict_distribution_zero_count_hidden(jinja_env):
    html = render(jinja_env, """
    {% import "components/_pill_distribution.html" as dist %}
    {% set d = {
        "contradicted":         {"count": 0,  "pct": 0},
        "needs_review":         {"count": 0,  "pct": 0},
        "insufficient_support": {"count": 0,  "pct": 0},
        "supported":            {"count": 5,  "pct": 100}
    } %}
    {{ dist.verdict_distribution(d) }}
    """)
    assert 'data-segment="supported"' in html
    assert 'data-segment="contradicted"' not in html


# ── Collapsible ───────────────────────────────────────────────────────────────

def test_collapsible_renders_title(jinja_env):
    html = render(jinja_env, """
    {% import "components/_collapsible.html" as coll %}
    {% call coll.collapsible("Evidence Sources", default_open=True, id="test") %}
      <p>body content</p>
    {% endcall %}
    """)
    assert 'data-component="collapsible"' in html
    assert "Evidence Sources" in html
    assert "body content" in html


def test_collapsible_closed_by_default(jinja_env):
    html = render(jinja_env, """
    {% import "components/_collapsible.html" as coll %}
    {% call coll.collapsible("Notes", default_open=False, id="closed") %}
      <p>hidden</p>
    {% endcall %}
    """)
    assert "Notes" in html
    # Alpine x-data should have open: false
    assert "false" in html


# ── Kbd ───────────────────────────────────────────────────────────────────────

def test_kbd_renders_key(jinja_env):
    html = render(jinja_env, """
    {% import "components/_kbd.html" as kbd %}
    {{ kbd.kbd("A") }}
    """)
    assert 'data-component="kbd"' in html
    assert 'data-key="A"' in html
    assert "A" in html


# ── Card ──────────────────────────────────────────────────────────────────────

def test_card_renders_title_and_body(jinja_env):
    html = render(jinja_env, """
    {% import "components/_card.html" as card %}
    {% call card.card(title="Test Card") %}
      <p class="body">Card body</p>
    {% endcall %}
    """)
    assert 'data-component="card"' in html
    assert "Test Card" in html
    assert "Card body" in html


def test_card_collapsible(jinja_env):
    html = render(jinja_env, """
    {% import "components/_card.html" as card %}
    {% call card.card(title="Collapsible", collapsible=True, default_open=True) %}
      <p>inner</p>
    {% endcall %}
    """)
    assert 'data-collapsible="true"' in html
    assert "Collapsible" in html


# ── Empty State ───────────────────────────────────────────────────────────────

def test_empty_state_no_cta(jinja_env):
    html = render(jinja_env, """
    {% import "components/_empty_state.html" as empty %}
    {{ empty.empty_state("Nothing here", "No results found.") }}
    """)
    assert 'data-component="empty-state"' in html
    assert "Nothing here" in html
    assert "No results found." in html


def test_empty_state_with_cta(jinja_env):
    html = render(jinja_env, """
    {% import "components/_empty_state.html" as empty %}
    {{ empty.empty_state("Empty", "Try again.", cta_label="Go home", cta_href="/") }}
    """)
    assert 'data-empty-cta' in html
    assert "Go home" in html
    assert 'href="/"' in html


# ── Evidence Span ─────────────────────────────────────────────────────────────

def test_evidence_span_packet(jinja_env):
    html = render(jinja_env, """
    {% import "components/_evidence_span.html" as ev %}
    {% set span = {"span_text": "Temperatures have risen.", "page_number": 4, "section_header": "SPM"} %}
    {{ ev.evidence_span(span, "packet") }}
    """)
    assert 'data-component="evidence-span"' in html
    assert 'data-method="packet"' in html
    assert "Temperatures have risen." in html
    assert "p.4" in html
    assert "SPM" in html


def test_evidence_span_web_fallback(jinja_env):
    html = render(jinja_env, """
    {% import "components/_evidence_span.html" as ev %}
    {% set span = {"span_text": "Arctic is warming.", "page_number": None, "section_header": None} %}
    {{ ev.evidence_span(span, "web_fallback") }}
    """)
    assert 'data-method="web_fallback"' in html
    assert "Web fallback" in html


# ── Provenance Step ───────────────────────────────────────────────────────────

def test_provenance_step_renders(jinja_env):
    html = render(jinja_env, """
    {% import "components/_provenance_step.html" as prov %}
    {% call prov.provenance_step(1, "Ingestion", "Parsed 10-page PDF.") %}{% endcall %}
    """)
    assert 'data-component="provenance-step"' in html
    assert 'data-step="1"' in html
    assert "Ingestion" in html
    assert "Parsed 10-page PDF." in html
