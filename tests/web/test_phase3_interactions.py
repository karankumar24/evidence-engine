"""Phase 3 interaction surfaces: keyboard help, HTMX toast, reviewed-row marker."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "src" / "evidenceengine" / "templates"


@pytest.fixture(scope="module")
def jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def test_keyboard_help_partial_renders_all_shortcuts(jinja_env: Environment) -> None:
    html = jinja_env.get_template("partials/_keyboard_help.html").render()
    for key in ("A", "R", "F", "J", "K", "?", "Esc"):
        assert f'data-key="{key}"' in html
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html


def test_htmx_toast_host_is_live_region(jinja_env: Environment) -> None:
    html = jinja_env.get_template("partials/_htmx_toast.html").render()
    assert 'aria-live="polite"' in html
    assert 'role="status"' in html
    assert 'id="htmx-toast"' in html


def test_queue_row_marks_reviewed_when_decision_present(jinja_env: Environment) -> None:
    claim_with_decision = SimpleNamespace(
        id="c1",
        claim_text="claim",
        verdicts=[SimpleNamespace(verdict_type="supported", confidence_score=0.9)],
        review_decisions=[SimpleNamespace(action="approve")],
    )
    claim_unreviewed = SimpleNamespace(
        id="c2",
        claim_text="claim",
        verdicts=[SimpleNamespace(verdict_type="supported", confidence_score=0.9)],
        review_decisions=[],
    )
    html = jinja_env.get_template("partials/queue_list.html").render(
        claims=[claim_with_decision, claim_unreviewed],
        packet=SimpleNamespace(id="p"),
        run=SimpleNamespace(id="r"),
        source_docs=[],
        active_filters={"verdict_filter": None, "confidence": None, "source_doc_id": None},
    )
    assert html.count('data-reviewed="true"') == 1


def test_topbar_includes_help_button(jinja_env: Environment) -> None:
    tpl_src = '{% import "components/_topbar.html" as tb %}{{ tb.topbar(packet, run, distribution) }}'
    tpl = jinja_env.from_string(tpl_src)
    html = tpl.render(
        packet=SimpleNamespace(report_filename="x.pdf"),
        run=SimpleNamespace(id="abcdef1234", status="classification_complete", completed_at=None),
        distribution={k: {"count": 0, "pct": 0.0} for k in
                      ("supported", "contradicted", "insufficient_support", "needs_review")},
    )
    assert "data-topbar-help" in html
    assert 'aria-label="Show keyboard shortcuts"' in html
