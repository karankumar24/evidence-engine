"""Render-smoke tests for the unreviewable-run guard in claim_detail + dashboard."""

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


def _claim(with_verdict: bool):
    verdicts = []
    if with_verdict:
        verdicts = [
            SimpleNamespace(
                verdict_type="supported",
                confidence_score=0.91,
                reasoning="Matches supporting evidence.",
                model_name="gpt-4o",
                verdict_evidence=[],
            )
        ]
    return SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        claim_text="Claim text.",
        char_start=0,
        char_end=10,
        verdicts=verdicts,
        review_decisions=[],
    )


def test_claim_detail_hides_review_buttons_when_no_verdict(jinja_env: Environment) -> None:
    tpl = jinja_env.get_template("partials/claim_detail.html")
    html = tpl.render(
        claim=_claim(with_verdict=False),
        context_snippet="Claim text.",
        packet_id="p",
        run_id="r",
    )
    assert "no verdict yet" in html
    assert 'hx-post="/api/runs/' not in html


def test_claim_detail_shows_review_buttons_when_verdict_present(jinja_env: Environment) -> None:
    tpl = jinja_env.get_template("partials/claim_detail.html")
    html = tpl.render(
        claim=_claim(with_verdict=True),
        context_snippet="Claim text.",
        packet_id="p",
        run_id="r",
    )
    assert "no verdict yet" not in html
    assert 'hx-post="/api/runs/r/claims/' in html


def test_dashboard_shows_banner_when_no_verdicts(jinja_env: Environment) -> None:
    tpl = jinja_env.get_template("dashboard.html")
    empty_dist = {k: {"count": 0, "pct": 0.0} for k in
                  ("supported", "contradicted", "insufficient_support", "needs_review")}
    html = tpl.render(
        packet=SimpleNamespace(id="p", report_filename="x.pdf", source_documents=[]),
        run=SimpleNamespace(id="r", status="extraction_complete", created_at=None),
        claims=[],
        distribution=empty_dist,
        source_docs=[],
        active_filters={"verdict_filter": None, "confidence": None, "source_doc_id": None},
    )
    assert "Nothing to review yet" in html
    assert "extraction_complete" in html


def test_dashboard_shows_default_prompt_when_verdicts_exist(jinja_env: Environment) -> None:
    tpl = jinja_env.get_template("dashboard.html")
    dist = {
        "supported": {"count": 3, "pct": 60.0},
        "contradicted": {"count": 1, "pct": 20.0},
        "insufficient_support": {"count": 1, "pct": 20.0},
        "needs_review": {"count": 0, "pct": 0.0},
    }
    html = tpl.render(
        packet=SimpleNamespace(id="p", report_filename="x.pdf", source_documents=[]),
        run=SimpleNamespace(id="r", status="classification_complete", created_at=None),
        claims=[],
        distribution=dist,
        source_docs=[],
        active_filters={"verdict_filter": None, "confidence": None, "source_doc_id": None},
    )
    assert "Nothing to review yet" not in html
    assert "Select a claim to review" in html
