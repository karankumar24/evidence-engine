"""A11y structural contracts verified via template source inspection."""
from pathlib import Path
import re

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "src" / "evidenceengine" / "templates"


def read(rel: str) -> str:
    return (TEMPLATES_DIR / rel).read_text()


# ── Collapsibles ──────────────────────────────────────────────────────────────

def test_collapsible_has_aria_expanded():
    src = read("components/_collapsible.html")
    assert ":aria-expanded" in src, "collapsible button missing :aria-expanded binding"


def test_collapsible_has_aria_controls():
    src = read("components/_collapsible.html")
    assert "aria-controls=" in src, "collapsible button missing aria-controls"


def test_collapsible_panel_has_role_region():
    src = read("components/_collapsible.html")
    assert 'role="region"' in src, "collapsible panel missing role=region"


def test_collapsible_panel_has_aria_labelledby():
    src = read("components/_collapsible.html")
    assert "aria-labelledby=" in src, "collapsible panel missing aria-labelledby"


def test_collapsible_chevron_is_aria_hidden():
    src = read("components/_collapsible.html")
    assert 'aria-hidden="true"' in src, "collapsible chevron should be aria-hidden"


def test_card_collapsible_has_aria_controls():
    src = read("components/_card.html")
    assert "aria-controls=" in src, "_card.html collapsible missing aria-controls"


def test_card_panel_has_role_region():
    src = read("components/_card.html")
    assert 'role="region"' in src, "_card.html panel missing role=region"


# ── Queue claim rows ──────────────────────────────────────────────────────────

def test_claim_row_has_role_button():
    src = read("partials/queue_list.html")
    assert 'role="button"' in src, "claim-row div missing role=button"


def test_claim_row_has_tabindex():
    src = read("partials/queue_list.html")
    assert "tabindex=" in src, "claim-row div missing tabindex (not keyboard reachable)"


def test_claim_row_has_aria_label():
    src = read("partials/queue_list.html")
    assert "aria-label=" in src, "claim-row div missing aria-label"


# ── Filter selects ────────────────────────────────────────────────────────────

def test_verdict_filter_has_label():
    src = read("partials/queue_list.html")
    assert "filter-verdict" in src, "verdict filter select missing associated label"


def test_confidence_filter_has_label():
    src = read("partials/queue_list.html")
    assert "filter-confidence" in src, "confidence filter select missing associated label"


# ── Review buttons live region ────────────────────────────────────────────────

def test_review_area_has_aria_live():
    src = read("partials/claim_detail.html")
    assert "aria-live=" in src, "review button area missing aria-live region"


# ── Skip link ────────────────────────────────────────────────────────────────

def test_base_has_skip_link():
    src = read("base.html")
    assert "sr-only focus:not-sr-only" in src, "base.html missing keyboard skip link"
    assert 'href="#detail-panel"' in src, "skip link doesn't point to #detail-panel"


# ── Modal accessibility ───────────────────────────────────────────────────────

def test_keyboard_help_modal_is_accessible():
    src = read("partials/_keyboard_help.html")
    assert 'role="dialog"' in src
    assert 'aria-modal="true"' in src
    assert "aria-labelledby=" in src


# ── Toast live region ─────────────────────────────────────────────────────────

def test_toast_has_role_status_and_aria_live():
    src = read("partials/_htmx_toast.html")
    assert 'role="status"' in src
    assert 'aria-live="polite"' in src
