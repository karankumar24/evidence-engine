"""Tests for Settings — classifier_backend selector + NLI threshold band.

Plan 02-01 locks the default classifier backend to ``nli_primary`` and
introduces four threshold constants consumed by Plan 02 (NLIClassifier) and
Plan 03 (dispatch + tiebreaker). The band-invariant test below protects the
ordering so a future tweak can't silently invert the tiebreaker math.
"""

from __future__ import annotations

from evidenceengine.core.config import Settings


def _fresh(**_: object) -> Settings:
    """Return a Settings instance with .env loading disabled."""
    return Settings(_env_file=None)  # type: ignore[call-arg]


# ── classifier_backend ────────────────────────────────────────────────────────


def test_classifier_backend_default_nli(monkeypatch):
    monkeypatch.delenv("CLASSIFIER_BACKEND", raising=False)
    s = _fresh()
    assert s.classifier_backend == "nli_primary"


def test_classifier_backend_env_override_llm_primary(monkeypatch):
    monkeypatch.setenv("CLASSIFIER_BACKEND", "llm_primary")
    s = _fresh()
    assert s.classifier_backend == "llm_primary"


# ── NLI threshold constants ───────────────────────────────────────────────────


def test_nli_tiebreaker_threshold_default(monkeypatch):
    monkeypatch.delenv("NLI_TIEBREAKER_THRESHOLD", raising=False)
    s = _fresh()
    assert s.nli_tiebreaker_threshold == 0.65


def test_nli_entailment_supported_threshold_default(monkeypatch):
    monkeypatch.delenv("NLI_ENTAILMENT_SUPPORTED_THRESHOLD", raising=False)
    s = _fresh()
    assert s.nli_entailment_supported_threshold == 0.80


def test_nli_contradiction_contradicted_threshold_default(monkeypatch):
    monkeypatch.delenv("NLI_CONTRADICTION_CONTRADICTED_THRESHOLD", raising=False)
    s = _fresh()
    assert s.nli_contradiction_contradicted_threshold == 0.80


def test_nli_min_confidence_for_verdict_default(monkeypatch):
    monkeypatch.delenv("NLI_MIN_CONFIDENCE_FOR_VERDICT", raising=False)
    s = _fresh()
    assert s.nli_min_confidence_for_verdict == 0.50


def test_threshold_band_invariant():
    """Locks the ordering 0.50 < 0.65 < 0.80 (and 0.65 < 0.80 for contradiction).

    Changing any of these defaults requires updating this test AND the band
    documentation in config.py next to the field definitions.
    """
    s = _fresh()
    assert (
        s.nli_min_confidence_for_verdict
        < s.nli_tiebreaker_threshold
        < s.nli_entailment_supported_threshold
    ), (
        "Band invariant broken: "
        f"min_confidence={s.nli_min_confidence_for_verdict} "
        f"tiebreaker={s.nli_tiebreaker_threshold} "
        f"entailment_supported={s.nli_entailment_supported_threshold}"
    )
    assert s.nli_tiebreaker_threshold < s.nli_contradiction_contradicted_threshold, (
        "Band invariant broken: "
        f"tiebreaker={s.nli_tiebreaker_threshold} "
        f"contradiction_contradicted={s.nli_contradiction_contradicted_threshold}"
    )
