"""Tests for the ClassifierBackend Protocol, get_backend() dispatcher, and the
backward-compat classifier.py shim.

Phase 02 Plan 01 — foundation only. No behavior change vs. today's LLM path.
Because Plan 02's NLIClassifier has not landed yet, any branch that requires
it uses ``pytest.importorskip`` so this file stays green standalone AND
auto-activates once Plan 02 lands.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evidenceengine.classification.backend import ClassifierBackend, get_backend
from evidenceengine.classification.llm_classifier import LLMClassifier
from evidenceengine.classification.schemas import VerdictClassificationResponse
from evidenceengine.core.config import settings


def test_llm_classifier_is_backend():
    """LLMClassifier satisfies the runtime_checkable ClassifierBackend Protocol."""
    assert isinstance(LLMClassifier(), ClassifierBackend)


def test_nli_classifier_is_backend():
    """NLIClassifier (Plan 02) satisfies the Protocol once it lands.

    Skipped until Plan 02 delivers the module.
    """
    mod = pytest.importorskip("evidenceengine.classification.nli_classifier")
    NLIClassifier = mod.NLIClassifier
    assert isinstance(NLIClassifier(), ClassifierBackend)


def test_get_backend_returns_llm_when_llm_primary(monkeypatch):
    monkeypatch.setattr(settings, "classifier_backend", "llm_primary")
    backend = get_backend()
    assert isinstance(backend, LLMClassifier)


def test_get_backend_attempts_nli_when_nli_primary(monkeypatch):
    """When classifier_backend='nli_primary' and NLIClassifier is available,
    get_backend() returns an instance of it. Skipped until Plan 02 lands."""
    mod = pytest.importorskip("evidenceengine.classification.nli_classifier")
    NLIClassifier = mod.NLIClassifier
    monkeypatch.setattr(settings, "classifier_backend", "nli_primary")
    backend = get_backend()
    assert isinstance(backend, NLIClassifier)


def test_classifier_shim_still_exports_classify_claim_and_asyncio():
    """Historical patch targets must still resolve as module-level attrs."""
    import evidenceengine.classification.classifier as c
    assert hasattr(c, "classify_claim")
    assert hasattr(c, "asyncio")
    # apply_confidence_threshold is imported by pipeline.py — must remain.
    assert hasattr(c, "apply_confidence_threshold")


@pytest.mark.asyncio
async def test_llm_classifier_delegates_to_classify_claim():
    """LLMClassifier.classify() must reproduce classify_claim() verbatim for
    a mocked LLM response.
    """
    canned = SimpleNamespace(
        refusal=None,
        parsed=VerdictClassificationResponse(
            reasoning="canned reasoning",
            verdict_type="supported",
            confidence_score=0.91,
        ),
    )

    def _fake_sync_call_with_fallback(**_kwargs):
        return canned

    with patch(
        "evidenceengine.classification.llm_classifier.sync_call_with_fallback",
        _fake_sync_call_with_fallback,
    ):
        result = await LLMClassifier().classify(
            "Revenue grew 12% YoY.",
            [
                {
                    "span_text": "Annual revenue increased 12% compared to prior year.",
                    "relevance_score": 0.9,
                    "rank": 1,
                }
            ],
        )

    assert isinstance(result, VerdictClassificationResponse)
    assert result.verdict_type == "supported"
    assert result.confidence_score == pytest.approx(0.91)
    assert result.reasoning == "canned reasoning"
