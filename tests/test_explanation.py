"""Unit tests for the post-verdict explanation generator (Phase 02 Plan 03).

All 8 tests patch the ``sync_call_with_fallback`` symbol bound at import
time in ``evidenceengine.classification.explanation`` — NOT the underlying
``evidenceengine.llm.fallback`` module — because Python import semantics
resolve the name inside the explanation module, not through the source.

These tests are fully hermetic: no network, no transformers, no torch. Any
failure here is genuinely a regression in explanation.py.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from evidenceengine.classification.schemas import VerdictClassificationResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verdict(
    verdict_type: str = "supported",
    confidence: float = 0.91,
    reasoning: str = "Entailment 0.91 cleared the 0.80 threshold.",
) -> VerdictClassificationResponse:
    return VerdictClassificationResponse(
        reasoning=reasoning,
        verdict_type=verdict_type,
        confidence_score=confidence,
    )


def _make_completion_with_parsed(parsed):
    """Mirror the shape sync_call_with_fallback returns (message object)."""
    msg = MagicMock()
    msg.parsed = parsed
    msg.refusal = None
    return msg


CLAIM = "Carbon emissions rose 12% in 2022."
SPANS = [
    {"span_text": "Global CO2 emissions reached 36.6 GtCO2 in 2022, a rise."},
    {"span_text": "Compared to 2021 the increase is ~12%."},
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explanation_always_runs():
    """generate_explanation triggers exactly one sync_call_with_fallback call."""
    from evidenceengine.classification import explanation as exp_mod
    from evidenceengine.classification.explanation import (
        ExplanationResponse,
        generate_explanation,
    )

    parsed = ExplanationResponse(
        explanation="Because entailment 0.91 cleared the 0.80 threshold.",
    )
    with patch.object(
        exp_mod, "sync_call_with_fallback",
        return_value=_make_completion_with_parsed(parsed),
    ) as spy:
        result = await generate_explanation(CLAIM, SPANS, _make_verdict())

    assert result == "Because entailment 0.91 cleared the 0.80 threshold."
    assert spy.call_count == 1


@pytest.mark.asyncio
async def test_explanation_failure_non_blocking(caplog):
    """RuntimeError (chain exhausted) → returns None, logs a warning, never raises."""
    from evidenceengine.classification import explanation as exp_mod
    from evidenceengine.classification.explanation import generate_explanation

    with patch.object(
        exp_mod, "sync_call_with_fallback",
        side_effect=RuntimeError("All models in fallback chain exhausted"),
    ):
        with caplog.at_level(logging.WARNING, logger="evidenceengine.classification.explanation"):
            result = await generate_explanation(CLAIM, SPANS, _make_verdict())

    assert result is None
    assert any("failed" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_explanation_timeout_non_blocking():
    """TimeoutError → returns None without raising."""
    from evidenceengine.classification import explanation as exp_mod
    from evidenceengine.classification.explanation import generate_explanation

    with patch.object(
        exp_mod, "sync_call_with_fallback",
        side_effect=TimeoutError("HTTP timed out"),
    ):
        result = await generate_explanation(CLAIM, SPANS, _make_verdict())

    assert result is None


@pytest.mark.asyncio
async def test_explanation_uses_explanation_prompt_not_verdict_prompt():
    """System message must be EXPLANATION_SYSTEM_PROMPT, NOT llm_classifier.SYSTEM_PROMPT."""
    from evidenceengine.classification import explanation as exp_mod
    from evidenceengine.classification.explanation import (
        EXPLANATION_SYSTEM_PROMPT,
        ExplanationResponse,
        generate_explanation,
    )
    from evidenceengine.classification.llm_classifier import SYSTEM_PROMPT as VERDICT_PROMPT

    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _make_completion_with_parsed(
            ExplanationResponse(explanation="because it entails the claim."),
        )

    with patch.object(exp_mod, "sync_call_with_fallback", side_effect=capture):
        await generate_explanation(CLAIM, SPANS, _make_verdict())

    messages = captured["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"] == EXPLANATION_SYSTEM_PROMPT
    assert system_msg["content"] != VERDICT_PROMPT


@pytest.mark.asyncio
async def test_explanation_passes_model_fallback_chain_and_base_url():
    """Verify fallback call wired to settings.model_fallback_chain + settings.llm_*."""
    from evidenceengine.classification import explanation as exp_mod
    from evidenceengine.classification.explanation import (
        ExplanationResponse,
        generate_explanation,
    )
    from evidenceengine.core.config import settings

    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return _make_completion_with_parsed(
            ExplanationResponse(explanation="entailment is clear evidence."),
        )

    with patch.object(exp_mod, "sync_call_with_fallback", side_effect=capture):
        await generate_explanation(CLAIM, SPANS, _make_verdict())

    assert captured["model_chain"] == settings.model_fallback_chain
    assert captured["api_key"] == settings.llm_api_key
    assert captured["base_url"] == (settings.llm_base_url or None)
    assert captured["timeout"] == settings.llm_fallback_timeout_seconds
    assert captured["response_format"] is ExplanationResponse


@pytest.mark.asyncio
async def test_explanation_returns_parsed_string():
    """Parsed ExplanationResponse.explanation is returned as the str."""
    from evidenceengine.classification import explanation as exp_mod
    from evidenceengine.classification.explanation import (
        ExplanationResponse,
        generate_explanation,
    )

    parsed = ExplanationResponse(explanation="Because entail=0.9 exceeds 0.80.")
    with patch.object(
        exp_mod, "sync_call_with_fallback",
        return_value=_make_completion_with_parsed(parsed),
    ):
        result = await generate_explanation(CLAIM, SPANS, _make_verdict())

    assert result == "Because entail=0.9 exceeds 0.80."


@pytest.mark.asyncio
async def test_explanation_returns_none_when_parsed_is_none():
    """Defensive: parsed is None → return None instead of raising AttributeError."""
    from evidenceengine.classification import explanation as exp_mod
    from evidenceengine.classification.explanation import generate_explanation

    with patch.object(
        exp_mod, "sync_call_with_fallback",
        return_value=_make_completion_with_parsed(None),
    ):
        result = await generate_explanation(CLAIM, SPANS, _make_verdict())

    assert result is None


def test_explanation_response_schema_min_length():
    """ExplanationResponse rejects <10-char strings, accepts real explanations."""
    from evidenceengine.classification.explanation import ExplanationResponse

    with pytest.raises(Exception):  # pydantic ValidationError
        ExplanationResponse(explanation="x")

    ok = ExplanationResponse(explanation="sufficient text here")
    assert ok.explanation == "sufficient text here"
