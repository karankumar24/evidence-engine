"""Tests for verdict classifier — schema validation, classify_claim(), apply_confidence_threshold()."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import ValidationError

from evidenceengine.classification.schemas import VerdictClassificationResponse
from evidenceengine.classification.classifier import (
    classify_claim,
    apply_confidence_threshold,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Section 1: Schema validation tests
# ---------------------------------------------------------------------------


class TestVerdictClassificationResponseSchema:
    """Validate VerdictClassificationResponse Pydantic model."""

    def test_accepts_supported_verdict(self):
        result = VerdictClassificationResponse(
            reasoning="Evidence clearly supports claim.",
            verdict_type="supported",
            confidence_score=0.9,
        )
        assert result.verdict_type == "supported"

    def test_accepts_all_four_valid_verdict_types(self):
        for vt in ("supported", "contradicted", "insufficient_support", "needs_review"):
            result = VerdictClassificationResponse(
                reasoning="Some reasoning.",
                verdict_type=vt,
                confidence_score=0.5,
            )
            assert result.verdict_type == vt

    def test_rejects_invalid_verdict_type(self):
        with pytest.raises(ValidationError):
            VerdictClassificationResponse(
                reasoning="Some reasoning.",
                verdict_type="wrong_label",
                confidence_score=0.5,
            )

    def test_rejects_confidence_score_below_zero(self):
        with pytest.raises(ValidationError):
            VerdictClassificationResponse(
                reasoning="Some reasoning.",
                verdict_type="supported",
                confidence_score=-0.1,
            )

    def test_rejects_confidence_score_above_one(self):
        with pytest.raises(ValidationError):
            VerdictClassificationResponse(
                reasoning="Some reasoning.",
                verdict_type="supported",
                confidence_score=1.1,
            )

    def test_field_order_reasoning_before_verdict_type_before_confidence(self):
        keys = list(VerdictClassificationResponse.model_fields.keys())
        assert keys == ["reasoning", "verdict_type", "confidence_score"], (
            f"Expected field order [reasoning, verdict_type, confidence_score], got {keys}"
        )


# ---------------------------------------------------------------------------
# Section 2: classify_claim() tests
# ---------------------------------------------------------------------------


def _make_mock_response(verdict_type: str, confidence_score: float, reasoning: str = "Test reasoning."):
    """Build a mock OpenAI response with a parsed VerdictClassificationResponse."""
    parsed = VerdictClassificationResponse(
        reasoning=reasoning,
        verdict_type=verdict_type,
        confidence_score=confidence_score,
    )
    mock_response = MagicMock()
    mock_response.choices[0].message.parsed = parsed
    mock_response.choices[0].message.refusal = None
    return mock_response


SAMPLE_EVIDENCE_SPANS = [
    {"span_text": "Revenue grew by 12% year over year.", "relevance_score": 0.95, "rank": 1},
    {"span_text": "Operating costs declined by 3%.", "relevance_score": 0.82, "rank": 2},
]


class TestClassifyClaimFunction:
    """Tests for the async classify_claim() function."""

    @pytest.mark.asyncio
    async def test_returns_supported_verdict(self):
        mock_response = _make_mock_response("supported", 0.9)
        with patch("evidenceengine.classification.classifier.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

            result = await classify_claim(
                claim_text="Revenue grew by 12%.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )

        assert result.verdict_type == "supported"
        assert result.confidence_score == 0.9

    @pytest.mark.asyncio
    async def test_returns_contradicted_verdict(self):
        mock_response = _make_mock_response("contradicted", 0.88)
        with patch("evidenceengine.classification.classifier.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

            result = await classify_claim(
                claim_text="Revenue declined.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )

        assert result.verdict_type == "contradicted"

    @pytest.mark.asyncio
    async def test_returns_insufficient_support_verdict(self):
        mock_response = _make_mock_response("insufficient_support", 0.6)
        with patch("evidenceengine.classification.classifier.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

            result = await classify_claim(
                claim_text="All financial metrics improved.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )

        assert result.verdict_type == "insufficient_support"

    @pytest.mark.asyncio
    async def test_returns_needs_review_verdict(self):
        mock_response = _make_mock_response("needs_review", 0.5)
        with patch("evidenceengine.classification.classifier.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

            result = await classify_claim(
                claim_text="Performance was mixed.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )

        assert result.verdict_type == "needs_review"

    @pytest.mark.asyncio
    async def test_returns_needs_review_with_zero_confidence_on_refusal(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.refusal = "I cannot classify this claim."
        mock_response.choices[0].message.parsed = None

        with patch("evidenceengine.classification.classifier.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

            result = await classify_claim(
                claim_text="Sensitive claim.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )

        assert result.verdict_type == "needs_review"
        assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_formats_evidence_spans_as_numbered_blocks(self):
        mock_response = _make_mock_response("supported", 0.9)
        captured_messages = []

        async def capture_parse(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return mock_response

        with patch("evidenceengine.classification.classifier.AsyncOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.chat.completions.parse = capture_parse

            await classify_claim(
                claim_text="Revenue grew.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )

        user_message = next(m for m in captured_messages if m["role"] == "user")
        content = user_message["content"]
        assert "[Evidence 1 (relevance: 0.95)]" in content
        assert "[Evidence 2 (relevance: 0.82)]" in content
        assert "Revenue grew by 12% year over year." in content


# ---------------------------------------------------------------------------
# Section 3: apply_confidence_threshold() tests — pure function, no mocking
# ---------------------------------------------------------------------------


class TestApplyConfidenceThreshold:
    """Tests for apply_confidence_threshold() pure function."""

    def _make_result(self, verdict_type: str, confidence_score: float) -> VerdictClassificationResponse:
        return VerdictClassificationResponse(
            reasoning="Test reasoning.",
            verdict_type=verdict_type,
            confidence_score=confidence_score,
        )

    def test_returns_unchanged_when_confidence_above_threshold(self):
        result = self._make_result("supported", 0.9)
        out = apply_confidence_threshold(result, threshold=0.7)
        assert out.verdict_type == "supported"
        assert out.confidence_score == 0.9

    def test_overrides_supported_to_needs_review_when_confidence_below_threshold(self):
        result = self._make_result("supported", 0.5)
        out = apply_confidence_threshold(result, threshold=0.7)
        assert out.verdict_type == "needs_review"

    def test_overrides_contradicted_to_needs_review_when_confidence_below_threshold(self):
        result = self._make_result("contradicted", 0.3)
        out = apply_confidence_threshold(result, threshold=0.7)
        assert out.verdict_type == "needs_review"

    def test_does_not_change_already_needs_review(self):
        result = self._make_result("needs_review", 0.1)
        out = apply_confidence_threshold(result, threshold=0.7)
        assert out.verdict_type == "needs_review"
        assert out.confidence_score == 0.1

    def test_does_not_change_confidence_score_when_overriding_verdict_type(self):
        result = self._make_result("supported", 0.5)
        out = apply_confidence_threshold(result, threshold=0.7)
        assert out.verdict_type == "needs_review"
        assert out.confidence_score == 0.5  # Confidence preserved, only verdict_type changes
