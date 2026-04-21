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


def _make_mock_message(verdict_type: str, confidence_score: float, reasoning: str = "Test reasoning."):
    """Build the message object that _sync_request returns (completion.choices[0].message)."""
    parsed = VerdictClassificationResponse(
        reasoning=reasoning,
        verdict_type=verdict_type,
        confidence_score=confidence_score,
    )
    mock_message = MagicMock()
    mock_message.parsed = parsed
    mock_message.refusal = None
    return mock_message


SAMPLE_EVIDENCE_SPANS = [
    {"span_text": "Revenue grew by 12% year over year.", "relevance_score": 0.95, "rank": 1},
    {"span_text": "Operating costs declined by 3%.", "relevance_score": 0.82, "rank": 2},
]


class TestClassifyClaimFunction:
    """Tests for the async classify_claim() function."""

    @pytest.mark.asyncio
    async def test_returns_supported_verdict(self):
        mock_message = _make_mock_message("supported", 0.9)
        with patch("evidenceengine.classification.classifier.asyncio.to_thread",
                   new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_message
            result = await classify_claim(
                claim_text="Revenue grew by 12%.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )
        assert result.verdict_type == "supported"
        assert result.confidence_score == 0.9

    @pytest.mark.asyncio
    async def test_returns_contradicted_verdict(self):
        mock_message = _make_mock_message("contradicted", 0.88)
        with patch("evidenceengine.classification.classifier.asyncio.to_thread",
                   new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_message
            result = await classify_claim(
                claim_text="Revenue declined.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )
        assert result.verdict_type == "contradicted"

    @pytest.mark.asyncio
    async def test_returns_insufficient_support_verdict(self):
        mock_message = _make_mock_message("insufficient_support", 0.6)
        with patch("evidenceengine.classification.classifier.asyncio.to_thread",
                   new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_message
            result = await classify_claim(
                claim_text="All financial metrics improved.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )
        assert result.verdict_type == "insufficient_support"

    @pytest.mark.asyncio
    async def test_returns_needs_review_verdict(self):
        mock_message = _make_mock_message("needs_review", 0.5)
        with patch("evidenceengine.classification.classifier.asyncio.to_thread",
                   new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_message
            result = await classify_claim(
                claim_text="Performance was mixed.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )
        assert result.verdict_type == "needs_review"

    @pytest.mark.asyncio
    async def test_returns_needs_review_with_zero_confidence_on_refusal(self):
        mock_message = MagicMock()
        mock_message.refusal = "I cannot classify this claim."
        mock_message.parsed = None
        with patch("evidenceengine.classification.classifier.asyncio.to_thread",
                   new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_message
            result = await classify_claim(
                claim_text="Sensitive claim.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )
        assert result.verdict_type == "needs_review"
        assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_formats_evidence_spans_as_numbered_blocks(self):
        """Verify the user message sent to the LLM contains correctly formatted evidence blocks.

        Patches openai.OpenAI so _sync_request runs fully (building the real message)
        but uses a fake sync client to capture what was sent.
        """
        captured_messages = []

        def fake_parse(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            from evidenceengine.classification.schemas import VerdictClassificationResponse
            parsed = VerdictClassificationResponse(
                reasoning="Test.", verdict_type="supported", confidence_score=0.9
            )
            result_msg = MagicMock()
            result_msg.refusal = None
            result_msg.parsed = parsed
            return MagicMock(choices=[MagicMock(message=result_msg)])

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = mock_openai_cls.return_value.__enter__.return_value
            mock_client.beta.chat.completions.parse = fake_parse

            await classify_claim(
                claim_text="Revenue grew.",
                evidence_spans=SAMPLE_EVIDENCE_SPANS,
            )

        user_msg = next(m for m in captured_messages if m["role"] == "user")
        content = user_msg["content"]
        assert "[Evidence 1 | relevance: 0.95]" in content
        assert "[Evidence 2 | relevance: 0.82]" in content
        assert "Revenue grew by 12% year over year." in content

    def test_format_evidence_block_tags_source_and_same_document(self):
        """_format_evidence_block renders source filename + SAME/EXTERNAL tag when available."""
        from evidenceengine.classification.classifier import _format_evidence_block

        same_doc_span = {
            "span_text": "body text",
            "relevance_score": 0.91,
            "source_filename": "report.pdf",
            "is_same_doc_as_claim": True,
        }
        external_span = {
            "span_text": "foo",
            "relevance_score": 0.70,
            "source_filename": "source_02_ipcc.pdf",
            "is_same_doc_as_claim": False,
        }
        same = _format_evidence_block(0, same_doc_span)
        ext = _format_evidence_block(1, external_span)

        assert "source: report.pdf" in same
        assert "SAME DOCUMENT AS CLAIM" in same
        assert "relevance: 0.91" in same

        assert "source: source_02_ipcc.pdf" in ext
        assert "EXTERNAL SOURCE" in ext
        assert "relevance: 0.70" in ext

    def test_format_evidence_block_backward_compat_without_source_metadata(self):
        """Spans without source_filename render the same compact format as pre-A1."""
        from evidenceengine.classification.classifier import _format_evidence_block

        result = _format_evidence_block(0, {"span_text": "text", "relevance_score": 0.5})
        # No source, no SAME/EXTERNAL tag — just relevance.
        assert result.startswith("[Evidence 1 | relevance: 0.50]")
        assert "SAME" not in result
        assert "EXTERNAL" not in result


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
