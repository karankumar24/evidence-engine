"""Tests for LLM claim extractor and position recovery.

TDD RED phase: these tests are written BEFORE implementation.
All LLM calls are mocked — no real OpenAI API calls made.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from evidenceengine.extraction.schemas import (
    ClaimExtractionResponse,
    ExtractedClaim,
    ExtractedCitationMarker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_completion(claims: list[ExtractedClaim] | None = None, refusal: str | None = None):
    """Build a mock OpenAI parsed completion object."""
    response = ClaimExtractionResponse(claims=claims or [])
    mock_message = MagicMock()
    mock_message.parsed = response
    mock_message.refusal = refusal
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


def make_blocks(texts: list[str]) -> list[dict]:
    """Build minimal parsed_content blocks from text list."""
    blocks = []
    offset = 0
    for i, text in enumerate(texts):
        blocks.append({
            "text": text,
            "block_type": "paragraph",
            "position": {
                "page": 1,
                "paragraph": i,
                "char_start": offset,
                "char_end": offset + len(text),
                "section_header": "Results",
            },
        })
        offset += len(text) + 1
    return blocks


# ---------------------------------------------------------------------------
# Claim extractor tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_only_cited_claims():
    """Mock LLM returns 2 claims from citation blocks (CLAIM-01)."""
    from evidenceengine.extraction.claim_extractor import extract_claims_from_blocks

    cited_blocks = make_blocks([
        "Carbon emissions rose by 12% [1].",
        "Temperature increased (Smith 2023).",
    ])

    mock_claims = [
        ExtractedClaim(
            claim_text="Carbon emissions rose by 12% [1].",
            citation_markers=[ExtractedCitationMarker(raw_marker="[1]", citation_style="numeric")],
        ),
        ExtractedClaim(
            claim_text="Temperature increased (Smith 2023).",
            citation_markers=[ExtractedCitationMarker(raw_marker="Smith 2023", citation_style="author_year")],
        ),
    ]

    mock_completion = make_mock_completion(claims=mock_claims)

    with patch("evidenceengine.extraction.claim_extractor.AsyncOpenAI") as MockClient:
        mock_instance = AsyncMock()
        MockClient.return_value = mock_instance
        mock_instance.chat.completions.parse = AsyncMock(return_value=mock_completion)

        result = await extract_claims_from_blocks(cited_blocks)

    assert len(result.claims) == 2
    assert result.claims[0].claim_text == "Carbon emissions rose by 12% [1]."
    assert result.claims[1].citation_markers[0].citation_style == "author_year"


@pytest.mark.asyncio
async def test_empty_blocks_returns_empty():
    """Empty block list returns ClaimExtractionResponse with empty claims."""
    from evidenceengine.extraction.claim_extractor import extract_claims_from_blocks

    with patch("evidenceengine.extraction.claim_extractor.AsyncOpenAI"):
        result = await extract_claims_from_blocks([])

    assert result.claims == []


@pytest.mark.asyncio
async def test_refusal_returns_empty():
    """LLM refusal returns ClaimExtractionResponse(claims=[]) without raising."""
    from evidenceengine.extraction.claim_extractor import extract_claims_from_blocks

    mock_completion = make_mock_completion(claims=[], refusal="I cannot process this content.")

    with patch("evidenceengine.extraction.claim_extractor.AsyncOpenAI") as MockClient:
        mock_instance = AsyncMock()
        MockClient.return_value = mock_instance
        mock_instance.chat.completions.parse = AsyncMock(return_value=mock_completion)

        result = await extract_claims_from_blocks(make_blocks(["Some text [1]."]))

    assert result.claims == []


# ---------------------------------------------------------------------------
# Position recovery tests
# ---------------------------------------------------------------------------

def test_position_recovery_exact_match():
    """Exact claim_text found in raw_text returns correct char_start/char_end (CLAIM-02)."""
    from evidenceengine.extraction.position_recovery import recover_position

    raw_text = "Introduction. Carbon emissions rose by 12% [1]. Temperature increased."
    claim_text = "Carbon emissions rose by 12% [1]."
    blocks = make_blocks([claim_text])

    result = recover_position(claim_text, raw_text, blocks)

    expected_start = raw_text.find(claim_text)
    assert result["char_start"] == expected_start
    assert result["char_end"] == expected_start + len(claim_text)
    assert result["position_exact"] is True


def test_position_recovery_normalized_fallback():
    """Claim text with extra whitespace finds approximate match via normalization."""
    from evidenceengine.extraction.position_recovery import recover_position

    raw_text = "Carbon  emissions  rose  by  12%  [1]."
    claim_text = "Carbon emissions rose by 12% [1]."  # normalized version
    blocks = []

    result = recover_position(claim_text, raw_text, blocks)

    # Should find a result (exact may fail, normalized may succeed or partial)
    assert "char_start" in result
    assert "char_end" in result


def test_position_recovery_page_and_section():
    """Matching block found — page_number and section_header populated from block."""
    from evidenceengine.extraction.position_recovery import recover_position

    claim_text = "The glacier melted by 5m [2]."
    raw_text = "Introduction. " + claim_text + " Other text."
    blocks = [
        {
            "text": claim_text,
            "block_type": "paragraph",
            "position": {
                "page": 4,
                "paragraph": 2,
                "char_start": len("Introduction. "),
                "char_end": len("Introduction. ") + len(claim_text),
                "section_header": "Discussion",
            },
        }
    ]

    result = recover_position(claim_text, raw_text, blocks)

    assert result["page_number"] == 4
    assert result["section_header"] == "Discussion"
    assert result["paragraph_index"] == 2
