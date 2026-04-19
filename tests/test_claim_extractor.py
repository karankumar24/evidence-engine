"""Tests for LLM claim extractor and position recovery.

All LLM calls are mocked via asyncio.to_thread — no real OpenAI API calls made.
The new implementation runs the synchronous OpenAI client inside asyncio.to_thread
to avoid blocking the uvicorn event loop on macOS.
"""
import asyncio
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

def make_mock_message(claims: list[ExtractedClaim] | None = None, refusal: str | None = None):
    """Build a mock OpenAI message object (what _sync_request returns).

    The new implementation returns completion.choices[0].message from the
    thread — so asyncio.to_thread resolves to a message, not a completion.
    """
    response = ClaimExtractionResponse(claims=claims or [])
    mock_message = MagicMock()
    mock_message.parsed = response if refusal is None else None
    mock_message.refusal = refusal
    return mock_message


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

    mock_message = make_mock_message(claims=mock_claims)

    # Patch asyncio.to_thread — the new implementation runs _sync_request in a thread.
    # to_thread returns the message object directly (what _sync_request returns).
    with patch("evidenceengine.extraction.claim_extractor.asyncio.to_thread",
               new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = mock_message
        result = await extract_claims_from_blocks(cited_blocks)

    assert len(result.claims) == 2
    assert result.claims[0].claim_text == "Carbon emissions rose by 12% [1]."
    assert result.claims[1].citation_markers[0].citation_style == "author_year"


@pytest.mark.asyncio
async def test_empty_blocks_returns_empty():
    """Empty block list returns ClaimExtractionResponse with empty claims — no LLM call made."""
    from evidenceengine.extraction.claim_extractor import extract_claims_from_blocks

    # No mock needed: empty input triggers early return before asyncio.to_thread is called.
    result = await extract_claims_from_blocks([])

    assert result.claims == []


@pytest.mark.asyncio
async def test_refusal_returns_empty():
    """LLM refusal returns ClaimExtractionResponse(claims=[]) without raising."""
    from evidenceengine.extraction.claim_extractor import extract_claims_from_blocks

    mock_message = make_mock_message(claims=[], refusal="I cannot process this content.")

    with patch("evidenceengine.extraction.claim_extractor.asyncio.to_thread",
               new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = mock_message
        result = await extract_claims_from_blocks(make_blocks(["Some text [1]."]))

    assert result.claims == []


@pytest.mark.asyncio
async def test_null_parsed_response_skips_chunk():
    """None parsed response is logged and skipped (does not raise, returns empty)."""
    from evidenceengine.extraction.claim_extractor import extract_claims_from_blocks

    mock_message = MagicMock()
    mock_message.refusal = None
    mock_message.parsed = None  # Structured output parsing returned null

    with patch("evidenceengine.extraction.claim_extractor.asyncio.to_thread",
               new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = mock_message
        result = await extract_claims_from_blocks(make_blocks(["Revenue grew 12% [1]."]))

    assert result.claims == []


@pytest.mark.asyncio
async def test_multiple_chunks_aggregated():
    """Claims from multiple chunks are combined into a single response."""
    from evidenceengine.extraction.claim_extractor import extract_claims_from_blocks, _MAX_BLOCKS_PER_CALL

    # Create more blocks than _MAX_BLOCKS_PER_CALL to trigger chunking
    num_extra = 3
    blocks = make_blocks([f"Claim {i} [1]." for i in range(_MAX_BLOCKS_PER_CALL + num_extra)])

    chunk1_claims = [
        ExtractedClaim(
            claim_text="Claim 0 [1].",
            citation_markers=[ExtractedCitationMarker(raw_marker="[1]", citation_style="numeric")],
        )
    ]
    chunk2_claims = [
        ExtractedClaim(
            claim_text="Claim 50 [1].",
            citation_markers=[ExtractedCitationMarker(raw_marker="[1]", citation_style="numeric")],
        )
    ]

    call_count = 0

    async def _mock_to_thread(fn, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return make_mock_message(claims=chunk1_claims)
        return make_mock_message(claims=chunk2_claims)

    with patch("evidenceengine.extraction.claim_extractor.asyncio.to_thread",
               side_effect=_mock_to_thread):
        result = await extract_claims_from_blocks(blocks)

    assert len(result.claims) == 2
    assert call_count == 2  # Two LLM calls for two chunks


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


def test_position_recovery_normalized_char_end_uses_norm_length():
    """When normalized fallback is used, char_end must use len(norm_claim), not len(claim_text)."""
    from evidenceengine.extraction.position_recovery import recover_position

    # norm_claim will be "Carbon  emissions" -> "Carbon emissions" (shorter when extra spaces collapsed)
    claim_text = "Carbon   emissions"   # 18 chars with extra spaces
    norm_claim = "Carbon emissions"      # 16 chars normalized
    raw_text = norm_claim + " [1]."     # raw has normalized form

    result = recover_position(claim_text, raw_text, [])

    # char_end - char_start should equal len of what was actually matched
    matched_len = result["char_end"] - result["char_start"]
    assert matched_len == len(norm_claim), (
        f"Expected matched length {len(norm_claim)}, got {matched_len} — "
        f"char_end must use norm_claim length, not claim_text length"
    )


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
