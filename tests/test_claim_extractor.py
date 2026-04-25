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


def test_position_recovery_not_found_does_not_match_block_zero():
    """When claim_text can't be located in raw_text, char_start=0 must NOT match block[0].

    Regression: the char_start=0 sentinel-on-failure would match the first block whose
    range starts at 0, lying about page_number / section_header.
    """
    from evidenceengine.extraction.position_recovery import recover_position

    raw_text = "Executive summary paragraph here.  Body follows."
    claim_text = "This claim does not appear in raw text at all."
    blocks = [
        {
            "text": "Executive summary paragraph here.",
            "block_type": "paragraph",
            "position": {
                "page": 1,
                "paragraph": 0,
                "char_start": 0,
                "char_end": 33,
                "section_header": "Introduction",
            },
        }
    ]

    result = recover_position(claim_text, raw_text, blocks)

    assert result["char_start"] == 0
    assert result["char_end"] == 0
    assert result["position_exact"] is False
    # Must NOT falsely attribute the not-found claim to block[0].
    assert result["page_number"] is None
    assert result["paragraph_index"] is None
    assert result["section_header"] is None


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


# ---------------------------------------------------------------------------
# _merge_short_blocks tests (BUG #2: line-level PDF blocks)
# ---------------------------------------------------------------------------

def test_merge_short_blocks_reconstructs_sentences():
    """Line-level blocks (no terminal punct) are merged into paragraph-level blocks."""
    from evidenceengine.extraction.claim_extractor import _merge_short_blocks

    blocks = [
        {"text": "The connection between mitochondrial dysfunction", "block_type": "paragraph", "position": {}},
        {"text": "and Parkinson disease was first established in", "block_type": "paragraph", "position": {}},
        {"text": "a landmark 1989 study by Schapira et al.", "block_type": "paragraph", "position": {}},
    ]
    merged = _merge_short_blocks(blocks)
    # All three fragments should be joined into one block ending in terminal punct
    assert len(merged) == 1
    assert merged[0]["text"].endswith("Schapira et al.")
    assert "mitochondrial dysfunction" in merged[0]["text"]


def test_merge_short_blocks_preserves_paragraph_level_pdfs():
    """Blocks that already end in terminal punctuation flush immediately (no merge)."""
    from evidenceengine.extraction.claim_extractor import _merge_short_blocks

    blocks = [
        {"text": "BERT achieves state-of-the-art results on eleven NLP tasks.", "block_type": "paragraph", "position": {}},
        {"text": "The model is pre-trained on BooksCorpus and English Wikipedia.", "block_type": "paragraph", "position": {}},
    ]
    merged = _merge_short_blocks(blocks)
    # Each block ends in punctuation — should produce 2 separate blocks
    assert len(merged) == 2
    assert merged[0]["text"] == blocks[0]["text"]
    assert merged[1]["text"] == blocks[1]["text"]


# ---------------------------------------------------------------------------
# _fix_word_boundaries tests (BUG #3: PDF concatenation artifacts)
# ---------------------------------------------------------------------------

def test_fix_word_boundaries_known_joins():
    """Common ML-paper line-break concatenations are split correctly."""
    from evidenceengine.extraction.claim_extractor import _fix_word_boundaries

    assert _fix_word_boundaries("Self-attention has beenused successfully.") ==         "Self-attention has been used successfully."
    assert _fix_word_boundaries("achieves thesame objective function.") ==         "achieves the same objective function."
    assert _fix_word_boundaries("asthe encoder-decoder architecture.") ==         "as the encoder-decoder architecture."


def test_fix_word_boundaries_camelcase_split():
    """Lowercase-to-Uppercase boundary is split via regex (genuine camelCase join).

    Occurs when PyMuPDF drops the space between a word ending in lowercase and a
    word starting with uppercase, e.g. 'self\nAttention' -> 'selfAttention'.
    Capital-initial joins like 'Thefeature' are NOT caught by the regex (T is
    uppercase) -- those go via _KNOWN_JOINS.
    """
    from evidenceengine.extraction.claim_extractor import _fix_word_boundaries

    assert _fix_word_boundaries("selfAttention mechanism") == "self Attention mechanism"


def test_fix_word_boundaries_preserves_acronyms():
    """All-caps acronyms (BERT, NLP, GPT) are not split."""
    from evidenceengine.extraction.claim_extractor import _fix_word_boundaries

    assert _fix_word_boundaries("BERT achieves state-of-the-art results.") ==         "BERT achieves state-of-the-art results."
    assert _fix_word_boundaries("NLP tasks require large-scale pre-training.") ==         "NLP tasks require large-scale pre-training."


# ---------------------------------------------------------------------------
# _detect_citation_markers tests (BUG #1: citation markers never detected)
# ---------------------------------------------------------------------------

def test_detect_citation_markers_numeric():
    """Numeric [N] citations are detected with correct style."""
    from evidenceengine.extraction.claim_extractor import _detect_citation_markers

    markers = _detect_citation_markers(
        "The model achieves state-of-the-art results on eleven NLP benchmarks [1]."
    )
    assert len(markers) == 1
    assert markers[0].raw_marker == "[1]"
    assert markers[0].citation_style == "numeric"


def test_detect_citation_markers_author_year():
    """Author-year (Vaswani et al., 2017) citations are detected with correct style."""
    from evidenceengine.extraction.claim_extractor import _detect_citation_markers

    markers = _detect_citation_markers(
        "The attention mechanism (Vaswani et al., 2017) is adopted in BERT."
    )
    assert len(markers) >= 1
    assert any(m.citation_style == "author_year" for m in markers)
    assert any("Vaswani" in m.raw_marker for m in markers)


def test_detect_citation_markers_empty_no_citation():
    """Sentences without citation markers return empty list."""
    from evidenceengine.extraction.claim_extractor import _detect_citation_markers

    markers = _detect_citation_markers(
        "BERT achieves state-of-the-art results on eleven NLP tasks."
    )
    assert markers == []


# ---------------------------------------------------------------------------
# Universal claim filter tests
# Verifies that _is_likely_claim() rejects doc-specific noise universally
# and accepts genuine verifiable claims regardless of domain.
# ---------------------------------------------------------------------------

def _claim(s: str) -> bool:
    from evidenceengine.extraction.claim_extractor import _is_likely_claim
    return _is_likely_claim(s)


# --- Universal legal filter (must reject) ---

def test_rejects_all_rights_reserved():
    assert not _claim("All rights reserved. No part of this publication may be reproduced.")

def test_rejects_copyright_line():
    assert not _claim("Copyright 2024 World Health Organization. All rights reserved.")

def test_rejects_isbn_line():
    assert not _claim("ISBN 978-92-4-156545-8. Printed in France.")

def test_rejects_liability_clause():
    assert not _claim("The publisher disclaims any liability for errors or omissions in this work.")

def test_rejects_warranty_disclaimer():
    assert not _claim("This material is provided without warranty of any kind, express or implied.")

def test_rejects_deontic_modal_reproduction():
    assert not _claim("This work may not be reproduced in whole or in part without written permission.")

def test_rejects_trademark():
    assert not _claim("The designation does not imply trademark or patent status of any product.")

def test_rejects_prohibited_use():
    assert not _claim("Unauthorized reproduction or redistribution of this content is prohibited.")


# --- VBG preamble filter (must reject) ---

def test_rejects_noting_preamble():
    assert not _claim("Noting that climate change poses an increasing threat to human health and ecosystems.")

def test_rejects_acknowledging_preamble():
    assert not _claim("Acknowledging the importance of international cooperation in addressing global challenges.")

def test_rejects_having_regard_preamble():
    assert not _claim("Having regard to the provisions of Article 12 of the International Covenant.")


# --- High proper-noun ratio filter (must reject) ---

def test_rejects_author_institution_line():
    assert not _claim("Dr Maria Rodriguez Instituto Nacional Salud Colombia Director Research Programs.")

def test_rejects_org_country_list():
    assert not _claim("Tedros Adhanom Ghebreyesus Director-General World Health Organization Geneva Switzerland.")


# --- Factual signal gate (must reject) ---

def test_rejects_vague_definitional_claim():
    assert not _claim("Protein folding is a complex and highly regulated biological process.")

def test_rejects_obvious_truism():
    assert not _claim("The immune system plays an important role in protecting the body.")

def test_rejects_general_introductory_sentence():
    assert not _claim("Climate change represents one of the most pressing challenges of our time.")


# --- Factual signal gate (must PASS) ---

def test_passes_numeric_percentage():
    assert _claim("Cardiovascular disease accounts for 17.9 million deaths per year, representing 31% of all global deaths.")

def test_passes_causal_language():
    assert _claim("Antibiotic resistance is caused by overuse and misuse of antimicrobial agents in humans and animals.")

def test_passes_comparative_language():
    assert _claim("Group A achieved 23% higher survival rates compared to the control group at 12 months.")

def test_passes_statistical_qualifier():
    assert _claim("Vaccine efficacy was significantly higher in participants aged 18-64 than in older adults.")

def test_passes_long_declarative_with_finding_verb():
    assert _claim("Studies have consistently found that influenza spreads primarily through respiratory droplets during close contact with infected individuals.")

def test_passes_measurement_with_unit():
    assert _claim("The average global temperature has increased by 1.1 degrees Celsius since the pre-industrial period.")

def test_passes_year_reference_in_context():
    assert _claim("In 2023, the WHO reported over 250 million cases of malaria worldwide, with 94% concentrated in the African region.")

def test_passes_ml_benchmark_result():
    assert _claim("The proposed model achieves 89.4% accuracy on the SQuAD 2.0 benchmark, outperforming all previous single-model baselines.")

def test_passes_economics_claim():
    assert _claim("GDP growth in sub-Saharan Africa is projected to reach 3.8% in 2025, driven by commodity exports and infrastructure investment.")


# --- Regression: existing valid patterns must still work ---

def test_still_rejects_meta_sentence():
    assert not _claim("In this paper, we propose a new method for neural machine translation.")

def test_still_rejects_acknowledgement():
    assert not _claim("We thank the reviewers for their helpful comments and suggestions.")

def test_still_rejects_url_sentence():
    assert not _claim("Full results are available at https://example.com/results/2024.")

def test_still_rejects_toc_artifact():
    assert not _claim("Introduction\t......\t\t\t17")

def test_rejects_toc_plain_dots():
    # BIS/ECB/World Bank style TOC with spaces+dots (no tab)
    assert not _claim("Conclusion  ...............................................................................................................................  29 References  ................................................................................................................................  30")

def test_rejects_toc_short_with_plain_dots():
    assert not _claim("Policy considerations ..........................................................  90")

def test_rejects_bibliography_reference():
    assert not _claim("For a review of methods and modes, see B Cohen, P Hördahl and D Xia, 'Term premia: models and some stylised facts', BIS Quarterly Review, September 2018, pp 79–91.")

def test_rejects_figure_dotted_lines():
    assert not _claim("The dotted horizontal lines indicate January 2007–June 2008 average.")

def test_rejects_figure_dashed_lines():
    assert not _claim("The dashed lines represent the 95% confidence interval around the estimate.")

def test_rejects_figure_bars():
    assert not _claim("Bars represent the interquartile range across 48 country samples from 2010 to 2023.")
