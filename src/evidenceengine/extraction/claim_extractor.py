"""Local claim extractor using NLTK sentence splitting.

Replaces the LLM-based extractor to eliminate API dependency from the
hot path. No network calls, no rate limits, no token costs.

Heuristic filters keep only sentence-length factual-looking text,
discarding headers, captions, and table fragments.
"""

import logging
import re

from evidenceengine.extraction.schemas import ClaimExtractionResponse, ExtractedClaim

logger = logging.getLogger(__name__)

# Prefixes that reliably indicate non-claim content
_SKIP_PREFIXES = (
    "fig", "figure", "table", "appendix", "note ", "notes ",
    "see ", "cf.", "e.g.", "i.e.", "et al.", "ibid",
    "acknowledgement", "acknowledgment", "reference", "bibliography",
)

# Patterns that suggest a sentence is a structural artifact, not a claim
_STRUCTURAL_RE = re.compile(
    r"^\s*(\d+[\.\)]\s|[-•–]\s|[A-Z]{2,}\s*:|\([a-z]\))",
    re.IGNORECASE,
)


def _is_likely_claim(sentence: str) -> bool:
    """Return True if the sentence looks like a verifiable factual claim."""
    s = sentence.strip()
    if not s:
        return False
    words = s.split()
    if len(words) < 6 or len(words) > 120:
        return False
    if not s[0].isupper():
        return False
    if not s[-1] in ".!?":
        return False
    lower = s.lower()
    if any(lower.startswith(pfx) for pfx in _SKIP_PREFIXES):
        return False
    if _STRUCTURAL_RE.match(s):
        return False
    # Must contain at least one verb-like token (ends in -s, -ed, -ing, or "is", "are", "was")
    verb_like = re.search(
        r"\b(is|are|was|were|has|have|had|shows?|shows?|demonstrates?|"
        r"achieves?|achieves?|reduces?|increases?|decreases?|improves?|"
        r"suggests?|indicates?|contains?|provides?|results?|found|"
        r"\w+ed|\w+ing)\b",
        lower,
    )
    if not verb_like:
        return False
    return True


async def extract_claims_from_blocks(
    citation_blocks: list[dict],
) -> ClaimExtractionResponse:
    """Extract factual claims from text blocks using NLTK sentence splitting.

    No LLM calls. No API dependency. Runs locally in milliseconds.

    Args:
        citation_blocks: Text blocks from the parsed document.
                         Each dict has at minimum a "text" key.

    Returns:
        ClaimExtractionResponse with filtered factual sentences as claims.
    """
    if not citation_blocks:
        return ClaimExtractionResponse(claims=[])

    try:
        from nltk.tokenize import sent_tokenize
    except ImportError:
        logger.error("nltk not available — returning empty claims")
        return ClaimExtractionResponse(claims=[])

    all_claims: list[ExtractedClaim] = []
    seen: set[str] = set()

    for block in citation_blocks:
        text = block.get("text", "").strip()
        if not text:
            continue
        try:
            sentences = sent_tokenize(text)
        except Exception as exc:
            logger.warning("sent_tokenize failed for block: %s", exc)
            sentences = [s.strip() for s in text.split(".") if s.strip()]

        for sentence in sentences:
            sentence = sentence.strip()
            if not _is_likely_claim(sentence):
                continue
            # Deduplicate
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            all_claims.append(ExtractedClaim(
                claim_text=sentence,
                citation_markers=[],
            ))

    logger.info("Local extractor produced %d candidate claims from %d blocks",
                len(all_claims), len(citation_blocks))
    return ClaimExtractionResponse(claims=all_claims)
