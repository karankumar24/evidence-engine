"""Local claim extractor using NLTK sentence splitting.

Replaces the LLM-based extractor to eliminate API dependency from the
hot path. No network calls, no rate limits, no token costs.

Heuristic filters keep only sentence-length factual-looking text,
discarding headers, captions, and table fragments.
"""

import logging
import re

from evidenceengine.extraction.schemas import (
    ClaimExtractionResponse,
    ExtractedClaim,
    ExtractedCitationMarker,
)

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

# Acknowledgement content anywhere in the sentence (not just prefixes)
_ACK_RE = re.compile(
    r"\b(thank|grateful|generously|supported by grant|funded by|"
    r"we gratefully|with the support of)\b",
    re.IGNORECASE,
)

# Self-referential meta-sentences: structural commentary, not verifiable claims
_META_RE = re.compile(
    r"^(in this (paper|work|study|article|section|chapter)|"
    r"this (paper|work|study|section) (presents|proposes|describes|introduces|discusses)|"
    r"(section|chapter|figure|table|appendix)\s+\d)",
    re.IGNORECASE,
)

# Known word concatenations from PyMuPDF line-break extraction in ML/scientific papers.
# PyMuPDF drops the space when two consecutive lines share a word boundary without a
# hyphen (e.g. "been\nused" → "beenused"). Lookup applied before sent_tokenize().
_KNOWN_JOINS: dict[str, str] = {
    "asthe": "as the",
    "thesame": "the same",
    "beenused": "been used",
    "isthe": "is the",
    "inthe": "in the",
    "ofthe": "of the",
    "tothe": "to the",
    "forthe": "for the",
    "andthe": "and the",
    "withthe": "with the",
    "fromthe": "from the",
    "bythe": "by the",
    "onthe": "on the",
    "atthe": "at the",
    "isbased": "is based",
    # Capital-prefix joins: "The\nfeature" → "Thefeature" (PyMuPDF line break
    # where the second line starts with lowercase drops the leading space)
    "Thefeature": "The feature",
    "Themodel": "The model",
    # ML paper compound joins seen in BERT/Attention papers
    "layerto": "layer to",
    "widerange": "wide range",
    "andlanguage": "and language",
    "initializemodels": "initialize models",
    "isfeature": "is feature",
    "outputlayer": "output layer",
    "taskspecific": "task specific",
    "downstreamtasks": "downstream tasks",
    "languagemodel": "language model",
    "trainingdata": "training data",
    "machinelearning": "machine learning",
    "deeplearning": "deep learning",
    "neuralnetwork": "neural network",
    # Hyphenation artifacts from PDF line-break extraction
    "re-sult": "result",
    "representa-tion": "representation",
    "informa-tion": "information",
    "pre-sented": "presented",
    "pre-diction": "prediction",
    "evalu-ation": "evaluation",
    "classi-fication": "classification",
}


def _fix_word_boundaries(text: str) -> str:
    """Restore spaces lost to PDF line-break extraction artifacts.

    Two passes:
    1. Regex: split at lowercase→Uppercase boundary ("Thefeature" → "The feature").
       Does NOT split all-caps acronyms (BERT, NLP) — regex requires a lowercase
       char before the uppercase char.
    2. Lookup: replace known common joins from ML papers ("beenused" → "been used").
    """
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    for bad, good in _KNOWN_JOINS.items():
        text = text.replace(bad, good)
    return text


def _merge_short_blocks(blocks: list[dict]) -> list[dict]:
    """Merge adjacent line-level blocks into paragraph-level blocks.

    PDFs exported from Word or with line-based layout produce one block per line
    (2-15 words, no terminal punctuation). sent_tokenize() returns each fragment
    as-is, and _is_likely_claim() rejects it because it lacks terminal punctuation.

    Merges blocks until the buffer ends in terminal punctuation or exceeds 50 words.
    Paragraph-level PDFs (BERT, Attention) flush immediately since each block already
    ends in terminal punctuation — no merging occurs, no regression.
    """
    merged: list[dict] = []
    buffer_text = ""
    buffer_block: dict | None = None
    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue
        buffer_text = (buffer_text + " " + text).strip() if buffer_text else text
        buffer_block = block
        if text[-1] in ".!?" or len(buffer_text.split()) >= 50:
            merged.append({**buffer_block, "text": buffer_text})
            buffer_text = ""
            buffer_block = None
    if buffer_text and buffer_block is not None:
        merged.append({**buffer_block, "text": buffer_text})
    return merged


# Regex patterns for inline citation marker detection.
# Numeric: [1], [1,2], [1, 2-4]
_NUMERIC_CITATION_RE = re.compile(r'\[\d+(?:[,\s\-]+\d+)*\]')
# Author-year parenthesized: (Smith, 2023), (Jones et al., 2020), (A et al., 2020; B, 2021)
_AUTHOR_YEAR_PAREN_RE = re.compile(
    r'\([A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?(?:,\s*\d{4}[a-z]?)?'
    r'(?:[;,]\s*[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?(?:,\s*\d{4}[a-z]?)?)*'
    r',?\s+\d{4}[a-z]?\)'
)
# Author-year bare: Smith (2023), Jones et al. (2020)
_AUTHOR_YEAR_BARE_RE = re.compile(
    r'[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?\s+\(\d{4}[a-z]?\)'
)


def _detect_citation_markers(text: str) -> list[ExtractedCitationMarker]:
    """Detect numeric and author-year citation markers in a sentence.

    The NLTK extractor can't ask an LLM to identify citations, so we use
    regex. Misses footnote superscripts (no reliable plain-text pattern)
    but catches the two dominant styles in academic PDFs.
    """
    markers: list[ExtractedCitationMarker] = []
    seen: set[str] = set()

    for m in _NUMERIC_CITATION_RE.finditer(text):
        raw = m.group()
        if raw not in seen:
            seen.add(raw)
            markers.append(ExtractedCitationMarker(raw_marker=raw, citation_style="numeric"))

    for pattern in (_AUTHOR_YEAR_PAREN_RE, _AUTHOR_YEAR_BARE_RE):
        for m in pattern.finditer(text):
            raw = m.group()
            if raw not in seen:
                seen.add(raw)
                markers.append(ExtractedCitationMarker(raw_marker=raw, citation_style="author_year"))

    return markers


def _is_likely_claim(sentence: str) -> bool:
    """Return True if the sentence looks like a verifiable factual claim."""
    s = sentence.strip()
    if not s:
        return False
    words = s.split()
    if len(words) < 5 or len(words) > 120:
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
    if _ACK_RE.search(s):
        return False
    if _META_RE.match(s):
        return False
    # Must contain at least one verb-like token (ends in -s, -ed, -ing, or "is", "are", "was")
    verb_like = re.search(
        r"\b(is|are|was|were|has|have|had|shows?|demonstrates?|"
        r"achieves?|reduces?|increases?|decreases?|improves?|"
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

    # Merge line-level blocks (Word-exported PDFs) into paragraph-level blocks so
    # sent_tokenize receives complete sentences with terminal punctuation.
    citation_blocks = _merge_short_blocks(citation_blocks)

    try:
        from nltk.tokenize import sent_tokenize
    except ImportError:
        logger.error("nltk not available — returning empty claims")
        return ClaimExtractionResponse(claims=[])

    all_claims: list[ExtractedClaim] = []
    seen: set[str] = set()

    for block in citation_blocks:
        text = _fix_word_boundaries(block.get("text", "").strip())
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
                citation_markers=_detect_citation_markers(sentence),
            ))

    logger.info("Local extractor produced %d candidate claims from %d blocks",
                len(all_claims), len(citation_blocks))
    return ClaimExtractionResponse(claims=all_claims)
