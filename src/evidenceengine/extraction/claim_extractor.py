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
    # ML paper compound joins seen in BERT/Attention papers (no hyphen, just
    # two words fused by a PyMuPDF line-break that dropped the space)
    "empiricallypowerful": "empirically powerful",
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

    Three passes:
    1. Regex: split at lowercase→Uppercase boundary ("Thefeature" → "The feature").
       Does NOT split all-caps acronyms (BERT, NLP) — regex requires a lowercase
       char before the uppercase char.
    2. Hyphen removal: delete mid-word hyphens before common word suffixes that
       appear only from line-break hyphenation, never in real compound words
       (e.g. "repre-sentation" → "representation", "classi-fied" → "classified").
       Leaves legitimate hyphens ("state-of-the-art", "fine-tuned") untouched.
    3. Lookup: case-insensitive replace of known common joins from ML papers
       ("beenused" → "been used", "Inthe" → "In the"). Preserves leading
       capitalisation so sentence-start joins stay capitalised.
    """
    # Pass 0: sentence boundary — period/comma touching a capital letter with no space
    # "powerful.It" → "powerful. It", "result,The" → "result, The"
    text = re.sub(r'([a-z])([.,])([A-Z])', r'\1\2 \3', text)
    # Pass 1: camelCase boundary split
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Pass 1.5: proper-noun line-break hyphenation ("Rad-ford" → "Radford").
    # Guard excludes function words ("State-of" stays "State-of") and long parts
    # (> 5 chars after hyphen are likely real compound word halves, not name fragments).
    _keep_after = {'of', 'in', 'to', 'by', 'at', 'an', 'the', 'a', 'is', 'or', 'as', 'on'}
    def _dehyphenate_proper(m: re.Match) -> str:
        return m.group(0) if m.group(2).lower() in _keep_after else m.group(1) + m.group(2)
    text = re.sub(r'([A-Z][a-z]{2,})-([a-z]{2,5})\b', _dehyphenate_proper, text)
    # Pass 2: remove hyphens before suffixes that are never real compound parts
    # Suffixes: -tion, -sion, -ation, -ization, -ment, -ness, -ful, -tion, -ance
    text = re.sub(r'([a-z]{3,})-(sentation|tion|sion|ation|ization|ment|ness|ful|ance|ence|ture|ive|ary|ory|able|ible|ly)\b',
                  r'\1\2', text, flags=re.IGNORECASE)
    # Pass 3: case-insensitive lookup replacements
    for bad, good in _KNOWN_JOINS.items():
        def _replace(m: re.Match, _good: str = good) -> str:
            return _good[0].upper() + _good[1:] if m.group(0)[0].isupper() else _good
        text = re.sub(re.escape(bad), _replace, text, flags=re.IGNORECASE)
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


_MIN_CLAIM_WORDS = 3    # lowered from 5 — catches short metric claims
_MAX_CLAIM_WORDS = 150  # raised from 120 — captures long technical sentences
_MIN_NOPUNCT_WORDS = 8  # min words to accept a sentence lacking terminal punctuation

_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|shows?|demonstrates?|"
    r"achieves?|reduces?|increases?|decreases?|improves?|"
    r"suggests?|indicates?|contains?|provides?|results?|found|"
    r"\w+ed|\w+ing)\b"
)
_METRIC_RE = re.compile(r":\s*[\d\.\-\+]")  # "Accuracy: 95%." style


def _is_likely_claim(sentence: str, _counts: dict | None = None) -> bool:
    """Return True if the sentence looks like a verifiable factual claim.

    _counts: optional mutable dict tracking per-reason rejection counts
             for diagnostic generation when 0 claims are extracted.
    """
    def reject(reason: str) -> bool:
        if _counts is not None:
            _counts[reason] = _counts.get(reason, 0) + 1
        return False

    s = sentence.strip()
    if not s:
        return reject("empty")
    words = s.split()
    if len(words) < _MIN_CLAIM_WORDS or len(words) > _MAX_CLAIM_WORDS:
        return reject("length")
    if not s[0].isupper():
        return reject("lowercase_start")
    lower = s.lower()
    if any(lower.startswith(pfx) for pfx in _SKIP_PREFIXES):
        return reject("skip_prefix")
    if _STRUCTURAL_RE.match(s):
        return reject("structural")
    if _ACK_RE.search(s):
        return reject("acknowledgement")
    if _META_RE.match(s):
        return reject("meta")

    has_punct = s[-1] in ".!?"
    has_verb = bool(_VERB_RE.search(lower))

    if not has_punct:
        # Slide/bullet-point style: accept if long enough AND has a verb.
        # Handles PowerPoint PDFs, Beamer slides, docs without terminal periods.
        if len(words) >= _MIN_NOPUNCT_WORDS and has_verb:
            return True
        return reject("no_terminal_punct")

    if not has_verb:
        # Metric/data-sheet style: accept "Accuracy: 95%." patterns.
        if _METRIC_RE.search(s):
            return True
        return reject("no_verb")

    return True


_ZERO_CLAIM_MESSAGES: dict[str, str] = {
    "no_terminal_punct": (
        "Most sentences lack terminal punctuation. "
        "This PDF may be a slide deck, use heavy bullet-point formatting, "
        "or was exported from a format that strips sentence endings."
    ),
    "no_verb": (
        "Most sentences lack a verb. "
        "Common in data-heavy documents, spreadsheets exported to PDF, "
        "or files dominated by numeric tables and captions."
    ),
    "length": (
        "Sentences are either too short (under 3 words) or very long (over 150 words). "
        "The document may consist mostly of headings, footnotes, or run-on text blocks."
    ),
    "structural": (
        "Content appears to be mostly structural: numbered lists, bullet markers, "
        "or ALL-CAPS headings that don't read as factual claims."
    ),
    "skip_prefix": (
        "Content is dominated by figures, tables, references, or acknowledgements — "
        "all filtered out as non-claim content by design."
    ),
    "lowercase_start": (
        "Many text fragments start with lowercase letters, which usually indicates "
        "parsing artifacts or incomplete sentences from PDF extraction."
    ),
    "meta": (
        "Content is mostly self-referential commentary (\"In this paper...\") "
        "rather than verifiable factual claims."
    ),
    "acknowledgement": "Content is mostly acknowledgements or funding statements.",
}


def _make_zero_claims_diagnostic(counts: dict[str, int], total_candidates: int) -> str:
    """Build a user-readable explanation for why 0 claims were extracted."""
    if total_candidates == 0:
        return (
            "The document produced no text blocks. "
            "It may be a scanned image-only PDF with no embedded text, "
            "or the file may be empty or corrupt. "
            "Try re-exporting as a PDF with selectable text."
        )
    top = max(counts, key=counts.__getitem__) if counts else "unknown"
    return _ZERO_CLAIM_MESSAGES.get(
        top,
        f"No verifiable claims were found across {total_candidates} candidate sentences.",
    )


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
        return ClaimExtractionResponse(
            claims=[],
            diagnostic=_make_zero_claims_diagnostic({}, 0),
        )

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
    rejection_counts: dict[str, int] = {}
    total_candidates = 0

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
            total_candidates += 1
            if not _is_likely_claim(sentence, rejection_counts):
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

    logger.info(
        "Local extractor produced %d claims from %d candidates across %d blocks",
        len(all_claims), total_candidates, len(citation_blocks),
    )

    if not all_claims:
        diagnostic = _make_zero_claims_diagnostic(rejection_counts, total_candidates)
        logger.warning("0 claims extracted — %s", diagnostic)
        return ClaimExtractionResponse(claims=[], diagnostic=diagnostic)

    return ClaimExtractionResponse(claims=all_claims)
