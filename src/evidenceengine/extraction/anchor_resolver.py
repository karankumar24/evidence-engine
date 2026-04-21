"""Anchor resolver: maps citation markers to SourceDocument records.

Uses exact reference-section lookup for numeric citations and difflib
fuzzy matching for author-year citations.

Key invariant: NEVER drop a CitationAnchor — if unresolvable, write
resolution_status='unresolvable' to DB. This is enforced by the caller
(pipeline.py), not here. This module only returns (doc_id|None, status).
"""

import difflib
import re

# Minimum fuzzy match score (0-100) to consider a citation resolved on raw
# score alone. Calibrated at 80 to balance precision/recall for author-year
# citations against document filenames and raw text snippets. The score is the
# max of (a) char-level SequenceMatcher ratio and (b) token-coverage of
# candidate tokens within target tokens — see resolve_to_source_document().
RESOLUTION_THRESHOLD = 80

# Secondary pathway: when the top candidate scores above MARGIN_THRESHOLD and
# beats the runner-up by MARGIN_LEAD points, we treat it as resolved. This
# handles the common case where a reference title uses words the source
# excerpt doesn't literally repeat (e.g. "Climate Change, 2022" as a journal
# title, vs. an excerpt that only discusses the specific chapter). If ONE
# source is a much closer match than any other, that's a strong signal even
# below 80. False-positive risk is low because we require BOTH a decent
# absolute score AND a clear lead over alternatives.
MARGIN_THRESHOLD = 65
MARGIN_LEAD = 20

# Token-coverage substantive-token guard: a candidate must contain at least
# one token of length >= 4 to qualify for token-coverage scoring. Prevents
# trivial matches (e.g. candidate "Foo" matching any doc containing "foo").
# 4 chars admits years (2023) and short author surnames (Wang, Park) while
# rejecting stop-word-grade tokens (a, to, of).
_MIN_SUBSTANTIVE_TOKEN_LEN = 4

# Token splitter: word characters of length 2+ (drops punctuation, single chars).
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

# Stopwords stripped from candidates before token-coverage scoring. These
# otherwise drag coverage down when the source document doesn't repeat function
# words present in the reference entry (e.g. candidate "IPCC Sixth Assessment
# Report — Working Group III, Mitigation of Climate Change, 2022" contains
# "of" which source_02_ipcc_carbon_budget.pdf's excerpt doesn't, knocking
# coverage below the 80% threshold even though every substantive proper noun
# matches). Keep this list tight — we only drop true English function words
# plus bibliography-specific noise ("et", "al", "pp", "vol", "eds", "ed").
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "of", "in", "on", "at", "to", "for", "by", "with",
    "from", "as", "an", "is", "it", "its", "be", "are", "was", "were",
    "or", "but", "not", "no", "do", "does", "has", "have", "had",
    "et", "al", "pp", "vol", "eds", "ed", "no", "ser", "issn", "isbn",
})


def _tokens(text: str) -> set[str]:
    """Lowercase token set, length >= 2, alphanumeric only, stopwords removed."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _has_substantive_token(tokens: set[str]) -> bool:
    """True if the token set contains at least one token of length >= 4."""
    return any(len(t) >= _MIN_SUBSTANTIVE_TOKEN_LEN for t in tokens)


# Recognized headings that introduce a references section.
_REFERENCES_HEADERS = frozenset(
    {"references", "bibliography", "works cited", "citations", "literature cited"}
)


def extract_references_entries(parsed_blocks: list[dict]) -> list[str]:
    """Scan parsed_content blocks for a references/bibliography section.

    Returns an ordered list of reference entry strings. Entry at index 0
    corresponds to reference [1], index 1 to [2], etc.

    Stops collecting when the next heading block is encountered after
    the references heading.

    Args:
        parsed_blocks: List of block dicts from parsed_content["blocks"].

    Returns:
        Ordered list of reference entry strings, or [] if no section found.
    """
    in_refs = False
    entries: list[str] = []

    for block in parsed_blocks:
        text_lower = block.get("text", "").strip().lower()
        block_type = block.get("block_type", "")

        # Some PDF parsers don't label headings — accept a standalone paragraph
        # whose entire text is a references-section header as the marker too.
        is_refs_header = text_lower in _REFERENCES_HEADERS

        if block_type == "heading":
            if is_refs_header:
                in_refs = True
                continue
            elif in_refs:
                # Next heading after references section — stop collecting
                break
        elif is_refs_header and not in_refs:
            in_refs = True
            continue

        if in_refs and block.get("text", "").strip():
            entries.append(block["text"].strip())

    return entries


def resolve_to_source_document(
    raw_marker: str,
    citation_style: str,
    source_documents: list,
    references_entries: list[str] | None = None,
) -> tuple[str | None, str]:
    """Resolve a citation marker to a SourceDocument using fuzzy matching.

    For numeric citations: extracts the reference number, looks up the
    corresponding entry in references_entries, then fuzzy-matches that
    entry against source document filenames and raw text snippets.

    For author-year citations: fuzzy-matches the raw marker directly.

    For footnote citations: attempts fuzzy match (usually unresolvable
    since footnote symbols carry no semantic content).

    Args:
        raw_marker: The raw citation marker string (e.g., "[1]", "Smith 2023", "¹").
        citation_style: One of "numeric", "author_year", "footnote".
        source_documents: List of SourceDocument ORM objects (or duck-typed objects
                          with .id, .filename, .is_report, .raw_text attributes).
        references_entries: Ordered reference entries from extract_references_entries().
                            Required for good numeric resolution. May be None.

    Returns:
        Tuple of (source_document_id_str | None, "resolved" | "unresolvable").
        source_document_id_str is str(doc.id) when resolved, None when unresolvable.
    """
    # Filter out report documents — they are never resolution targets
    non_report_docs = [d for d in source_documents if not d.is_report]
    if not non_report_docs:
        return None, "unresolvable"

    candidate = _build_candidate_string(raw_marker, citation_style, references_entries)
    if not candidate:
        return None, "unresolvable"

    best_score = 0.0
    second_score = 0.0
    best_doc_id: str | None = None

    candidate_lower = candidate.lower()
    candidate_tokens = _tokens(candidate_lower)

    for doc in non_report_docs:
        # Clean filename: replace underscores/dots with spaces for better token matching
        filename_clean = doc.filename.replace("_", " ").replace(".", " ")
        # Build target string: cleaned filename + first 2000 chars of raw_text.
        # 2000 covers a typical source excerpt's introductory + first body pages
        # where authors/dates/section titles tend to live, without incurring the
        # O(n^2) SequenceMatcher cost of scanning full documents.
        target = f"{filename_clean} {(doc.raw_text or '')[:2000]}"
        target_lower = target.lower()
        # Combine character-level fuzzy with token-coverage. SequenceMatcher
        # alone misses clean matches when filenames inject extra tokens (e.g.
        # candidate "Smith 2023. Climate" vs target "smith climate 2023 pdf
        # climate ..." scores only 77 — one extra "pdf" token tanks below 80).
        # Token coverage = fraction of candidate tokens present in target,
        # gated by MIN_TOKENS so single-word candidates can't match anything.
        seq_score = difflib.SequenceMatcher(None, candidate_lower, target_lower).ratio() * 100
        target_tokens = _tokens(target_lower)
        if candidate_tokens and _has_substantive_token(candidate_tokens):
            coverage = len(candidate_tokens & target_tokens) / len(candidate_tokens) * 100
        else:
            coverage = 0
        score = max(seq_score, coverage)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_doc_id = str(doc.id)
        elif score > second_score:
            second_score = score

    if best_score >= RESOLUTION_THRESHOLD:
        return best_doc_id, "resolved"
    # Margin-of-separation path: top candidate has a decent score and leaves
    # the runner-up far behind. Typical false-positive shape (two docs scoring
    # similarly both below threshold) is rejected by the LEAD requirement.
    if best_score >= MARGIN_THRESHOLD and (best_score - second_score) >= MARGIN_LEAD:
        return best_doc_id, "resolved"
    return None, "unresolvable"


def _build_candidate_string(
    raw_marker: str,
    citation_style: str,
    references_entries: list[str] | None,
) -> str:
    """Extract a matchable string from a raw marker for fuzzy comparison.

    For numeric: resolves the reference number to its entry text if available.
    For author_year: uses the raw marker directly.
    For footnote: uses the raw marker (usually short, low match probability).
    """
    if citation_style == "numeric":
        m = re.search(r"\d+", raw_marker)
        if m and references_entries:
            idx = int(m.group()) - 1  # [1] → index 0
            if 0 <= idx < len(references_entries):
                return references_entries[idx]
        # Fallback: use the raw marker itself (very low match probability)
        return raw_marker

    elif citation_style == "author_year":
        return raw_marker  # "Smith 2023" or "(Smith et al., 2023)"

    elif citation_style == "footnote":
        return raw_marker  # superscript symbol — low match probability

    return raw_marker
