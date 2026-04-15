"""Anchor resolver: maps citation markers to SourceDocument records.

Uses exact reference-section lookup for numeric citations and RapidFuzz
fuzzy matching for author-year citations.

Key invariant: NEVER drop a CitationAnchor — if unresolvable, write
resolution_status='unresolvable' to DB. This is enforced by the caller
(pipeline.py), not here. This module only returns (doc_id|None, status).
"""

import re

from rapidfuzz import fuzz

# Minimum fuzzy match score to consider a citation resolved.
# Research-backed: token_sort_ratio >= 80 balances precision/recall for
# author-year citations against document filenames and raw text snippets.
RESOLUTION_THRESHOLD = 80

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

        if block_type == "heading":
            if text_lower in _REFERENCES_HEADERS:
                in_refs = True
                continue
            elif in_refs:
                # Next heading after references section — stop collecting
                break

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

    best_score = 0
    best_doc_id: str | None = None

    for doc in non_report_docs:
        # Clean filename: replace underscores/dots with spaces for better token matching
        filename_clean = doc.filename.replace("_", " ").replace(".", " ")
        # Build target string: cleaned filename + first 500 chars of raw_text
        target = f"{filename_clean} {(doc.raw_text or '')[:500]}"
        # WRatio combines multiple algorithms — best for mixed filename/text matching
        score = fuzz.WRatio(candidate, target)
        if score > best_score:
            best_score = score
            best_doc_id = str(doc.id)

    if best_score >= RESOLUTION_THRESHOLD:
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
