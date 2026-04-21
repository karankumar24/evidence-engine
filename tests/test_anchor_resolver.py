"""Tests for anchor resolver — pure unit tests, no DB, no LLM.

TDD RED phase: written BEFORE implementation.
All tests use plain Python objects (MagicMock for SourceDocument attributes).
"""
import uuid
from unittest.mock import MagicMock

import pytest


def make_source_doc(filename: str, is_report: bool = False, raw_text: str = "") -> MagicMock:
    """Create a mock SourceDocument with the given attributes."""
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.filename = filename
    doc.is_report = is_report
    doc.raw_text = raw_text
    return doc


def make_parsed_blocks(entries: list[str], before_heading: str = "Introduction") -> list[dict]:
    """Create parsed_content blocks with a References section."""
    blocks = [
        {"text": before_heading, "block_type": "heading", "position": {}},
        {"text": "Some intro text.", "block_type": "paragraph", "position": {}},
        {"text": "References", "block_type": "heading", "position": {}},
    ]
    for entry in entries:
        blocks.append({"text": entry, "block_type": "paragraph", "position": {}})
    return blocks


# ---------------------------------------------------------------------------
# extract_references_entries tests
# ---------------------------------------------------------------------------

def test_extract_references_entries_finds_section():
    """extract_references_entries returns ordered list of entries under References heading."""
    from evidenceengine.extraction.anchor_resolver import extract_references_entries

    parsed_blocks = make_parsed_blocks([
        "Smith J. 2023. Climate data analysis.",
        "Jones A. 2021. Temperature trends.",
    ])
    entries = extract_references_entries(parsed_blocks)
    assert len(entries) == 2
    assert "Smith" in entries[0]
    assert "Jones" in entries[1]


def test_extract_references_entries_stops_at_next_heading():
    """Stops collecting entries when the next heading is encountered after References."""
    from evidenceengine.extraction.anchor_resolver import extract_references_entries

    blocks = [
        {"text": "References", "block_type": "heading", "position": {}},
        {"text": "Smith 2023.", "block_type": "paragraph", "position": {}},
        {"text": "Appendix", "block_type": "heading", "position": {}},  # stop here
        {"text": "This should NOT be included.", "block_type": "paragraph", "position": {}},
    ]
    entries = extract_references_entries(blocks)
    assert len(entries) == 1
    assert "Smith" in entries[0]


def test_extract_references_entries_empty_when_no_section():
    """Returns empty list when no References heading found."""
    from evidenceengine.extraction.anchor_resolver import extract_references_entries

    blocks = [
        {"text": "Introduction", "block_type": "heading", "position": {}},
        {"text": "Some text.", "block_type": "paragraph", "position": {}},
    ]
    entries = extract_references_entries(blocks)
    assert entries == []


def test_extract_references_entries_paragraph_header():
    """'References' marker in a paragraph block (not heading) still starts the section.

    PDF parsers that don't detect headings classify 'References' as a paragraph.
    The extractor must still recognize it and collect subsequent entries.
    """
    from evidenceengine.extraction.anchor_resolver import extract_references_entries

    blocks = [
        {"text": "Body text with [1] marker.", "block_type": "paragraph", "position": {}},
        {"text": "References", "block_type": "paragraph", "position": {}},
        {"text": "[1] IPCC Sixth Assessment Report, 2022", "block_type": "paragraph", "position": {}},
        {"text": "[2] Global Carbon Project, Friedlingstein et al.", "block_type": "paragraph", "position": {}},
    ]
    entries = extract_references_entries(blocks)
    assert len(entries) == 2
    assert entries[0].startswith("[1]")
    assert entries[1].startswith("[2]")


def test_extract_references_entries_bibliography_alias():
    """'Bibliography' heading is also recognized as a references section."""
    from evidenceengine.extraction.anchor_resolver import extract_references_entries

    blocks = [
        {"text": "Bibliography", "block_type": "heading", "position": {}},
        {"text": "Adams B. 2020. Glacial retreat.", "block_type": "paragraph", "position": {}},
    ]
    entries = extract_references_entries(blocks)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# resolve_to_source_document tests
# ---------------------------------------------------------------------------

def test_resolves_numeric_by_references_section():
    """Numeric [1] resolves to first reference entry matched against source doc (CLAIM-04)."""
    from evidenceengine.extraction.anchor_resolver import resolve_to_source_document

    # [1] maps to index 0 → "Smith 2023. Climate data."
    references_entries = ["Smith 2023. Climate data analysis."]
    smith_doc = make_source_doc(filename="smith_climate_2023.pdf", raw_text="Climate data analysis paper.")
    sources = [smith_doc]

    doc_id, status = resolve_to_source_document(
        raw_marker="[1]",
        citation_style="numeric",
        source_documents=sources,
        references_entries=references_entries,
    )
    assert status == "resolved", f"Expected 'resolved' but got '{status}'"
    assert doc_id == str(smith_doc.id)


def test_resolves_author_year_fuzzy():
    """Author-year marker fuzzy-matches source doc filename with score >=80 (CLAIM-04)."""
    from evidenceengine.extraction.anchor_resolver import resolve_to_source_document

    smith_doc = make_source_doc(filename="smith_climate_2023.pdf")
    sources = [smith_doc]

    doc_id, status = resolve_to_source_document(
        raw_marker="Smith 2023",
        citation_style="author_year",
        source_documents=sources,
        references_entries=None,
    )
    assert status == "resolved", f"Expected resolved, got '{status}'"
    assert doc_id == str(smith_doc.id)


def test_unresolvable_when_no_match():
    """Returns (None, 'unresolvable') when no source doc matches the marker (CLAIM-06)."""
    from evidenceengine.extraction.anchor_resolver import resolve_to_source_document

    unrelated_doc = make_source_doc(filename="totally_unrelated_paper_xyz.pdf")
    sources = [unrelated_doc]

    doc_id, status = resolve_to_source_document(
        raw_marker="Jones 2020",
        citation_style="author_year",
        source_documents=sources,
        references_entries=None,
    )
    assert status == "unresolvable"
    assert doc_id is None


def test_margin_path_resolves_clear_winner_below_threshold():
    """Below the 80 threshold but ≥65 with ≥20pt lead → resolved (disambiguation signal)."""
    from evidenceengine.extraction.anchor_resolver import resolve_to_source_document

    # Candidate mentions IPCC explicitly; source_02 filename matches IPCC but
    # its excerpt doesn't repeat "climate/change/2022" — a common real-world
    # mismatch where the title-level reference exceeds the excerpt's vocabulary.
    references_entries = [
        "IPCC Sixth Assessment Report, Working Group III, Mitigation of Climate Change, 2022"
    ]
    ipcc_doc = make_source_doc(
        filename="source_02_ipcc_carbon_budget.pdf",
        raw_text="IPCC Sixth Assessment Report Working Group III Chapter 3 Mitigation Pathways",
    )
    other_doc = make_source_doc(
        filename="source_01_unrelated_topic.pdf",
        raw_text="Totally different content about fisheries and aquaculture.",
    )
    doc_id, status = resolve_to_source_document(
        raw_marker="[1]",
        citation_style="numeric",
        source_documents=[ipcc_doc, other_doc],
        references_entries=references_entries,
    )
    assert status == "resolved"
    assert doc_id == str(ipcc_doc.id)


def test_margin_path_rejects_when_two_candidates_tie():
    """Two source docs score similarly below threshold → unresolvable (ambiguous)."""
    from evidenceengine.extraction.anchor_resolver import resolve_to_source_document

    # Both docs contain overlapping tokens with the candidate; neither stands out.
    references_entries = ["Generic Research Paper, Smith et al., 2023"]
    doc_a = make_source_doc(filename="paper.pdf", raw_text="Generic research paper content.")
    doc_b = make_source_doc(filename="article.pdf", raw_text="Generic research paper content.")
    doc_id, status = resolve_to_source_document(
        raw_marker="[1]",
        citation_style="numeric",
        source_documents=[doc_a, doc_b],
        references_entries=references_entries,
    )
    # Near-identical targets → best - second < MARGIN_LEAD → reject
    assert status == "unresolvable"
    assert doc_id is None


def test_skips_report_document():
    """Source docs with is_report=True are never returned as resolution targets."""
    from evidenceengine.extraction.anchor_resolver import resolve_to_source_document

    report_doc = make_source_doc(filename="smith_2023_report.pdf", is_report=True)
    sources = [report_doc]

    doc_id, status = resolve_to_source_document(
        raw_marker="Smith 2023",
        citation_style="author_year",
        source_documents=sources,
        references_entries=None,
    )
    assert status == "unresolvable"
    assert doc_id is None


def test_returns_unresolvable_with_no_sources():
    """Returns (None, 'unresolvable') when source_documents list is empty."""
    from evidenceengine.extraction.anchor_resolver import resolve_to_source_document

    doc_id, status = resolve_to_source_document(
        raw_marker="[1]",
        citation_style="numeric",
        source_documents=[],
        references_entries=["Smith 2023"],
    )
    assert status == "unresolvable"
    assert doc_id is None
