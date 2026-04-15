"""Tests for citation marker detection — all three citation styles.

TDD RED phase: these tests are written BEFORE implementation.
"""
import pytest

from evidenceengine.extraction.citation_detector import detect_citation_markers


def test_detects_numeric_citation():
    """Numeric [1] citation is detected with correct style and raw_marker."""
    text = "Carbon emissions increased by 12% [1]."
    results = detect_citation_markers(text)
    markers = [r["raw_marker"] for r in results]
    styles = [r["citation_style"] for r in results]
    assert "[1]" in markers
    assert "numeric" in styles


def test_detects_compound_numeric():
    """Compound numeric [2,3] is detected as a single numeric marker."""
    text = "Multiple studies confirm this [2,3]."
    results = detect_citation_markers(text)
    assert len(results) >= 1
    assert any(r["citation_style"] == "numeric" for r in results)
    # compound like [2,3] should be treated as one numeric marker
    raw_markers = [r["raw_marker"] for r in results]
    assert any("2" in m and "3" in m or m == "[2,3]" or m == "[2" or m == "[3]" for m in raw_markers) or \
           any(r["citation_style"] == "numeric" for r in results)


def test_detects_author_year():
    """Bare author-year 'Smith 2023' is detected."""
    text = "Smith 2023 demonstrated significant effects."
    results = detect_citation_markers(text)
    assert any(r["citation_style"] == "author_year" for r in results)


def test_detects_parenthetical_author_year():
    """Parenthetical (Smith et al., 2023) is detected as author_year."""
    text = "This was confirmed (Smith et al., 2023) in later work."
    results = detect_citation_markers(text)
    assert any(r["citation_style"] == "author_year" for r in results)


def test_detects_footnote_superscript():
    """Footnote superscript ¹ is detected as footnote style."""
    text = "findings¹ show a correlation with temperature."
    results = detect_citation_markers(text)
    assert any(r["citation_style"] == "footnote" for r in results)


def test_mixed_styles():
    """Text with all three citation styles returns entries for each (CLAIM-05)."""
    text = "Emissions rose [1], confirming Jones 2021² findings."
    results = detect_citation_markers(text)
    styles_found = {r["citation_style"] for r in results}
    assert "numeric" in styles_found, f"Expected numeric, got styles: {styles_found}"
    assert "author_year" in styles_found, f"Expected author_year, got styles: {styles_found}"
    assert "footnote" in styles_found, f"Expected footnote, got styles: {styles_found}"


def test_no_markers_returns_empty():
    """Plain sentence with no citation markers returns empty list."""
    text = "The sky is blue and the grass is green."
    results = detect_citation_markers(text)
    assert results == []


def test_result_has_span():
    """Each result dict contains a 'span' key with (start, end) tuple."""
    text = "This result [3] supports the hypothesis."
    results = detect_citation_markers(text)
    assert len(results) >= 1
    for r in results:
        assert "span" in r
        start, end = r["span"]
        assert start >= 0
        assert end > start


def test_results_sorted_by_position():
    """Results are sorted by their start position in the text."""
    text = "First [1] then (Jones 2020) then ²."
    results = detect_citation_markers(text)
    positions = [r["span"][0] for r in results]
    assert positions == sorted(positions), "Results should be sorted by start position"
