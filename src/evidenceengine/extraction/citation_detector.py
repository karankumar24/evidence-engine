"""Citation marker detection using regular expressions.

Detects three citation styles:
- Numeric:     [1], [2,3], [1-3]
- Author-year: Smith 2023, (Smith et al., 2023), Jones and Lee 2021
- Footnote:    ¹, ², ³, superscript Unicode characters

Known limitation: bare "Proper Noun Year" patterns (e.g., "Paris 2015 Agreement")
may be matched by the author_year pattern. This is a regex trade-off documented
in the plan; users should validate author_year matches in context.
"""

import re

# Numeric: [1], [2,3], [1, 3], [2–4] (en-dash), [2-4] (hyphen)
NUMERIC_PATTERN = re.compile(r"\[(\d+(?:[,\s\u2013\-]\s*\d+)*)\]")

# Author-year: Smith 2023, Smith and Jones 2023, Smith et al. 2023, (Smith et al., 2023)
AUTHOR_YEAR_PATTERN = re.compile(
    r"\(?"
    r"([A-Z][A-Za-z\-]+"
    r"(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-]+)?"
    r"(?:\s+et\s+al\.?)?),"
    r"?\s+"
    r"((?:19|20)\d{2}[a-z]?)"
    r"\)?"
)

# Footnote superscripts: ¹²³⁴⁵⁶⁷⁸⁹ and Unicode superscript digits
FOOTNOTE_PATTERN = re.compile(
    r"[¹²³⁴⁵⁶⁷⁸⁹\u2070\u00B9\u00B2\u00B3\u2074-\u2079]+"
)


def detect_citation_markers(text: str) -> list[dict]:
    """Detect all citation markers in text and return them sorted by position.

    Args:
        text: Raw text to scan for citation markers.

    Returns:
        List of dicts, each with:
            - raw_marker (str): verbatim matched text
            - citation_style (str): "numeric" | "author_year" | "footnote"
            - span (tuple[int, int]): (char_start, char_end) in text
        Sorted by span[0] ascending (order of appearance).
    """
    results: list[dict] = []

    for m in NUMERIC_PATTERN.finditer(text):
        results.append(
            {"raw_marker": m.group(), "citation_style": "numeric", "span": m.span()}
        )

    for m in AUTHOR_YEAR_PATTERN.finditer(text):
        results.append(
            {"raw_marker": m.group(), "citation_style": "author_year", "span": m.span()}
        )

    for m in FOOTNOTE_PATTERN.finditer(text):
        results.append(
            {"raw_marker": m.group(), "citation_style": "footnote", "span": m.span()}
        )

    return sorted(results, key=lambda x: x["span"][0])
