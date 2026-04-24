"""Pydantic models for LLM structured output from claim extraction.

These models are used as response_format in AsyncOpenAI.chat.completions.parse()
so they must be plain Pydantic BaseModel subclasses (not SQLAlchemy models).
"""

from pydantic import BaseModel


class ExtractedCitationMarker(BaseModel):
    """A single citation marker found within a claim sentence."""

    raw_marker: str
    """Verbatim citation marker as it appears in the text.
    Examples: "[1]", "Smith 2023", "(Jones et al., 2020)", "¹"
    """

    citation_style: str
    """One of: "numeric" | "author_year" | "footnote"."""


class ExtractedClaim(BaseModel):
    """A single claim sentence extracted from the report text."""

    claim_text: str
    """Verbatim claim sentence(s), word-for-word from the report. No paraphrasing."""

    citation_markers: list[ExtractedCitationMarker]
    """All citation markers found within this claim sentence."""


class ClaimExtractionResponse(BaseModel):
    """Top-level structured output from the LLM extraction call."""

    claims: list[ExtractedClaim]
    """Only sentences that carry at least one citation marker. Uncited sentences excluded."""

    diagnostic: str | None = None
    """Human-readable explanation of why 0 claims were extracted, if applicable."""
