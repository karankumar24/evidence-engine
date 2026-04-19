"""Pydantic response schemas for claim extraction API responses.

These schemas serialize ORM Claim and CitationAnchor objects for API responses.
They are separate from the extraction schemas (evidenceengine.extraction.schemas)
which are used for LLM structured output.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class CitationAnchorResponse(BaseModel):
    """API response schema for a CitationAnchor row."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    raw_marker: str
    citation_style: str
    target_document_id: uuid.UUID | None
    resolution_status: str


class ClaimResponse(BaseModel):
    """API response schema for a Claim row with its citation anchors."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    claim_text: str
    section_header: str | None
    page_number: int | None
    paragraph_index: int | None
    char_start: int
    char_end: int
    status: str
    citation_anchors: list[CitationAnchorResponse]
    created_at: datetime


class ExtractionResponse(BaseModel):
    """Top-level response for POST /api/packets/{id}/extract."""

    run_version_id: uuid.UUID
    claim_count: int
    claims: list[ClaimResponse]
