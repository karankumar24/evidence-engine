"""Pydantic schemas for evidence retrieval API responses."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EvidenceSpanResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    claim_id: uuid.UUID
    source_document_id: uuid.UUID
    span_text: str
    page_number: int | None
    paragraph_index: int | None
    section_header: str | None
    relevance_score: float | None
    retrieval_rank: int | None
    retrieval_method: str
    created_at: datetime


class RetrievalResponse(BaseModel):
    run_version_id: uuid.UUID
    evidence_count: int
    claim_count: int
