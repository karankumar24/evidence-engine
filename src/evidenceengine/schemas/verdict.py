"""Pydantic schemas for verdict classification API responses."""

import uuid

from pydantic import BaseModel


class VerdictResponse(BaseModel):
    """Response schema for a single verdict."""

    id: uuid.UUID
    claim_id: uuid.UUID
    verdict_type: str
    confidence_score: float
    reasoning: str
    model_name: str


class ClassificationResponse(BaseModel):
    """Response schema for POST /api/packets/{id}/classify."""

    run_version_id: uuid.UUID
    verdict_count: int
    claim_count: int
