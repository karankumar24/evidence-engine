"""Pydantic schemas for pipeline run endpoints (Phase 5+)."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunTriggerResponse(BaseModel):
    """Response body when a new pipeline run is triggered."""

    run_version_id: uuid.UUID
    status: str  # always "queued" on creation


class RunStatusResponse(BaseModel):
    """Response body for GET /runs/{run_version_id}/status.

    Reads directly from the RunVersion ORM object via model_validate.
    The `id` column is aliased to `run_version_id` to match the API contract.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    run_version_id: uuid.UUID = Field(alias="id")
    packet_id: uuid.UUID
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_summary: str | None = None
    failed_claim_count: int = 0

    @model_validator(mode="before")
    @classmethod
    def extract_failed_count(cls, data: object) -> object:
        """Pull failed_claim_count out of pipeline_config JSONB."""
        if hasattr(data, "pipeline_config") and data.pipeline_config:
            # Pydantic ORM mode: data is the ORM instance — use object.__setattr__ to bypass frozen
            try:
                object.__setattr__(
                    data,
                    "failed_claim_count",
                    data.pipeline_config.get("failed_claim_count", 0),
                )
            except (AttributeError, TypeError):
                pass
        return data


class VerdictResultItem(BaseModel):
    """A single verdict in RunResultsResponse.verdicts."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    verdict_id: uuid.UUID = Field(alias="id")
    claim_id: uuid.UUID
    verdict_type: str
    confidence_score: float
    reasoning: str
    model_name: str


class RunResultsResponse(BaseModel):
    """Response body for GET /runs/{run_version_id}/results."""

    run_version_id: uuid.UUID
    status: str
    verdicts: list[VerdictResultItem]
    claim_errors: list[dict]  # from pipeline_config["claim_errors"]
