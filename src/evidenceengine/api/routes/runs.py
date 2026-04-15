"""GET /api/runs/{run_id} — status; GET /api/runs/{run_id}/results — full verdicts."""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.api.dependencies import get_db
from evidenceengine.models.run import RunVersion
from evidenceengine.models.verdict import Verdict
from evidenceengine.schemas.common import APIError
from evidenceengine.schemas.run import RunResultsResponse, RunStatusResponse, VerdictResultItem

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("/{run_id}", response_model=RunStatusResponse, response_model_by_alias=False)
async def get_run_status(
    run_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> RunStatusResponse:
    """Return current status of a pipeline run.

    Includes started_at, completed_at, failed_claim_count, and error_summary.
    """
    run = await db.get(RunVersion, run_id)
    if run is None:
        raise APIError(code="RUN_NOT_FOUND", message=f"Run {run_id} not found", status=404)
    return RunStatusResponse.model_validate(run)


@router.get("/{run_id}/results", response_model=RunResultsResponse)
async def get_run_results(
    run_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> RunResultsResponse:
    """Return full verdict list and claim-level errors for a completed pipeline run.

    claim_errors is populated from pipeline_config["claim_errors"] — empty list if none.
    """
    run = await db.get(RunVersion, run_id)
    if run is None:
        raise APIError(code="RUN_NOT_FOUND", message=f"Run {run_id} not found", status=404)

    result = await db.execute(select(Verdict).where(Verdict.run_version_id == run_id))
    verdicts = result.scalars().all()

    claim_errors = (run.pipeline_config or {}).get("claim_errors", [])

    return RunResultsResponse(
        run_version_id=run.id,
        status=run.status,
        verdicts=[VerdictResultItem.model_validate(v) for v in verdicts],
        claim_errors=claim_errors,
    )
