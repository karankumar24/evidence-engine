"""POST /api/packets/{packet_id}/run — trigger full async pipeline."""

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.api.dependencies import get_db
from evidenceengine.models.document import DocumentPacket
from evidenceengine.models.run import RunVersion
from evidenceengine.pipeline.orchestrator import run_full_pipeline
from evidenceengine.schemas.common import APIError
from evidenceengine.schemas.run import RunTriggerResponse

router = APIRouter(prefix="/api/packets", tags=["pipeline"])


@router.post("/{packet_id}/run", status_code=202, response_model=RunTriggerResponse)
async def trigger_full_run(
    packet_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> RunTriggerResponse:
    """Trigger a full pipeline run for a document packet.

    Returns immediately with 202 Accepted. The pipeline runs in the background.
    Poll GET /api/runs/{run_version_id} to track progress.
    """
    # 1. Validate packet exists
    packet = await db.get(DocumentPacket, packet_id)
    if packet is None:
        raise APIError(
            code="PACKET_NOT_FOUND",
            message=f"Packet {packet_id} not found",
            status=404,
        )

    # 2. Create RunVersion — commit before scheduling so the background task's
    #    independent session can see the row (it opens after the request completes).
    run = RunVersion(packet_id=packet_id, status="queued")
    db.add(run)
    await db.flush()
    await db.commit()  # must commit before background task opens its own session

    # 3. Schedule background task with primitive string ID only — NEVER pass db session
    background_tasks.add_task(run_full_pipeline, str(run.id))

    return RunTriggerResponse(run_version_id=run.id, status="queued")
