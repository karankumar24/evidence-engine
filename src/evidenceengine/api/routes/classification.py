"""Classification API route: POST /api/packets/{packet_id}/classify."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.api.dependencies import get_db
from evidenceengine.classification.pipeline import classify_verdicts_for_run
from evidenceengine.core.config import settings
from evidenceengine.models.claim import Claim
from evidenceengine.models.document import DocumentPacket
from evidenceengine.models.run import RunVersion
from evidenceengine.schemas.verdict import ClassificationResponse

router = APIRouter(prefix="/api/packets", tags=["classification"])


@router.post("/{packet_id}/classify", status_code=202)
async def trigger_classification(
    packet_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ClassificationResponse:
    """Trigger verdict classification for all claims in a packet.

    Finds the most recent RunVersion for the packet, runs classification,
    and updates RunVersion lifecycle fields. Returns 202 with run_version_id,
    verdict_count, and claim_count.

    Phase 5 will wrap this in an async background task. For now it runs synchronously
    in the request. The 202 status signals intent for future async operation.
    """
    # 1. Load packet — 404 if not found
    result = await db.execute(
        select(DocumentPacket).where(DocumentPacket.id == packet_id)
    )
    packet = result.scalar_one_or_none()
    if packet is None:
        raise HTTPException(status_code=404, detail="DocumentPacket not found")

    # 2. Find most recent RunVersion for this packet — 422 if none
    run_result = await db.execute(
        select(RunVersion)
        .where(RunVersion.packet_id == packet_id)
        .order_by(RunVersion.created_at.desc())
    )
    run_version = run_result.scalars().first()

    if run_version is None:
        raise HTTPException(
            status_code=422,
            detail="No RunVersion found for this packet. Run extraction first.",
        )

    # 3. Check claims exist — 422 if zero
    claim_result = await db.execute(
        select(Claim).where(Claim.run_version_id == run_version.id)
    )
    claims = claim_result.scalars().all()
    claim_count = len(claims)
    if claim_count == 0:
        raise HTTPException(
            status_code=422,
            detail="No claims found for this packet's run. Run extraction first.",
        )

    # 4. Mark run as started
    run_version.started_at = datetime.now(timezone.utc)
    await db.flush()

    # 5. Run classification pipeline
    verdicts = await classify_verdicts_for_run(
        run_version_id=str(run_version.id),
        db=db,
    )

    # 6. Update RunVersion lifecycle fields (pipeline does NOT manage these)
    run_version.completed_at = datetime.now(timezone.utc)
    run_version.status = "classification_complete"
    run_version.model_versions = {
        **(run_version.model_versions or {}),
        "classification_model": settings.classification_model,
    }
    await db.commit()

    return ClassificationResponse(
        run_version_id=run_version.id,
        verdict_count=len(verdicts),
        claim_count=claim_count,
    )
