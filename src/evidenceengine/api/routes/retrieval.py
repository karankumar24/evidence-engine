"""Retrieval API route: POST /api/packets/{packet_id}/retrieve."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.api.dependencies import get_db
from evidenceengine.core.config import settings
from evidenceengine.models.claim import Claim
from evidenceengine.models.document import DocumentPacket
from evidenceengine.models.run import RunVersion
from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run
from evidenceengine.schemas.common import APIError
from evidenceengine.schemas.evidence import RetrievalResponse

router = APIRouter(prefix="/api/packets", tags=["retrieval"])


@router.post("/{packet_id}/retrieve", status_code=202)
async def trigger_retrieval(
    packet_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> RetrievalResponse:
    """Trigger evidence retrieval for all claims in a packet.

    Finds the most recent RunVersion for the packet, runs retrieval,
    and logs recall@k metrics. Returns 202 with run_version_id and evidence_count.

    Phase 5 will wrap this in an async background task. For now it runs synchronously
    in the request. The 202 status signals intent for future async operation.
    """
    # 1. Load packet — 404 if not found
    result = await db.execute(
        select(DocumentPacket).where(DocumentPacket.id == packet_id)
    )
    packet = result.scalar_one_or_none()
    if packet is None:
        raise APIError(code="PACKET_NOT_FOUND", message="DocumentPacket not found", status=404)

    # 2. Find RunVersion for this packet
    run_result = await db.execute(
        select(RunVersion)
        .where(RunVersion.packet_id == packet_id)
        .order_by(RunVersion.created_at.desc())
    )
    run_version = run_result.scalars().first()

    if run_version is None:
        raise APIError(
            code="NO_RUN_VERSION",
            message="No RunVersion found for this packet. Run extraction first.",
            status=422,
        )

    # 3. Check claims exist
    claim_result = await db.execute(
        select(Claim).where(Claim.run_version_id == run_version.id)
    )
    claims = claim_result.scalars().all()
    claim_count = len(claims)
    if claim_count == 0:
        raise APIError(
            code="NO_CLAIMS",
            message="No claims found for this packet's run. Run extraction first.",
            status=422,
        )

    # 4. Run retrieval pipeline
    run_version.started_at = datetime.now(timezone.utc)
    await db.flush()

    evidence_spans = await retrieve_evidence_for_run(
        run_version_id=str(run_version.id),
        db=db,
    )

    run_version.completed_at = datetime.now(timezone.utc)
    run_version.status = "retrieval_complete"
    run_version.model_versions = {
        **(run_version.model_versions or {}),
        "reranker": settings.reranker_model,
    }

    return RetrievalResponse(
        run_version_id=run_version.id,
        evidence_count=len(evidence_spans),
        claim_count=claim_count,
    )
