"""Extraction API endpoint.

POST /api/packets/{packet_id}/extract — Trigger claim extraction for a packet.
Returns 202 Accepted with run metadata (extraction runs synchronously in this phase;
Phase 5 will move this to an async task queue).
"""

import uuid as uuid_module
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from evidenceengine.api.dependencies import get_db
from evidenceengine.core.config import settings
from evidenceengine.extraction.pipeline import extract_claims_for_document
from evidenceengine.models.claim import Claim, CitationAnchor
from evidenceengine.models.document import DocumentPacket, SourceDocument
from evidenceengine.models.run import RunVersion
from evidenceengine.schemas.claim import ClaimResponse, ExtractionResponse

router = APIRouter(prefix="/api/packets", tags=["extraction"])


@router.post("/{packet_id}/extract", status_code=202)
async def trigger_extraction(
    packet_id: uuid_module.UUID,
    db: AsyncSession = Depends(get_db),
) -> ExtractionResponse:
    """Trigger claim extraction for an uploaded packet.

    Finds the report document, runs the full extraction pipeline,
    and returns the extraction results with persisted claim rows.

    Returns 202 Accepted on success.
    Returns 404 if the packet is not found.
    Returns 422 if the packet has no report document.
    """
    # 1. Load packet — 404 if not found
    result = await db.execute(
        select(DocumentPacket).where(DocumentPacket.id == packet_id)
    )
    packet = result.scalar_one_or_none()
    if packet is None:
        raise HTTPException(status_code=404, detail=f"Packet {packet_id} not found")

    # 2. Find the report document in this packet
    result = await db.execute(
        select(SourceDocument).where(
            SourceDocument.packet_id == packet_id,
            SourceDocument.is_report.is_(True),
        )
    )
    report_doc = result.scalar_one_or_none()
    if report_doc is None:
        raise HTTPException(
            status_code=422,
            detail=f"Packet {packet_id} has no report document (is_report=True)",
        )

    # 3. Create RunVersion to track this extraction run
    run_version = RunVersion(
        packet_id=packet_id,
        status="running",
        pipeline_config={
            "phase": "claim_extraction",
            "model": settings.extraction_model,
        },
        started_at=datetime.now(timezone.utc),
    )
    db.add(run_version)
    await db.flush()  # populate run_version.id

    # 4. Run the extraction pipeline (pipeline handles its own commit)
    claims = await extract_claims_for_document(
        report_document_id=str(report_doc.id),
        run_version_id=str(run_version.id),
        db=db,
    )

    # 5. Update RunVersion with completion status
    run_version.status = "completed"
    run_version.completed_at = datetime.now(timezone.utc)
    await db.commit()

    # 6. Reload claims with citation_anchors eagerly (avoid N+1 queries)
    if claims:
        claim_ids = [c.id for c in claims]
        result = await db.execute(
            select(Claim)
            .where(Claim.id.in_(claim_ids))
            .options(selectinload(Claim.citation_anchors))
        )
        loaded_claims = list(result.scalars().all())
    else:
        loaded_claims = []

    return ExtractionResponse(
        run_version_id=run_version.id,
        claim_count=len(loaded_claims),
        claims=[ClaimResponse.model_validate(c) for c in loaded_claims],
    )
