"""Dashboard routes — GET /dashboard/{packet_id}/{run_id} and partial endpoints."""

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from evidenceengine.api.dependencies import get_db
from evidenceengine.api.services.dashboard_queries import (
    load_claim_detail,
    load_dashboard_context,
    load_runs_index,
    upsert_review_decision,
)
from evidenceengine.core.config import settings
from evidenceengine.models.document import SourceDocument
from evidenceengine.schemas.common import APIError

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_index(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Index page — lists recent pipeline runs for THIS visitor (or demos).

    Packets are filtered by the visitor's session cookie. Demo packets
    (is_demo=True) are visible to every visitor so the dashboard isn't
    empty on first load.
    """
    session_id = getattr(request.state, "session_id", "") or ""
    runs = await load_runs_index(db, session_id=session_id)
    return templates.TemplateResponse(
        request=request,
        name="dashboard_index.html",
        context={"runs": runs, "debug": settings.debug},
    )


@router.get("/dashboard/{packet_id}/{run_id}", response_class=HTMLResponse)
async def dashboard_view(
    request: Request,
    packet_id: uuid.UUID,
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Full dashboard page — loads packet context and renders dashboard.html."""
    context = await load_dashboard_context(packet_id, run_id, db)
    if context is None:
        raise APIError(
            code="NOT_FOUND",
            message=f"Packet {packet_id} or run {run_id} not found",
            status=404,
        )
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "packet": context["packet"],
            "run": context["run"],
            "claims": context["claims"],
            "distribution": context["distribution"],
            "source_docs": context["source_docs"],
            "active_filters": {
                "verdict_filter": None,
                "confidence": None,
                "source_doc_id": None,
            },
        },
    )


@router.get("/dashboard/{packet_id}/{run_id}/queue", response_class=HTMLResponse)
async def queue_partial(
    request: Request,
    packet_id: uuid.UUID,
    run_id: uuid.UUID,
    verdict_filter: str | None = None,
    confidence: str | None = None,
    source_doc_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Queue partial — filtered list of claims for HTMX queue updates."""
    context = await load_dashboard_context(packet_id, run_id, db)
    if context is None:
        raise APIError(
            code="NOT_FOUND",
            message=f"Packet {packet_id} or run {run_id} not found",
            status=404,
        )

    claims = context["claims"]

    # Apply verdict_filter
    if verdict_filter:
        claims = [
            c
            for c in claims
            if any(v.verdict_type == verdict_filter for v in c.verdicts)
        ]

    # Apply confidence band filter based on first verdict
    if confidence:
        filtered = []
        for claim in claims:
            if not claim.verdicts:
                continue
            score = claim.verdicts[0].confidence_score
            if confidence == "high" and score >= 0.8:
                filtered.append(claim)
            elif confidence == "medium" and 0.5 <= score < 0.8:
                filtered.append(claim)
            elif confidence == "low" and score < 0.5:
                filtered.append(claim)
        claims = filtered

    # Apply source_doc_id filter
    if source_doc_id:
        try:
            source_uuid = uuid.UUID(source_doc_id)
            claims = [c for c in claims if c.source_document_id == source_uuid]
        except ValueError:
            pass  # Invalid UUID — ignore filter

    # Return ONLY the items partial (not the full queue_list with filters) so
    # HTMX swap into #queue-list does not duplicate the filter UI inside the list.
    return templates.TemplateResponse(
        request=request,
        name="partials/_queue_items.html",
        context={
            "packet": context["packet"],
            "run": context["run"],
            "claims": claims,
            "active_filters": {
                "verdict_filter": verdict_filter,
                "confidence": confidence,
                "source_doc_id": source_doc_id,
            },
        },
    )


@router.get(
    "/dashboard/{packet_id}/{run_id}/claims/{claim_id}", response_class=HTMLResponse
)
async def claim_detail_view(
    request: Request,
    packet_id: uuid.UUID,
    run_id: uuid.UUID,
    claim_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Claim detail partial — full claim with verdicts and evidence spans."""
    result = await load_claim_detail(claim_id, run_id, db)
    if result is None:
        raise APIError(
            code="NOT_FOUND",
            message=f"Claim {claim_id} not found",
            status=404,
        )
    claim, context_snippet = result

    # Build source doc map for evidence badge display: {str(doc.id): {"filename", "is_draft"}}
    docs_result = await db.execute(
        select(SourceDocument).where(SourceDocument.packet_id == claim.packet_id)
    )
    source_doc_map = {
        str(doc.id): {"filename": doc.filename, "is_draft": doc.is_report}
        for doc in docs_result.scalars().all()
    }

    return templates.TemplateResponse(
        request=request,
        name="partials/claim_detail.html",
        context={
            "claim": claim,
            "context_snippet": context_snippet,
            "packet_id": packet_id,
            "run_id": run_id,
            "source_doc_map": source_doc_map,
        },
    )


@router.post(
    "/api/runs/{run_id}/claims/{claim_id}/review", response_class=HTMLResponse
)
async def submit_review(
    request: Request,
    run_id: uuid.UUID,
    claim_id: uuid.UUID,
    action: Annotated[str, Form()],
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Submit a review decision for a claim verdict."""
    try:
        await upsert_review_decision(claim_id, run_id, action, db)
    except ValueError as exc:
        raise APIError(
            code="INVALID_ACTION",
            message=str(exc),
            status=422,
        ) from exc
    except LookupError as exc:
        raise APIError(
            code="NOT_FOUND",
            message=str(exc),
            status=404,
        ) from exc

    return templates.TemplateResponse(
        request=request,
        name="partials/review_buttons.html",
        context={
            "claim_id": claim_id,
            "action": action,
            "run_id": run_id,
        },
    )


@router.delete("/api/packets/{packet_id}", response_class=HTMLResponse)
async def delete_packet(
    request: Request,
    packet_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Delete a packet and ALL its dependents (runs, claims, verdicts, decisions, sources, files).

    Per-FK explicit deletion in dependency order. SQLAlchemy declared-relationship
    cascades cover packet→source_doc and packet→run_version, but RunVersion has no
    declared relationship to Claim and the FK has no `ondelete=CASCADE`, so we must
    delete claims (and their cascading dependents) before the run rows.

    Authorization: the caller's session cookie must match packet.session_id.
    Demo packets (is_demo=True) are NEVER deletable by visitors — they're the
    canonical public example. Missing cookie + NULL session_id on a legacy row
    is rejected as "not yours" rather than ambiguously allowed.

    File cleanup runs BEFORE commit so a crash leaves DB + disk in sync —
    the row is still present and can be re-deleted.

    Returns 200 with empty body — caller's HTMX swap removes the dashboard row.
    """
    from sqlalchemy import delete as _delete, select as _select
    from evidenceengine.models.claim import Claim, CitationAnchor
    from evidenceengine.models.document import DocumentPacket
    from evidenceengine.models.review import ReviewDecision
    from evidenceengine.models.run import RunVersion
    from evidenceengine.models.verdict import Verdict, VerdictEvidence
    from evidenceengine.api.dependencies import get_file_store

    pkt = await db.get(DocumentPacket, packet_id)
    if pkt is None:
        raise APIError(code="NOT_FOUND", message=f"Packet {packet_id} not found", status=404)

    # Ownership: treat demo packets as undeletable, and require session match
    # for everything else. Legacy rows with NULL session_id are not owned by
    # anyone, so deletion is rejected — the correct path is a server-side
    # cleanup script with explicit authority.
    visitor_session = getattr(request.state, "session_id", "") or ""
    if pkt.is_demo:
        raise APIError(code="FORBIDDEN", message="Demo packets cannot be deleted", status=403)
    if pkt.session_id is None or pkt.session_id != visitor_session:
        raise APIError(code="FORBIDDEN", message="You don't own this packet", status=403)

    claim_ids = (await db.execute(
        _select(Claim.id).where(Claim.packet_id == packet_id)
    )).scalars().all()
    verdict_ids = (await db.execute(
        _select(Verdict.id).where(Verdict.claim_id.in_(claim_ids))
    )).scalars().all() if claim_ids else []
    run_ids = (await db.execute(
        _select(RunVersion.id).where(RunVersion.packet_id == packet_id)
    )).scalars().all()

    # Delete leaves first (DB)
    if verdict_ids:
        await db.execute(_delete(VerdictEvidence).where(VerdictEvidence.verdict_id.in_(verdict_ids)))
    if claim_ids:
        await db.execute(_delete(Verdict).where(Verdict.claim_id.in_(claim_ids)))
        await db.execute(_delete(CitationAnchor).where(CitationAnchor.claim_id.in_(claim_ids)))
        await db.execute(_delete(ReviewDecision).where(ReviewDecision.claim_id.in_(claim_ids)))
        await db.execute(_delete(Claim).where(Claim.packet_id == packet_id))
    if run_ids:
        await db.execute(_delete(RunVersion).where(RunVersion.packet_id == packet_id))
    # SourceDocument + DocumentPacket cascade declared via relationship
    await db.delete(pkt)

    # Clean up files BEFORE the DB commit — if file removal fails, the
    # transaction rolls back and the caller sees a 500 rather than the
    # silent disk leak we'd have otherwise. The file_store delete is
    # idempotent so a retry cleans up anything that got partially removed.
    file_store = get_file_store()
    try:
        file_store.delete_packet(str(packet_id))
    except Exception as exc:  # noqa: BLE001 — log + rollback
        await db.rollback()
        raise APIError(
            code="FILE_DELETE_FAILED",
            message=f"Could not remove packet files ({type(exc).__name__}); DB not modified",
            status=500,
        ) from exc

    await db.commit()

    # Return empty body — HTMX hx-swap=outerHTML removes the row
    return HTMLResponse("", status_code=200)
