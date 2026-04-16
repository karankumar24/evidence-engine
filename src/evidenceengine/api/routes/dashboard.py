"""Dashboard routes — GET /dashboard/{packet_id}/{run_id} and partial endpoints."""

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.api.dependencies import get_db
from evidenceengine.api.services.dashboard_queries import (
    load_claim_detail,
    load_dashboard_context,
    load_runs_index,
    upsert_review_decision,
)
from evidenceengine.schemas.common import APIError

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_index(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Index page — lists recent pipeline runs for the reviewer to select."""
    runs = await load_runs_index(db)
    return templates.TemplateResponse(
        request=request,
        name="dashboard_index.html",
        context={"runs": runs},
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

    return templates.TemplateResponse(
        request=request,
        name="partials/queue_list.html",
        context={
            "packet": context["packet"],
            "run": context["run"],
            "claims": claims,
            "source_docs": context["source_docs"],
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
    return templates.TemplateResponse(
        request=request,
        name="partials/claim_detail.html",
        context={
            "claim": claim,
            "context_snippet": context_snippet,
            "packet_id": packet_id,
            "run_id": run_id,
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
