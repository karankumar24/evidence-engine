"""On-demand claim explanation endpoint.

POST /api/claims/{claim_id}/explain

Generates a 2-3 sentence explanation of a claim's verdict when the user
clicks "Explain this verdict". Returns an HTML fragment for HTMX to inject.

Requires an LLM API key configured via LLM_API_KEY env var. Returns empty
HTML fragment if no key is configured or the LLM call fails — never blocks
the dashboard from loading.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.api.dependencies import get_db
from evidenceengine.classification.explanation import generate_explanation
from evidenceengine.models.claim import Claim
from evidenceengine.models.evidence import EvidenceSpan
from evidenceengine.models.verdict import Verdict

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/claims/{claim_id}/explain", response_class=HTMLResponse)
async def explain_claim_verdict(
    claim_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    """Generate a 2-3 sentence explanation for a claim's verdict.

    Returns an HTML fragment — HTMX injects this directly into the explanation div.
    Returns empty fragment on any failure (missing API key, LLM error, missing data).
    """
    # Load claim
    claim_result = await db.execute(select(Claim).where(Claim.id == claim_id))
    claim = claim_result.scalar_one_or_none()
    if claim is None:
        return HTMLResponse("")

    # Load verdict (first available for this claim)
    verdict_result = await db.execute(
        select(Verdict).where(Verdict.claim_id == claim_id).limit(1)
    )
    verdict = verdict_result.scalar_one_or_none()
    if verdict is None:
        return HTMLResponse("")

    # Load top evidence spans
    spans_result = await db.execute(
        select(EvidenceSpan)
        .where(EvidenceSpan.claim_id == claim_id)
        .order_by(EvidenceSpan.retrieval_rank)
        .limit(3)
    )
    spans = spans_result.scalars().all()
    span_dicts = [{"text": s.span_text, "relevance_score": s.relevance_score} for s in spans]

    # Generate explanation — returns None if no API key or LLM fails
    try:
        explanation = await generate_explanation(
            claim_text=claim.claim_text,
            evidence_spans=span_dicts,
            verdict=verdict,  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Explanation generation failed for claim %s: %s", claim_id, exc)
        explanation = None

    if not explanation:
        return HTMLResponse(
            '<span class="text-ink-400 text-xs italic">'
            "Explanation unavailable — configure LLM_API_KEY to enable."
            "</span>"
        )

    return HTMLResponse(
        f'<p class="text-xs font-serif italic text-ink-600 leading-relaxed">'
        f"{explanation}"
        f"</p>"
    )
