"""Verdict classification pipeline orchestrator.

Loads EvidenceSpan rows grouped by Claim, calls classify_claim() per claim,
applies confidence threshold routing, and persists Verdict + VerdictEvidence rows.

Handles edge cases without LLM calls:
- Zero EvidenceSpans → insufficient_support
- Claim status=unresolvable_anchor → needs_review
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from evidenceengine.classification.classifier import apply_confidence_threshold, classify_claim
from evidenceengine.core.config import settings
from evidenceengine.models.claim import Claim
from evidenceengine.models.verdict import Verdict, VerdictEvidence


async def classify_verdicts_for_run(
    run_version_id: str,
    db: AsyncSession,
) -> list[Verdict]:
    """Classify all claims in a run and persist Verdict + VerdictEvidence rows.

    Idempotent: existing verdicts are skipped (no UniqueConstraint error on rerun).

    Args:
        run_version_id: UUID string of the RunVersion to classify.
        db: Async SQLAlchemy session. Caller manages transaction lifecycle.

    Returns:
        List of newly created Verdict ORM instances (excludes pre-existing verdicts).
    """
    run_version_uuid = uuid.UUID(run_version_id)

    # 1. Load all Claims for this run with their evidence_spans eagerly (no N+1)
    result = await db.execute(
        select(Claim)
        .where(Claim.run_version_id == run_version_uuid)
        .options(selectinload(Claim.evidence_spans))
    )
    claims = result.scalars().all()

    if not claims:
        return []

    all_verdicts: list[Verdict] = []

    for claim in claims:
        # Idempotency: skip if verdict already exists for this claim+run
        existing_result = await db.execute(
            select(Verdict).where(
                Verdict.claim_id == claim.id,
                Verdict.run_version_id == run_version_uuid,
            )
        )
        if existing_result.scalar_one_or_none() is not None:
            continue

        # Edge case: unresolvable anchor — no LLM call
        if claim.status == "unresolvable_anchor":
            verdict = Verdict(
                claim_id=claim.id,
                run_version_id=run_version_uuid,
                verdict_type="needs_review",
                confidence_score=0.0,
                reasoning="All citation anchors unresolvable — cited source documents could not be matched.",
                model_name="none",
                prompt_version=None,
            )
            db.add(verdict)
            await db.flush()
            all_verdicts.append(verdict)
            continue

        # Edge case: zero evidence spans — no LLM call (VERDICT-02)
        if not claim.evidence_spans:
            verdict = Verdict(
                claim_id=claim.id,
                run_version_id=run_version_uuid,
                verdict_type="insufficient_support",
                confidence_score=0.0,
                reasoning="No evidence spans retrieved for this claim.",
                model_name="none",
                prompt_version=None,
            )
            db.add(verdict)
            await db.flush()
            all_verdicts.append(verdict)
            continue

        # Normal case: classify with LLM
        evidence_span_dicts = [
            {
                "span_text": span.span_text,
                "relevance_score": span.relevance_score or 0.0,
                "rank": span.retrieval_rank or 0,
            }
            for span in claim.evidence_spans
        ]

        classification = await classify_claim(claim.claim_text, evidence_span_dicts)
        classification = apply_confidence_threshold(
            classification, settings.verdict_needs_review_threshold
        )

        verdict = Verdict(
            claim_id=claim.id,
            run_version_id=run_version_uuid,
            verdict_type=classification.verdict_type,
            confidence_score=classification.confidence_score,
            reasoning=classification.reasoning,
            model_name=settings.classification_model,
            prompt_version=settings.verdict_prompt_version,
        )
        db.add(verdict)
        await db.flush()  # get verdict.id before linking evidence spans

        for span in claim.evidence_spans:
            db.add(
                VerdictEvidence(
                    verdict_id=verdict.id,
                    evidence_span_id=span.id,
                    weight=span.relevance_score,
                )
            )

        all_verdicts.append(verdict)

    await db.commit()
    return all_verdicts
