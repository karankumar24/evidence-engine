"""Verdict classification pipeline orchestrator.

Loads EvidenceSpan rows grouped by Claim, calls classify_claim() per claim,
applies confidence threshold routing, and persists Verdict + VerdictEvidence rows.

Handles edge cases without LLM calls:
- Zero EvidenceSpans → insufficient_support
- Claim status=unresolvable_anchor → needs_review
"""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
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

    # 1. Load all Claims for this run with evidence_spans + citation_anchors
    # eagerly (no N+1). citation_anchors is needed for self-verify detection.
    result = await db.execute(
        select(Claim)
        .where(Claim.run_version_id == run_version_uuid)
        .options(
            selectinload(Claim.evidence_spans),
            selectinload(Claim.citation_anchors),
        )
    )
    claims = result.scalars().all()

    if not claims:
        return []

    all_verdicts: list[Verdict] = []
    claim_errors_accumulator: list[dict] = []

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

        # Edge case: zero evidence spans — no LLM call (VERDICT-02).
        # Check evidence first: self-verification may produce spans even when
        # citation anchors were unresolvable, and we want those to reach the LLM.
        if not claim.evidence_spans:
            if claim.status == "unresolvable_anchor":
                default_verdict_type = "needs_review"
                default_reasoning = "All citation anchors unresolvable — cited source documents could not be matched."
            else:
                default_verdict_type = "insufficient_support"
                default_reasoning = "No evidence spans retrieved for this claim."
            verdict = Verdict(
                claim_id=claim.id,
                run_version_id=run_version_uuid,
                verdict_type=default_verdict_type,
                confidence_score=0.0,
                reasoning=default_reasoning,
                model_name="none",
                prompt_version=None,
            )
            try:
                async with db.begin_nested():
                    db.add(verdict)
                    await db.flush()
            except IntegrityError:
                continue  # concurrent insert won the race — skip
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

        # Detect self-verify mode: evidence is from the report itself when
        # the claim has no resolved citation anchors AND its evidence spans
        # all came from the same doc as the claim.
        self_verify_mode = not any(
            a.resolution_status == "resolved" and a.target_document_id is not None
            for a in (claim.citation_anchors or [])
        )

        # Per-claim try/except so a single LLM failure (chain exhausted,
        # structured-output violation, etc.) doesn't kill the whole run.
        # Records per-claim errors for the orchestrator to surface as
        # pipeline_config["claim_errors"] (PIPE-05).
        try:
            classification = await classify_claim(claim.claim_text, evidence_span_dicts)
        except Exception as exc:
            claim_errors_accumulator.append(
                {"claim_id": str(claim.id), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        # Self-verify trust guard (VERDICT-02): cap SUPPORTED confidence so
        # weak self-referential evidence routes to needs_review via the
        # existing threshold (default 0.7).
        if (
            self_verify_mode
            and classification.verdict_type == "supported"
            and classification.confidence_score > 0.80
        ):
            from evidenceengine.classification.schemas import VerdictClassificationResponse
            classification = VerdictClassificationResponse(
                verdict_type="supported",
                confidence_score=0.80,
                reasoning=f"[Self-verify cap 0.80] {classification.reasoning}",
            )

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
        try:
            async with db.begin_nested():
                db.add(verdict)
                await db.flush()  # get verdict.id before linking evidence spans
        except IntegrityError:
            continue  # concurrent insert won the race — skip

        for span in claim.evidence_spans:
            db.add(
                VerdictEvidence(
                    verdict_id=verdict.id,
                    evidence_span_id=span.id,
                    weight=span.relevance_score,
                )
            )

        all_verdicts.append(verdict)

    # Surface per-claim classification errors (PIPE-05) to the run so the
    # dashboard can show "X of Y classified — N failed".
    if claim_errors_accumulator:
        from evidenceengine.models.run import RunVersion  # lazy to avoid cycles
        run = await db.get(RunVersion, run_version_uuid)
        if run is not None:
            existing = (run.pipeline_config or {}).get("claim_errors", [])
            run.pipeline_config = {
                **(run.pipeline_config or {}),
                "claim_errors": existing + claim_errors_accumulator,
                "failed_claim_count": len(existing) + len(claim_errors_accumulator),
            }

    await db.commit()
    return all_verdicts
