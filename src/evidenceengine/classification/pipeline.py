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

    # Collect every source_document_id referenced by the claims and their
    # evidence spans, then resolve to {id: filename} in one query. The
    # classifier uses these filenames + a "same doc vs external" flag to
    # distinguish self-verify mode deterministically (VERDICT-02 layer 2).
    from evidenceengine.models.document import SourceDocument  # noqa: PLC0415
    source_doc_ids: set[uuid.UUID] = set()
    for c in claims:
        if c.source_document_id is not None:
            source_doc_ids.add(c.source_document_id)
        for s in c.evidence_spans:
            if s.source_document_id is not None:
                source_doc_ids.add(s.source_document_id)
    doc_filename_by_id: dict[uuid.UUID, str] = {}
    if source_doc_ids:
        doc_rows = (await db.execute(
            select(SourceDocument.id, SourceDocument.filename)
            .where(SourceDocument.id.in_(source_doc_ids))
        )).all()
        doc_filename_by_id = {r[0]: r[1] for r in doc_rows}

    all_verdicts: list[Verdict] = []
    claim_errors_accumulator: list[dict] = []
    # Telemetry accumulators — surfaced in pipeline_config["telemetry"] below.
    # These are ADDITIVE observability, not control-flow — removing them never
    # changes verdicts. The point is making silent-failure modes visible.
    self_verify_claims = 0                # claims whose anchors all unresolved
    self_verify_caps_applied = 0          # SUPPORTED verdicts hard-capped by cap
    threshold_overrides_to_review = 0     # below-threshold -> needs_review
    chain_exhausted_count = 0             # fallback chain gave up entirely
    model_refusal_count = 0               # model returned refusal
    bypass_unresolvable_anchor = 0        # short-circuited without LLM call
    nli_second_opinion_overrides = 0      # NLI forced needs_review (A- : new trust layer)

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

        # Edge case: claim whose citations all failed to resolve — bypass LLM
        # and route to needs_review per module invariant. Even if retrieval
        # produced self-verify spans, "we couldn't match your citations" is
        # the stronger signal and warrants human review.
        if claim.status == "unresolvable_anchor":
            default_verdict_type = "needs_review"
            default_reasoning = "All citation anchors unresolvable — cited source documents could not be matched."
            bypass_unresolvable_anchor += 1
        elif not claim.evidence_spans:
            default_verdict_type = "insufficient_support"
            default_reasoning = "No evidence spans retrieved for this claim."
        else:
            default_verdict_type = None

        if default_verdict_type is not None:
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

        # Normal case: classify with LLM. Pass per-span source metadata so
        # the classifier prompt can tag each span with its source filename
        # and a SAME/EXTERNAL flag — makes self-verify reasoning structural
        # (the model sees the tag directly) instead of inferential.
        evidence_span_dicts = [
            {
                "span_text": span.span_text,
                "relevance_score": span.relevance_score or 0.0,
                "rank": span.retrieval_rank or 0,
                "source_filename": doc_filename_by_id.get(span.source_document_id),
                "is_same_doc_as_claim": (
                    span.source_document_id is not None
                    and span.source_document_id == claim.source_document_id
                ),
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
        if self_verify_mode:
            self_verify_claims += 1

        # Per-claim try/except so a single LLM failure (chain exhausted,
        # structured-output violation, etc.) doesn't kill the whole run.
        # Records per-claim errors for the orchestrator to surface as
        # pipeline_config["claim_errors"] (PIPE-05).
        try:
            classification = await classify_claim(claim.claim_text, evidence_span_dicts)
        except Exception as exc:
            msg = str(exc)
            if "exhausted" in msg.lower() or "chain" in msg.lower():
                chain_exhausted_count += 1
            claim_errors_accumulator.append(
                {"claim_id": str(claim.id), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        # Model refusal — classify_claim returns needs_review @ 0.0.
        if (
            classification.verdict_type == "needs_review"
            and classification.confidence_score == 0.0
            and "refused" in (classification.reasoning or "").lower()
        ):
            model_refusal_count += 1

        # Self-verify trust guard (VERDICT-02): cap SUPPORTED confidence so
        # weak self-referential evidence routes to needs_review via the
        # existing threshold. The cap AND threshold are co-designed — see
        # core/config.py for the band invariant.
        cap = settings.self_verify_supported_cap
        if (
            self_verify_mode
            and classification.verdict_type == "supported"
            and classification.confidence_score > cap
        ):
            from evidenceengine.classification.schemas import VerdictClassificationResponse
            classification = VerdictClassificationResponse(
                verdict_type="supported",
                confidence_score=cap,
                reasoning=f"[Self-verify cap {cap:.2f}] {classification.reasoning}",
            )
            self_verify_caps_applied += 1

        # NLI second-opinion: catches high-confidence contradictions the LLM
        # missed (downgrade SUPPORTED → needs_review) and vice versa for
        # contradicted verdicts. Never upgrades; only downgrades to needs_review.
        if settings.nli_second_opinion_enabled:
            from evidenceengine.classification.nli_second_opinion import (  # noqa: PLC0415
                nli_judgment,
                should_force_review,
            )
            from evidenceengine.classification.schemas import VerdictClassificationResponse  # noqa: PLC0415
            import asyncio as _asyncio  # noqa: PLC0415
            nli = await _asyncio.to_thread(
                nli_judgment, claim.claim_text, [s["span_text"] for s in evidence_span_dicts],
            )
            force, suffix = should_force_review(classification.verdict_type, nli)
            if force:
                classification = VerdictClassificationResponse(
                    verdict_type="needs_review",
                    confidence_score=min(classification.confidence_score, 0.50),
                    reasoning=f"{suffix} {classification.reasoning}",
                )
                nli_second_opinion_overrides += 1

        pre_threshold_type = classification.verdict_type
        classification = apply_confidence_threshold(
            classification, settings.verdict_needs_review_threshold
        )
        if (
            pre_threshold_type != "needs_review"
            and classification.verdict_type == "needs_review"
        ):
            threshold_overrides_to_review += 1

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

    # Surface per-claim classification errors (PIPE-05) + trust-model
    # telemetry to the run. The dashboard can show "X of Y classified — N
    # failed"; operators can monitor self_verify_rate and chain_exhausted_rate
    # to detect pipeline regressions without reading logs.
    from evidenceengine.models.run import RunVersion  # lazy to avoid cycles
    run = await db.get(RunVersion, run_version_uuid)
    if run is not None:
        telemetry = {
            "claims_total": len(claims),
            "verdicts_produced": len(all_verdicts),
            "self_verify_claims": self_verify_claims,
            "self_verify_rate": round(self_verify_claims / len(claims), 3) if claims else 0.0,
            "self_verify_caps_applied": self_verify_caps_applied,
            "threshold_overrides_to_review": threshold_overrides_to_review,
            "chain_exhausted_count": chain_exhausted_count,
            "model_refusal_count": model_refusal_count,
            "bypass_unresolvable_anchor": bypass_unresolvable_anchor,
            "nli_second_opinion_overrides": nli_second_opinion_overrides,
        }
        merged_config: dict = {
            **(run.pipeline_config or {}),
            "classification_telemetry": telemetry,
        }
        if claim_errors_accumulator:
            existing = (run.pipeline_config or {}).get("claim_errors", [])
            merged_config["claim_errors"] = existing + claim_errors_accumulator
            merged_config["failed_claim_count"] = len(existing) + len(claim_errors_accumulator)
        run.pipeline_config = merged_config

    await db.commit()
    return all_verdicts
