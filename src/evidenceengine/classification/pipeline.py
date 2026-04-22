"""Verdict classification pipeline orchestrator.

Loads EvidenceSpan rows grouped by Claim, dispatches verdict classification
through :func:`evidenceengine.classification.backend.get_backend` (NLI-primary
or LLM-primary per :data:`settings.classifier_backend`), applies a tiebreaker
for low-confidence NLI verdicts, fires a post-verdict explanation call, and
persists Verdict + VerdictEvidence rows.

Handles edge cases without LLM calls:
- Zero EvidenceSpans → insufficient_support
- Claim status=unresolvable_anchor → needs_review

Phase 02 Plan 03 wiring (NLI-primary by default):
- Per-claim dispatch goes through ``backend.classify`` (not ``classify_claim``).
- When ``classifier_backend == "nli_primary"`` AND the primary verdict's
  ``confidence_score < settings.nli_tiebreaker_threshold`` (0.65), an LLM
  tiebreaker fires via ``LLMClassifier``. ``llm_tiebreaker_fired`` ticks on
  every ATTEMPT. ``RuntimeError`` (chain exhausted / quota) fails-open to
  ``needs_review`` with the NLI confidence — never raises to the caller.
- A post-verdict explanation call runs on BOTH backends; failure logs a
  warning and never blocks persistence.
- When ``classifier_backend == "nli_primary"``:
    * the legacy ``nli_second_opinion`` post-step is SKIPPED (Pitfall 7
      — avoid double NLI).
    * the legacy ``apply_confidence_threshold(0.70)`` override is SKIPPED
      (NLI-primary owns its band: 0.50 < 0.65 < 0.80).
- Telemetry gains ``classifier_backend``, ``nli_verdict_counts``,
  ``llm_tiebreaker_fired``, ``explanation_generated``, ``explanation_failed``
  — additive, no existing key renamed or removed.
"""

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from evidenceengine.classification.backend import get_backend
from evidenceengine.classification.classifier import apply_confidence_threshold, classify_claim  # noqa: F401
from evidenceengine.classification.explanation import generate_explanation
from evidenceengine.classification.schemas import VerdictClassificationResponse
from evidenceengine.core.config import settings
from evidenceengine.models.claim import Claim
from evidenceengine.models.verdict import Verdict, VerdictEvidence

logger = logging.getLogger(__name__)


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
    # evidence spans, then resolve to {id: filename} in one query.
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

    # Resolve backend ONCE per run. get_backend() reads settings and dispatches
    # to NLIClassifier or LLMClassifier — Plan 01 foundation / Plan 02 brain.
    backend = get_backend()
    is_nli_primary = settings.classifier_backend == "nli_primary"

    all_verdicts: list[Verdict] = []
    claim_errors_accumulator: list[dict] = []
    # Telemetry accumulators — surfaced in pipeline_config["classification_telemetry"] below.
    self_verify_claims = 0
    self_verify_caps_applied = 0
    threshold_overrides_to_review = 0
    chain_exhausted_count = 0
    model_refusal_count = 0
    bypass_unresolvable_anchor = 0
    nli_second_opinion_overrides = 0
    # v1.2.9 Phase 02 Plan 03 — NLI-primary + tiebreaker + explanation counters.
    llm_tiebreaker_fired = 0
    explanation_generated = 0
    explanation_failed = 0
    nli_verdict_counts: dict[str, int] = {
        "supported": 0, "contradicted": 0,
        "insufficient_support": 0, "needs_review": 0,
    }

    for claim in claims:
        # Idempotency: skip if verdict already exists for this claim+run.
        # The EXPLANATION + TIEBREAKER calls also live inside this guard so
        # reruns never re-spend LLM budget.
        existing_result = await db.execute(
            select(Verdict).where(
                Verdict.claim_id == claim.id,
                Verdict.run_version_id == run_version_uuid,
            )
        )
        if existing_result.scalar_one_or_none() is not None:
            continue

        # Edge case: claim whose citations all failed to resolve — bypass LLM.
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
                continue
            all_verdicts.append(verdict)
            continue

        # Normal case: dispatch via the resolved backend (NLI or LLM).
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

        # Detect self-verify mode.
        self_verify_mode = not any(
            a.resolution_status == "resolved" and a.target_document_id is not None
            for a in (claim.citation_anchors or [])
        )
        if self_verify_mode:
            self_verify_claims += 1

        # Per-claim try/except — a single failure never kills the run.
        try:
            classification = await backend.classify(claim.claim_text, evidence_span_dicts)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "exhausted" in msg.lower() or "chain" in msg.lower():
                chain_exhausted_count += 1
            claim_errors_accumulator.append(
                {"claim_id": str(claim.id), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        # NLI-primary tiebreaker path (CLF-05). Fires when NLI is not confident
        # enough; increments on ATTEMPT; fails-open to needs_review on quota.
        if is_nli_primary and classification.confidence_score < settings.nli_tiebreaker_threshold:
            llm_tiebreaker_fired += 1
            try:
                # Reuse the LLMClassifier verdict path — same prompt, same schema.
                from evidenceengine.classification.llm_classifier import LLMClassifier  # noqa: PLC0415
                tiebreaker_backend = LLMClassifier()
                classification = await tiebreaker_backend.classify(
                    claim.claim_text, evidence_span_dicts,
                )
            except RuntimeError as exc:
                logger.warning(
                    "NLI tiebreaker failed-open to needs_review: %s", exc,
                )
                classification = VerdictClassificationResponse(
                    reasoning=(
                        f"[NLI low-conf {classification.confidence_score:.2f}, "
                        f"tiebreaker quota exhausted] — needs human review. "
                        f"Original NLI reasoning: {classification.reasoning}"
                    ),
                    verdict_type="needs_review",
                    confidence_score=classification.confidence_score,
                )

        # Track which verdict-type the chosen backend (post-tiebreaker) produced.
        nli_verdict_counts[classification.verdict_type] = (
            nli_verdict_counts.get(classification.verdict_type, 0) + 1
        )

        # Model refusal — classify_claim returns needs_review @ 0.0.
        if (
            classification.verdict_type == "needs_review"
            and classification.confidence_score == 0.0
            and "refused" in (classification.reasoning or "").lower()
        ):
            model_refusal_count += 1

        # Self-verify trust guard (VERDICT-02) — unchanged from pre-Plan-03.
        cap = settings.self_verify_supported_cap
        if (
            self_verify_mode
            and classification.verdict_type == "supported"
            and classification.confidence_score > cap
        ):
            classification = VerdictClassificationResponse(
                verdict_type="supported",
                confidence_score=cap,
                reasoning=f"[Self-verify cap {cap:.2f}] {classification.reasoning}",
            )
            self_verify_caps_applied += 1

        # Legacy NLI second-opinion: GATED OFF for nli_primary (Pitfall 7 —
        # avoid double NLI; the NLI classifier already IS the primary here).
        # llm_primary still gets the second-opinion downgrader.
        if not is_nli_primary and settings.nli_second_opinion_enabled:
            from evidenceengine.classification.nli_second_opinion import (  # noqa: PLC0415
                nli_judgment,
                should_force_review,
            )
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

        # Legacy 0.70 confidence override — GATED OFF for nli_primary (Open Q
        # #4: NLI-primary owns its band 0.50 < 0.65 < 0.80; re-applying 0.70
        # here would re-override tiebreaker-passed 0.68 verdicts).
        if not is_nli_primary:
            pre_threshold_type = classification.verdict_type
            classification = apply_confidence_threshold(
                classification, settings.verdict_needs_review_threshold,
            )
            if (
                pre_threshold_type != "needs_review"
                and classification.verdict_type == "needs_review"
            ):
                threshold_overrides_to_review += 1

        # Always-on post-verdict explanation (BOTH backends). Non-blocking.
        explanation_text = await generate_explanation(
            claim.claim_text, evidence_span_dicts, classification,
        )
        if explanation_text is not None:
            explanation_generated += 1
            classification = classification.model_copy(
                update={"explanation": explanation_text},
            )
        else:
            explanation_failed += 1

        # Concatenate explanation into reasoning for DB audit trail (Open Q
        # #3). No schema migration — the raw `explanation` also lives on the
        # response but Verdict.reasoning is the canonical persisted string.
        reasoning_for_db = classification.reasoning
        if classification.explanation:
            reasoning_for_db = (
                f"{classification.reasoning}\n\n"
                f"[explanation: {classification.explanation}]"
            )

        verdict = Verdict(
            claim_id=claim.id,
            run_version_id=run_version_uuid,
            verdict_type=classification.verdict_type,
            confidence_score=classification.confidence_score,
            reasoning=reasoning_for_db,
            model_name=settings.classification_model,
            prompt_version=settings.verdict_prompt_version,
        )
        try:
            async with db.begin_nested():
                db.add(verdict)
                await db.flush()
        except IntegrityError:
            continue

        for span in claim.evidence_spans:
            db.add(
                VerdictEvidence(
                    verdict_id=verdict.id,
                    evidence_span_id=span.id,
                    weight=span.relevance_score,
                )
            )

        all_verdicts.append(verdict)

    # Surface per-claim classification errors (PIPE-05) + trust-model telemetry.
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
            # v1.2.9 Phase 02 Plan 03 additions (additive — no existing keys renamed/removed).
            "classifier_backend": settings.classifier_backend,
            "nli_verdict_counts": nli_verdict_counts,
            "llm_tiebreaker_fired": llm_tiebreaker_fired,
            "explanation_generated": explanation_generated,
            "explanation_failed": explanation_failed,
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
