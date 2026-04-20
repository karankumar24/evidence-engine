"""Dashboard data-access service — queries for the reviewer dashboard."""

import uuid
from collections import Counter

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from evidenceengine.models.claim import Claim
from evidenceengine.models.document import DocumentPacket, SourceDocument
from evidenceengine.models.review import ReviewDecision
from evidenceengine.models.run import RunVersion
from evidenceengine.models.verdict import Verdict, VerdictEvidence

# Severity ordering for queue sort (lower = shown first)
_SEVERITY: dict[str, int] = {
    "contradicted": 0,
    "needs_review": 1,
    "insufficient_support": 2,
    "supported": 3,
}

_VERDICT_TYPES = ["contradicted", "needs_review", "insufficient_support", "supported"]
_VALID_ACTIONS = {"approve", "reject", "mark_insufficient"}


def compute_verdict_distribution(verdicts: list) -> dict:
    """Compute counts and percentages per verdict type.

    Args:
        verdicts: List of Verdict ORM instances (or any objects with .verdict_type).

    Returns:
        Dict with all four verdict types as keys:
            {"contradicted": {"count": N, "pct": X.X}, ...}
        Empty verdicts list returns all keys with count=0, pct=0.0.
    """
    counts = Counter(v.verdict_type for v in verdicts)
    total = len(verdicts)

    result: dict = {}
    for vtype in _VERDICT_TYPES:
        count = counts.get(vtype, 0)
        pct = round((count / total * 100), 1) if total > 0 else 0.0
        result[vtype] = {"count": count, "pct": pct}
    return result


async def load_dashboard_context(
    packet_id: uuid.UUID,
    run_id: uuid.UUID,
    db: AsyncSession,
) -> dict | None:
    """Load all data needed to render the dashboard overview.

    Returns dict with keys: packet, run, claims (sorted by severity), distribution,
    source_docs. Returns None if packet or run not found.
    """
    # Load packet with source documents via select+options (db.get() with options
    # does not reliably populate relationships in async context)
    packet_result = await db.execute(
        select(DocumentPacket)
        .where(DocumentPacket.id == packet_id)
        .options(selectinload(DocumentPacket.source_documents))
    )
    packet = packet_result.scalar_one_or_none()
    if packet is None:
        return None

    # Load run
    run = await db.get(RunVersion, run_id)
    if run is None:
        return None

    # Load all claims for this run with verdicts and review_decisions
    result = await db.execute(
        select(Claim)
        .where(Claim.run_version_id == run_id)
        .options(
            selectinload(Claim.verdicts),
            selectinload(Claim.review_decisions),
        )
    )
    claims = list(result.scalars().all())

    # Sort claims by severity (based on first/worst verdict)
    def claim_severity(claim: Claim) -> int:
        if not claim.verdicts:
            return 4
        # Use the worst (lowest severity number) verdict type
        return min(_SEVERITY.get(v.verdict_type, 4) for v in claim.verdicts)

    claims.sort(key=claim_severity)

    # Collect all verdicts for distribution
    all_verdicts = [v for c in claims for v in c.verdicts]
    distribution = compute_verdict_distribution(all_verdicts)

    return {
        "packet": packet,
        "run": run,
        "claims": claims,
        "distribution": distribution,
        "source_docs": packet.source_documents,
    }


async def load_claim_detail(
    claim_id: uuid.UUID,
    run_id: uuid.UUID,
    db: AsyncSession,
) -> tuple[Claim, str] | None:
    """Load a single claim with full eager-loaded relations for the detail panel.

    Returns (claim, context_snippet) or None if not found.
    context_snippet is raw_text[max(0,char_start-200):char_end+200] from the
    report SourceDocument in the packet.
    """
    result = await db.execute(
        select(Claim)
        .where(Claim.id == claim_id)
        .options(
            selectinload(Claim.verdicts).selectinload(Verdict.verdict_evidence).selectinload(
                VerdictEvidence.evidence_span
            ),
            selectinload(Claim.review_decisions),
        )
    )
    claim = result.scalar_one_or_none()
    if claim is None:
        return None

    # Retrieve context_snippet from report SourceDocument
    src_result = await db.execute(
        select(SourceDocument).where(
            SourceDocument.packet_id == claim.packet_id,
            SourceDocument.is_report.is_(True),
        )
    )
    report_doc = src_result.scalar_one_or_none()

    context_snippet = ""
    if report_doc and report_doc.raw_text:
        start = max(0, claim.char_start - 200)
        end = claim.char_end + 200
        context_snippet = report_doc.raw_text[start:end]

    return claim, context_snippet


async def upsert_review_decision(
    claim_id: uuid.UUID,
    run_id: uuid.UUID,
    action: str,
    db: AsyncSession,
) -> ReviewDecision:
    """Delete existing ReviewDecision(s) for claim_id and insert a new one.

    Args:
        claim_id: UUID of the claim being reviewed.
        run_id: UUID of the run version (used to look up the verdict).
        action: One of "approve", "reject", "mark_insufficient".
        db: Async SQLAlchemy session.

    Returns:
        The newly created ReviewDecision.

    Raises:
        ValueError: If action is not a valid value.
        LookupError: If no Verdict exists for claim_id + run_id.
    """
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"Invalid action '{action}'. Must be one of: {', '.join(sorted(_VALID_ACTIONS))}"
        )

    # Find the verdict for this claim + run
    verdict_result = await db.execute(
        select(Verdict).where(
            Verdict.claim_id == claim_id,
            Verdict.run_version_id == run_id,
        )
    )
    verdict = verdict_result.scalar_one_or_none()
    if verdict is None:
        raise LookupError(f"No verdict found for claim_id={claim_id}, run_id={run_id}")

    # Delete any existing review decisions for this claim
    await db.execute(
        delete(ReviewDecision).where(ReviewDecision.claim_id == claim_id)
    )

    # Insert new decision
    decision = ReviewDecision(
        claim_id=claim_id,
        verdict_id=verdict.id,
        reviewer_id="reviewer-stub",
        action=action,
        verdict_at_decision=verdict.verdict_type,
    )
    db.add(decision)
    await db.commit()
    await db.refresh(decision)
    return decision


async def load_runs_index(db: AsyncSession, limit: int = 50) -> list[RunVersion]:
    """Load LATEST run per packet for the dashboard index page (deduplicated).

    Without this dedup, the dashboard accumulates rows across every re-upload of
    the same packet, hiding the user's most recent verdict result among historical
    failures. Uses a window function to pick row_number=1 per partition_by(packet_id),
    ordered by created_at DESC. Falls back gracefully if the DB has no runs.
    """
    rn = func.row_number().over(
        partition_by=RunVersion.packet_id,
        order_by=RunVersion.created_at.desc(),
    ).label("rn")
    latest_subq = select(RunVersion.id, rn).subquery()
    result = await db.execute(
        select(RunVersion)
        .join(latest_subq, RunVersion.id == latest_subq.c.id)
        .where(latest_subq.c.rn == 1)
        .options(selectinload(RunVersion.packet))
        .order_by(RunVersion.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
