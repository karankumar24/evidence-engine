"""Log recall@k metrics to RunVersion.pipeline_config JSONB after a retrieval run.

Recall@k denominator is total_claim_count (includes unresolvable claims).
Unresolvable claims count as "not retrieved" — they inflate the denominator
but not the numerator, giving an honest recall figure for the run.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.models.run import RunVersion

logger = logging.getLogger(__name__)


async def log_retrieval_metrics(
    run_version_id: str,
    claim_count: int,
    retrieved_count: int,
    k: int,
    db: AsyncSession,
) -> None:
    """Write recall@k metrics to RunVersion.pipeline_config["retrieval_metrics"].

    Args:
        run_version_id: UUID str of the RunVersion to update.
        claim_count: Total claims in the run (including unresolvable — honest denominator).
        retrieved_count: Claims that had at least one EvidenceSpan persisted.
        k: The final top-k value used (settings.retrieval_top_k_final).
        db: Active AsyncSession.
    """
    result = await db.execute(
        select(RunVersion).where(RunVersion.id == run_version_id)
    )
    run = result.scalar_one()
    config = run.pipeline_config or {}
    config["retrieval_metrics"] = {
        "recall_at_k": retrieved_count / claim_count if claim_count > 0 else 0.0,
        "claim_count": claim_count,
        "retrieved_count": retrieved_count,
        "k": k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    run.pipeline_config = config
    await db.flush()
    logger.info(
        "Retrieval metrics logged: recall@%d=%.3f (%d/%d claims)",
        k,
        config["retrieval_metrics"]["recall_at_k"],
        retrieved_count,
        claim_count,
    )
