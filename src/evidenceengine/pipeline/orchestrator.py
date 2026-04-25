"""Pipeline orchestrator — async background task that chains extract → retrieve → classify.

IMPORTANT: run_full_pipeline() opens its own DB session via async_session_factory.
NEVER call it with a request-scope session. The RunVersion row MUST be committed
before this task is scheduled (BackgroundTasks or asyncio.create_task).
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.classification.pipeline import classify_verdicts_for_run
from evidenceengine.core.config import settings
from evidenceengine.core.database import async_session_factory
from evidenceengine.extraction.pipeline import extract_claims_for_document
from evidenceengine.ingestion.validation import (
    serialize_parsed_document,
    validate_and_parse_file,
)
from evidenceengine.models.document import SourceDocument
from evidenceengine.models.run import RunVersion
from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run

logger = logging.getLogger(__name__)

STAGE_STATUSES = ["queued", "parsing", "extracting", "retrieving", "classifying", "completed", "failed"]


async def _parse_pending_documents(packet_id: uuid.UUID, session: AsyncSession) -> None:
    """Parse any SourceDocument rows for this packet with parse_status='pending'.

    Runs PyMuPDF/pymupdf4llm in a worker thread so the event loop stays free.
    Commits each doc individually so a later failure doesn't lose prior parses.
    """
    result = await session.execute(
        select(SourceDocument).where(
            SourceDocument.packet_id == packet_id,
            SourceDocument.parse_status == "pending",
        )
    )
    pending = list(result.scalars().all())
    for doc in pending:
        logger.info("Parsing source document %s (%s)", doc.id, doc.filename)
        try:
            parsed = await asyncio.to_thread(validate_and_parse_file, doc.file_path)
        except Exception as exc:
            doc.parse_status = "failed"
            await session.commit()
            raise RuntimeError(f"Parse failed for {doc.filename}: {exc}") from exc
        doc.parsed_content = serialize_parsed_document(parsed)
        doc.raw_text = (parsed.raw_text or "").replace("\x00", "") or None
        doc.markdown_text = (parsed.markdown_text or "").replace("\x00", "") or None
        doc.total_pages = parsed.total_pages
        doc.parse_status = "completed"
        await session.commit()


async def _set_stage(run_version_id: str, stage: str, session: AsyncSession) -> RunVersion:
    """Load RunVersion, update status (and started_at if entering 'extracting'), commit."""
    run = await session.get(RunVersion, uuid.UUID(run_version_id))
    run.status = stage
    if stage == "extracting":
        run.started_at = datetime.now(timezone.utc)
    await session.commit()
    return run


async def _mark_failed(run_version_id: str, exc: Exception) -> None:
    """Write status=failed and error_summary using a **fresh** session.

    Called from the except block in run_full_pipeline — by then the primary
    session may be in a bad state, so we always open a new one here.
    """
    try:
        async with async_session_factory() as err_session:
            run = await err_session.get(RunVersion, uuid.UUID(run_version_id))
            if run is not None:
                run.status = "failed"
                run.error_summary = f"{type(exc).__name__}: {exc}"
                run.completed_at = datetime.now(timezone.utc)
                await err_session.commit()
    except Exception:
        logger.exception("_mark_failed itself failed for run %s — run may be left in intermediate state", run_version_id)


async def _classify_with_error_collection(
    run_version_id: str,
    session: AsyncSession,
) -> tuple[list, list[dict]]:
    """Wrap classify_verdicts_for_run, collecting stage-level errors without aborting.

    Per-claim error collection strategy (PIPE-05):
    The classification pipeline already handles zero-evidence and unresolvable-anchor
    edge cases gracefully (needs_review / insufficient_support). The remaining failure
    mode is LLM API errors, which abort the entire classify call. We collect those at
    the stage level here — a single claim_error entry is added if classification fails.

    Full per-claim granularity (v1.1 enhancement): would require modifying
    classify_verdicts_for_run to yield errors per claim. Not in scope for Phase 5.

    Returns:
        (verdicts, claim_errors) where claim_errors is [] on success.
    """
    claim_errors: list[dict] = []
    try:
        verdicts = await classify_verdicts_for_run(run_version_id, session)
    except Exception as exc:
        logger.warning(
            "Classification stage error for run %s: %s: %s",
            run_version_id,
            type(exc).__name__,
            exc,
        )
        claim_errors.append(
            {
                "stage": "classification",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        verdicts = []
    return verdicts, claim_errors


async def run_full_pipeline(run_version_id: str) -> None:
    """Background task: extract → retrieve → classify for a given RunVersion.

    Owns its own DB session (via async_session_factory). The RunVersion must
    already exist in the DB with status="queued" before this is called.

    Status progression:
        queued → extracting → retrieving → classifying → completed
        (or → failed if a top-level stage raises)

    Per-claim errors from classification are stored in pipeline_config["claim_errors"].
    The run still reaches "completed" even if some claims fail to classify.

    Top-level stage errors (extract/retrieve explode) → status=failed, error_summary set.

    Args:
        run_version_id: String UUID of the RunVersion row to process.

    Raises:
        Exception: Re-raises any top-level exception after marking the run as failed,
                   so that the task runner (uvicorn / asyncio) can log it.
    """
    try:
        # Only require an API key if the LLM classifier is actually selected.
        # NLI-primary mode (the default) runs fully locally with no API calls.
        if settings.classifier_backend != "nli_primary":
            key = settings.llm_api_key
            if not key or key.startswith("sk-REPLACE"):
                raise RuntimeError(
                    "LLM_API_KEY is not configured. Set a real key (e.g. an OpenRouter "
                    "sk-or-v1-… key) in .env before running the pipeline. "
                    "Legacy OPENAI_API_KEY is also accepted."
                )

        # Stages 0–2 use one session. It commits and closes before Stage 3 so
        # the DB connection is not held idle during the CPU-intensive NLI run.
        # A 190-page document can take 10+ minutes of NLI inference, which
        # causes PostgreSQL to close idle connections (InterfaceError).
        async with async_session_factory() as session:
            # ── Stage 0: Parse any pending source documents ───────────────────
            await _set_stage(run_version_id, "parsing", session)
            run = await session.get(RunVersion, uuid.UUID(run_version_id))
            await _parse_pending_documents(run.packet_id, session)

            # ── Stage 1: Extract ──────────────────────────────────────────────
            await _set_stage(run_version_id, "extracting", session)

            # Find the report SourceDocument for this run's packet
            result = await session.execute(
                select(SourceDocument).where(
                    SourceDocument.packet_id == run.packet_id,
                    SourceDocument.is_report.is_(True),
                )
            )
            report = result.scalar_one()

            await extract_claims_for_document(str(report.id), run_version_id, session)

            # ── Stage 2: Retrieve ─────────────────────────────────────────────
            await _set_stage(run_version_id, "retrieving", session)
            await retrieve_evidence_for_run(run_version_id, session)

            # Mark as classifying BEFORE closing this session so the UI
            # reflects the current stage while NLI runs in the next block.
            await _set_stage(run_version_id, "classifying", session)
        # Session closes here — connection returned to pool.

        # Stage 3 uses a fresh connection so no idle timeout during NLI.
        async with async_session_factory() as session:
            # ── Stage 3: Classify (with error collection) ─────────────────────
            _verdicts, claim_errors = await _classify_with_error_collection(
                run_version_id, session
            )

            # ── Finalize ──────────────────────────────────────────────────────
            run = await session.get(RunVersion, uuid.UUID(run_version_id))
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            run.pipeline_config = {
                **(run.pipeline_config or {}),
                "claim_errors": claim_errors,
                "failed_claim_count": len(claim_errors),
            }
            await session.commit()

            logger.info(
                "Pipeline completed for run %s — %d claim errors",
                run_version_id,
                len(claim_errors),
            )

    except Exception as exc:
        logger.error(
            "Pipeline failed for run %s: %s: %s",
            run_version_id,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        await _mark_failed(run_version_id, exc)
        # Do NOT re-raise: BackgroundTasks propagate exceptions up through
        # FastAPI's response pipeline, which triggers the request-scope session's
        # rollback handler and erases the run/packet rows the endpoint just inserted.
        # The failure is already recorded in run.error_summary via _mark_failed.
