"""TDD tests for the pipeline orchestrator and error/run schemas.

Tests cover:
- Schema contracts (ErrorEnvelope, APIError, RunTriggerResponse, RunStatusResponse, RunResultsResponse)
- Orchestrator happy path (status progression, started_at, completed_at, claim_errors=[])
- Status progression (queued -> extracting -> retrieving -> classifying -> completed)
- Per-claim error collection (classify fails → claim_errors populated, run still completed)
- Top-level failure recovery (extract raises → status=failed, error_summary set)
"""

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Schema contracts
# ──────────────────────────────────────────────────────────────────────────────


def test_error_schema_imports():
    """ErrorBody, ErrorEnvelope, APIError must be importable from schemas.common."""
    from evidenceengine.schemas.common import APIError, ErrorBody, ErrorEnvelope

    assert ErrorBody is not None
    assert ErrorEnvelope is not None
    assert APIError is not None


def test_run_schema_imports():
    """RunTriggerResponse, RunStatusResponse, RunResultsResponse must be importable."""
    from evidenceengine.schemas.run import (
        RunResultsResponse,
        RunStatusResponse,
        RunTriggerResponse,
    )

    assert RunTriggerResponse is not None
    assert RunStatusResponse is not None
    assert RunResultsResponse is not None


def test_orchestrator_import():
    """run_full_pipeline must be importable from pipeline.orchestrator."""
    from evidenceengine.pipeline.orchestrator import run_full_pipeline

    assert run_full_pipeline is not None


def test_error_envelope_shape():
    """ErrorEnvelope must serialize to {error: {code, message, detail}} shape."""
    from evidenceengine.schemas.common import ErrorBody, ErrorEnvelope

    body = ErrorBody(code="PACKET_NOT_FOUND", message="Packet not found", detail={"id": "abc"})
    envelope = ErrorEnvelope(error=body)
    data = envelope.model_dump()

    assert "error" in data
    assert data["error"]["code"] == "PACKET_NOT_FOUND"
    assert data["error"]["message"] == "Packet not found"
    assert data["error"]["detail"] == {"id": "abc"}


def test_api_error_construction():
    """APIError must be an Exception with code, message, detail, status."""
    from evidenceengine.schemas.common import APIError

    err = APIError(code="PACKET_NOT_FOUND", message="Packet not found", status=404)

    assert err.code == "PACKET_NOT_FOUND"
    assert err.message == "Packet not found"
    assert err.status == 404
    assert err.detail is None  # optional


def test_api_error_is_exception():
    """APIError must be raise-able as an exception."""
    from evidenceengine.schemas.common import APIError

    with pytest.raises(APIError) as exc_info:
        raise APIError(code="TEST_ERROR", message="test", status=400)

    assert exc_info.value.code == "TEST_ERROR"


def test_run_trigger_response():
    """RunTriggerResponse must have run_version_id (UUID) and status."""
    from evidenceengine.schemas.run import RunTriggerResponse

    run_id = uuid.uuid4()
    resp = RunTriggerResponse(run_version_id=run_id, status="queued")

    assert resp.run_version_id == run_id
    assert resp.status == "queued"


def test_run_status_response_from_orm():
    """RunStatusResponse must accept ORM object via model_validate (from_attributes=True)."""
    from evidenceengine.schemas.run import RunStatusResponse

    # Simulate ORM object with attribute access
    mock_run = MagicMock()
    mock_run.id = uuid.uuid4()
    mock_run.packet_id = uuid.uuid4()
    mock_run.status = "completed"
    mock_run.started_at = datetime.now(timezone.utc)
    mock_run.completed_at = datetime.now(timezone.utc)
    mock_run.error_summary = None
    mock_run.pipeline_config = {"failed_claim_count": 2, "claim_errors": []}

    resp = RunStatusResponse.model_validate(mock_run)

    assert resp.run_version_id == mock_run.id
    assert resp.packet_id == mock_run.packet_id
    assert resp.status == "completed"
    assert resp.failed_claim_count == 2


def test_run_status_response_no_pipeline_config():
    """RunStatusResponse.failed_claim_count defaults to 0 when pipeline_config is None."""
    from evidenceengine.schemas.run import RunStatusResponse

    mock_run = MagicMock()
    mock_run.id = uuid.uuid4()
    mock_run.packet_id = uuid.uuid4()
    mock_run.status = "queued"
    mock_run.started_at = None
    mock_run.completed_at = None
    mock_run.error_summary = None
    mock_run.pipeline_config = None

    resp = RunStatusResponse.model_validate(mock_run)

    assert resp.failed_claim_count == 0


def test_run_results_response_structure():
    """RunResultsResponse must have run_version_id, status, verdicts, claim_errors."""
    from evidenceengine.schemas.run import RunResultsResponse

    run_id = uuid.uuid4()
    resp = RunResultsResponse(
        run_version_id=run_id,
        status="completed",
        verdicts=[],
        claim_errors=[],
    )

    assert resp.run_version_id == run_id
    assert resp.verdicts == []
    assert resp.claim_errors == []


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Orchestrator happy path
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_happy_path(db_session):
    """run_full_pipeline completes with status=completed, started_at/completed_at set."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from evidenceengine.models.document import DocumentPacket, SourceDocument
    from evidenceengine.models.run import RunVersion

    # Create packet + report document + run version in DB
    packet = DocumentPacket(
        status="queued",
        report_filename="test.pdf",
        report_file_path="/tmp/test.pdf",
    )
    db_session.add(packet)
    await db_session.flush()

    report_doc = SourceDocument(
        packet_id=packet.id,
        is_report=True,
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_type="pdf",
        parsed_content={"blocks": []},
    )
    db_session.add(report_doc)
    await db_session.flush()

    run = RunVersion(packet_id=packet.id, status="queued")
    db_session.add(run)
    await db_session.commit()

    run_id_str = str(run.id)

    # Patch the three pipeline functions AND async_session_factory
    with (
        patch(
            "evidenceengine.pipeline.orchestrator.extract_claims_for_document",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_extract,
        patch(
            "evidenceengine.pipeline.orchestrator.retrieve_evidence_for_run",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_retrieve,
        patch(
            "evidenceengine.pipeline.orchestrator.classify_verdicts_for_run",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_classify,
        patch(
            "evidenceengine.pipeline.orchestrator.async_session_factory",
        ) as mock_factory,
    ):
        # Wire mock_factory to use the test db_session
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        from evidenceengine.pipeline.orchestrator import run_full_pipeline

        await run_full_pipeline(run_id_str)

    # Reload run from DB and verify
    await db_session.refresh(run)

    assert run.status == "completed"
    assert run.started_at is not None
    assert run.completed_at is not None
    assert run.pipeline_config is not None
    assert run.pipeline_config["claim_errors"] == []


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Status progression
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_progression(db_session):
    """RunVersion status progresses: queued -> extracting -> retrieving -> classifying -> completed."""
    from evidenceengine.models.document import DocumentPacket, SourceDocument
    from evidenceengine.models.run import RunVersion

    packet = DocumentPacket(
        status="queued",
        report_filename="test.pdf",
        report_file_path="/tmp/test.pdf",
    )
    db_session.add(packet)
    await db_session.flush()

    report_doc = SourceDocument(
        packet_id=packet.id,
        is_report=True,
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_type="pdf",
        parsed_content={"blocks": []},
    )
    db_session.add(report_doc)
    await db_session.flush()

    run = RunVersion(packet_id=packet.id, status="queued")
    db_session.add(run)
    await db_session.commit()

    run_id_str = str(run.id)
    statuses_seen: list[str] = []

    async def capture_extract(doc_id, run_ver_id, session):
        await session.refresh(run)
        statuses_seen.append(run.status)
        return []

    async def capture_retrieve(run_ver_id, session):
        await session.refresh(run)
        statuses_seen.append(run.status)
        return []

    async def capture_classify(run_ver_id, session):
        await session.refresh(run)
        statuses_seen.append(run.status)
        return []

    with (
        patch(
            "evidenceengine.pipeline.orchestrator.extract_claims_for_document",
            side_effect=capture_extract,
        ),
        patch(
            "evidenceengine.pipeline.orchestrator.retrieve_evidence_for_run",
            side_effect=capture_retrieve,
        ),
        patch(
            "evidenceengine.pipeline.orchestrator.classify_verdicts_for_run",
            side_effect=capture_classify,
        ),
        patch(
            "evidenceengine.pipeline.orchestrator.async_session_factory",
        ) as mock_factory,
    ):
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        from evidenceengine.pipeline.orchestrator import run_full_pipeline

        await run_full_pipeline(run_id_str)

    # When extract is called, status should be "extracting"
    # When retrieve is called, status should be "retrieving"
    # When classify is called, status should be "classifying"
    assert statuses_seen == ["extracting", "retrieving", "classifying"], (
        f"Expected ['extracting', 'retrieving', 'classifying'] but got {statuses_seen}"
    )

    await db_session.refresh(run)
    assert run.status == "completed"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Per-claim error collection
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_per_claim_error_collection(db_session):
    """When classify raises, error is collected in pipeline_config['claim_errors'], run still completed."""
    from evidenceengine.models.document import DocumentPacket, SourceDocument
    from evidenceengine.models.run import RunVersion

    packet = DocumentPacket(
        status="queued",
        report_filename="test.pdf",
        report_file_path="/tmp/test.pdf",
    )
    db_session.add(packet)
    await db_session.flush()

    report_doc = SourceDocument(
        packet_id=packet.id,
        is_report=True,
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_type="pdf",
        parsed_content={"blocks": []},
    )
    db_session.add(report_doc)
    await db_session.flush()

    run = RunVersion(packet_id=packet.id, status="queued")
    db_session.add(run)
    await db_session.commit()

    run_id_str = str(run.id)

    # classify_verdicts_for_run raises — should be caught as a claim error
    async def failing_classify(run_ver_id, session):
        raise RuntimeError("LLM API error: rate limit exceeded")

    with (
        patch(
            "evidenceengine.pipeline.orchestrator.extract_claims_for_document",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "evidenceengine.pipeline.orchestrator.retrieve_evidence_for_run",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "evidenceengine.pipeline.orchestrator.classify_verdicts_for_run",
            side_effect=failing_classify,
        ),
        patch(
            "evidenceengine.pipeline.orchestrator.async_session_factory",
        ) as mock_factory,
    ):
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        from evidenceengine.pipeline.orchestrator import run_full_pipeline

        await run_full_pipeline(run_id_str)

    await db_session.refresh(run)

    assert run.status == "completed", f"Expected 'completed' but got '{run.status}'"
    assert run.pipeline_config is not None
    assert len(run.pipeline_config["claim_errors"]) == 1
    assert run.pipeline_config["failed_claim_count"] == 1
    error_entry = run.pipeline_config["claim_errors"][0]
    assert error_entry["stage"] == "classification"
    assert "RuntimeError" in error_entry["error"]


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Top-level failure recovery
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_top_level_failure_recovery(db_session):
    """When extract raises, RunVersion.status=failed and error_summary is set."""
    from evidenceengine.models.document import DocumentPacket, SourceDocument
    from evidenceengine.models.run import RunVersion

    packet = DocumentPacket(
        status="queued",
        report_filename="test.pdf",
        report_file_path="/tmp/test.pdf",
    )
    db_session.add(packet)
    await db_session.flush()

    report_doc = SourceDocument(
        packet_id=packet.id,
        is_report=True,
        filename="test.pdf",
        file_path="/tmp/test.pdf",
        file_type="pdf",
        parsed_content={"blocks": []},
    )
    db_session.add(report_doc)
    await db_session.flush()

    run = RunVersion(packet_id=packet.id, status="queued")
    db_session.add(run)
    await db_session.commit()

    run_id_str = str(run.id)

    async def failing_extract(doc_id, run_ver_id, session):
        raise RuntimeError("network timeout")

    with (
        patch(
            "evidenceengine.pipeline.orchestrator.extract_claims_for_document",
            side_effect=failing_extract,
        ),
        patch(
            "evidenceengine.pipeline.orchestrator.async_session_factory",
        ) as mock_factory,
        patch(
            "evidenceengine.pipeline.orchestrator._mark_failed",
            new_callable=AsyncMock,
        ) as mock_mark_failed,
    ):
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        from evidenceengine.pipeline.orchestrator import run_full_pipeline

        # run_full_pipeline re-raises after marking failed
        with pytest.raises(RuntimeError, match="network timeout"):
            await run_full_pipeline(run_id_str)

    # _mark_failed must have been called with correct args
    mock_mark_failed.assert_called_once()
    call_args = mock_mark_failed.call_args
    assert call_args[0][0] == run_id_str  # first positional arg = run_version_id
    assert isinstance(call_args[0][1], RuntimeError)
    assert "network timeout" in str(call_args[0][1])


@pytest.mark.asyncio
async def test_mark_failed_writes_db(db_session):
    """_mark_failed writes status=failed and error_summary using a fresh session."""
    from evidenceengine.models.document import DocumentPacket, SourceDocument
    from evidenceengine.models.run import RunVersion

    packet = DocumentPacket(
        status="queued",
        report_filename="test.pdf",
        report_file_path="/tmp/test.pdf",
    )
    db_session.add(packet)
    await db_session.flush()

    run = RunVersion(packet_id=packet.id, status="extracting")
    db_session.add(run)
    await db_session.commit()

    run_id_str = str(run.id)
    exc = RuntimeError("network timeout")

    # _mark_failed uses async_session_factory internally for a fresh session
    with patch(
        "evidenceengine.pipeline.orchestrator.async_session_factory",
    ) as mock_factory:
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        from evidenceengine.pipeline.orchestrator import _mark_failed

        await _mark_failed(run_id_str, exc)

    await db_session.refresh(run)

    assert run.status == "failed"
    assert run.error_summary is not None
    assert "RuntimeError" in run.error_summary
    assert "network timeout" in run.error_summary
    assert run.completed_at is not None
