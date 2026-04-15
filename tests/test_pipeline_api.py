"""TDD API tests for Phase 5 pipeline endpoints.

Tests the three REST endpoints introduced in Phase 5:
  POST /api/packets/{packet_id}/run  — trigger async pipeline run
  GET  /api/runs/{run_id}            — poll run status
  GET  /api/runs/{run_id}/results    — retrieve full verdict + error list

All tests use httpx AsyncClient + SQLAlchemy rollback fixtures.
run_full_pipeline is always patched to a no-op so tests don't hit the real pipeline.
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine):
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    """HTTPX test client with get_db dependency overridden to the test session."""
    from evidenceengine.api.app import create_app
    from evidenceengine.api.dependencies import get_db

    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def make_packet(db: AsyncSession) -> tuple:
    """Create a DocumentPacket + report SourceDocument; flush but do not commit."""
    from evidenceengine.models.document import DocumentPacket, SourceDocument

    packet = DocumentPacket(
        status="queued",
        report_filename="report.pdf",
        report_file_path="/uploads/report.pdf",
    )
    db.add(packet)
    await db.flush()

    report_doc = SourceDocument(
        packet_id=packet.id,
        is_report=True,
        filename="report.pdf",
        file_path="/uploads/report.pdf",
        file_type="pdf",
        parsed_content={"blocks": []},
    )
    db.add(report_doc)
    await db.flush()

    return packet, report_doc


async def make_run_version(
    db: AsyncSession,
    packet_id: uuid.UUID,
    status: str = "queued",
    pipeline_config: dict | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> object:
    """Create a RunVersion row with given state; flush to populate id."""
    from evidenceengine.models.run import RunVersion

    run = RunVersion(
        packet_id=packet_id,
        status=status,
        pipeline_config=pipeline_config,
        started_at=started_at,
        completed_at=completed_at,
    )
    db.add(run)
    await db.flush()
    return run


async def make_verdict(
    db: AsyncSession,
    claim_id: uuid.UUID,
    run_version_id: uuid.UUID,
) -> object:
    """Create a Verdict row; flush to populate id."""
    from evidenceengine.models.verdict import Verdict

    verdict = Verdict(
        claim_id=claim_id,
        run_version_id=run_version_id,
        verdict_type="supported",
        confidence_score=0.95,
        reasoning="Strong evidence supports this claim.",
        model_name="gpt-4o-mini",
    )
    db.add(verdict)
    await db.flush()
    return verdict


async def make_claim(
    db: AsyncSession,
    packet_id: uuid.UUID,
    source_document_id: uuid.UUID,
    run_version_id: uuid.UUID,
) -> object:
    """Create a Claim row; flush to populate id."""
    from evidenceengine.models.claim import Claim

    claim = Claim(
        packet_id=packet_id,
        source_document_id=source_document_id,
        run_version_id=run_version_id,
        claim_text="The study shows 95% efficacy.",
        char_start=0,
        char_end=30,
        status="pending",
    )
    db.add(claim)
    await db.flush()
    return claim


# ---------------------------------------------------------------------------
# Test 1: POST /api/packets/{id}/run — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_run_returns_202_with_queued_status(
    db_session: AsyncSession, client: AsyncClient
):
    """POST /run creates RunVersion, returns 202 with run_version_id + status=queued."""
    from evidenceengine.models.run import RunVersion
    from sqlalchemy import select

    packet, _report = await make_packet(db_session)

    with patch(
        "evidenceengine.api.routes.pipeline.run_full_pipeline",
        new_callable=AsyncMock,
    ) as mock_pipeline:
        response = await client.post(f"/api/packets/{packet.id}/run")

    assert response.status_code == 202, response.text
    body = response.json()
    assert "run_version_id" in body
    assert body["status"] == "queued"

    # Verify RunVersion was persisted
    run_id = uuid.UUID(body["run_version_id"])
    result = await db_session.execute(
        select(RunVersion).where(RunVersion.id == run_id)
    )
    run = result.scalar_one_or_none()
    assert run is not None
    assert run.status == "queued"

    # Verify background task was scheduled with string run_id
    mock_pipeline.assert_called_once_with(str(run_id))


# ---------------------------------------------------------------------------
# Test 2: POST /api/packets/{id}/run — packet not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trigger_run_packet_not_found_returns_error_envelope(client: AsyncClient):
    """POST /run for nonexistent packet returns 404 with structured error envelope."""
    nonexistent_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    response = await client.post(f"/api/packets/{nonexistent_id}/run")

    assert response.status_code == 404, response.text
    body = response.json()

    # Must use error envelope — NOT FastAPI's default {"detail": "..."}
    assert "error" in body, f"Expected 'error' key, got: {body}"
    assert "detail" not in body or body.get("detail") is None
    assert body["error"]["code"] == "PACKET_NOT_FOUND"
    assert "message" in body["error"]


# ---------------------------------------------------------------------------
# Test 3: GET /api/runs/{id} — status polling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_run_status_returns_current_state(
    db_session: AsyncSession, client: AsyncClient
):
    """GET /runs/{id} returns run_version_id, status, started_at, failed_claim_count."""
    packet, _report = await make_packet(db_session)
    run = await make_run_version(
        db_session,
        packet_id=packet.id,
        status="classifying",
        started_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    response = await client.get(f"/api/runs/{run.id}")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_version_id"] == str(run.id)
    assert body["status"] == "classifying"
    assert body["started_at"] is not None
    assert body["failed_claim_count"] == 0


# ---------------------------------------------------------------------------
# Test 4: GET /api/runs/{id}/results — after completion with verdicts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_run_results_returns_verdicts_and_claim_errors(
    db_session: AsyncSession, client: AsyncClient
):
    """GET /runs/{id}/results returns verdict list + empty claim_errors."""
    packet, report_doc = await make_packet(db_session)
    run = await make_run_version(
        db_session,
        packet_id=packet.id,
        status="completed",
        pipeline_config={"claim_errors": [], "failed_claim_count": 0},
        completed_at=datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
    )

    # Create a claim (needed for the Verdict FK)
    claim = await make_claim(
        db_session,
        packet_id=packet.id,
        source_document_id=report_doc.id,
        run_version_id=run.id,
    )

    # Create a Verdict
    verdict = await make_verdict(db_session, claim_id=claim.id, run_version_id=run.id)

    response = await client.get(f"/api/runs/{run.id}/results")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["run_version_id"] == str(run.id)
    assert body["status"] == "completed"
    assert body["claim_errors"] == []
    assert len(body["verdicts"]) == 1

    v = body["verdicts"][0]
    assert v["verdict_type"] == "supported"
    assert v["confidence_score"] == pytest.approx(0.95)
    assert "reasoning" in v
    assert "claim_id" in v


# ---------------------------------------------------------------------------
# Test 5: GET /api/runs/{id}/results — with partial classification failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_run_results_with_partial_failures(
    db_session: AsyncSession, client: AsyncClient
):
    """GET /runs/{id}/results surfaces claim_errors from pipeline_config."""
    packet, _report = await make_packet(db_session)
    run = await make_run_version(
        db_session,
        packet_id=packet.id,
        status="completed",
        pipeline_config={
            "claim_errors": [{"stage": "classification", "error": "timeout"}],
            "failed_claim_count": 1,
        },
    )

    response = await client.get(f"/api/runs/{run.id}/results")

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["claim_errors"]) == 1
    assert body["claim_errors"][0]["stage"] == "classification"
    assert body["claim_errors"][0]["error"] == "timeout"


# ---------------------------------------------------------------------------
# Test 6: Error envelope — validation error (invalid UUID in path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_uuid_in_path_returns_error_envelope(client: AsyncClient):
    """POST /api/packets/not-a-uuid/run returns error envelope, not FastAPI default shape."""
    response = await client.post("/api/packets/not-a-uuid/run")

    # FastAPI returns 422 for path validation failures
    assert response.status_code in (422, 404), response.text
    body = response.json()

    # Must NOT be FastAPI default {"detail": [...]}
    assert "error" in body, f"Expected 'error' key, got: {body}"
    assert body["error"]["code"] in ("VALIDATION_ERROR", "HTTP_422", "HTTP_404")
    assert "message" in body["error"]


# ---------------------------------------------------------------------------
# Test 7: GET /api/runs/nonexistent — run not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_run_status_not_found_returns_error_envelope(client: AsyncClient):
    """GET /runs/{nonexistent_id} returns 404 with error envelope."""
    nonexistent_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
    response = await client.get(f"/api/runs/{nonexistent_id}")

    assert response.status_code == 404, response.text
    body = response.json()

    assert "error" in body, f"Expected 'error' key, got: {body}"
    assert body["error"]["code"] == "RUN_NOT_FOUND"
    assert "message" in body["error"]
