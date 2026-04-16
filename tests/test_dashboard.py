"""TDD tests for Phase 6 dashboard backend.

Tests cover:
- compute_verdict_distribution (unit tests — pure function)
- load_dashboard_context queue sort order and filter behaviour (integration)
- load_claim_detail (integration)
- upsert_review_decision persist + overwrite (integration)
- POST /api/runs/{run_id}/claims/{claim_id}/review endpoint (integration)
- GET /dashboard/{pid}/{rid} and partials (integration)
- xfail: source badge rendering (template dependent — Plan 02)
"""

import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine",
)


# ---------------------------------------------------------------------------
# Function-scoped engine (avoids asyncpg event-loop conflicts)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def dash_engine():
    """Function-scoped async engine for dashboard tests."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(dash_engine) -> AsyncGenerator[AsyncSession, None]:
    """Function-scoped session — rolls back after each test."""
    factory = async_sessionmaker(dash_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    """HTTPX test client with get_db overridden to the test session."""
    from evidenceengine.api.app import create_app
    from evidenceengine.api.dependencies import get_db

    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# DB helper factories
# ---------------------------------------------------------------------------


async def make_packet(
    db: AsyncSession,
    raw_text: str = "This is a test report document with some content.",
) -> tuple:
    """Create DocumentPacket + report SourceDocument (with raw_text)."""
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
        raw_text=raw_text,
    )
    db.add(report_doc)
    await db.flush()

    return packet, report_doc


async def make_run(db: AsyncSession, packet_id: uuid.UUID, status: str = "completed"):
    """Create a RunVersion."""
    from evidenceengine.models.run import RunVersion

    run = RunVersion(packet_id=packet_id, status=status)
    db.add(run)
    await db.flush()
    return run


async def make_claim(
    db: AsyncSession,
    packet_id: uuid.UUID,
    source_document_id: uuid.UUID,
    run_version_id: uuid.UUID,
    claim_text: str = "The sky is blue.",
    char_start: int = 0,
    char_end: int = 16,
    status: str = "pending",
):
    """Create a Claim."""
    from evidenceengine.models.claim import Claim

    claim = Claim(
        packet_id=packet_id,
        source_document_id=source_document_id,
        run_version_id=run_version_id,
        claim_text=claim_text,
        char_start=char_start,
        char_end=char_end,
        status=status,
    )
    db.add(claim)
    await db.flush()
    return claim


async def make_verdict(
    db: AsyncSession,
    claim_id: uuid.UUID,
    run_version_id: uuid.UUID,
    verdict_type: str = "supported",
    confidence_score: float = 0.9,
):
    """Create a Verdict."""
    from evidenceengine.models.verdict import Verdict

    verdict = Verdict(
        claim_id=claim_id,
        run_version_id=run_version_id,
        verdict_type=verdict_type,
        confidence_score=confidence_score,
        reasoning="Test reasoning.",
        model_name="test-model",
    )
    db.add(verdict)
    await db.flush()
    return verdict


# ---------------------------------------------------------------------------
# Unit tests: compute_verdict_distribution (pure function — no DB needed)
# ---------------------------------------------------------------------------


def test_verdict_distribution_empty():
    """compute_verdict_distribution([]) returns all four keys with count=0, pct=0.0."""
    from evidenceengine.api.services.dashboard_queries import compute_verdict_distribution

    result = compute_verdict_distribution([])

    assert set(result.keys()) == {"contradicted", "needs_review", "insufficient_support", "supported"}
    for vtype, stats in result.items():
        assert stats["count"] == 0, f"{vtype}: expected count=0, got {stats['count']}"
        assert stats["pct"] == 0.0, f"{vtype}: expected pct=0.0, got {stats['pct']}"


def test_verdict_distribution_counts():
    """Mixed verdict list returns correct counts."""
    from evidenceengine.api.services.dashboard_queries import compute_verdict_distribution

    verdicts = [
        MagicMock(verdict_type="contradicted"),
        MagicMock(verdict_type="contradicted"),
        MagicMock(verdict_type="supported"),
        MagicMock(verdict_type="needs_review"),
        MagicMock(verdict_type="insufficient_support"),
    ]
    result = compute_verdict_distribution(verdicts)

    assert result["contradicted"]["count"] == 2
    assert result["supported"]["count"] == 1
    assert result["needs_review"]["count"] == 1
    assert result["insufficient_support"]["count"] == 1
    # Total: 5, so pct for contradicted = 40.0
    assert result["contradicted"]["pct"] == 40.0


def test_verdict_distribution_percentages():
    """1 contradicted of 4 total = 25.0%."""
    from evidenceengine.api.services.dashboard_queries import compute_verdict_distribution

    verdicts = [
        MagicMock(verdict_type="contradicted"),
        MagicMock(verdict_type="supported"),
        MagicMock(verdict_type="supported"),
        MagicMock(verdict_type="supported"),
    ]
    result = compute_verdict_distribution(verdicts)

    assert result["contradicted"]["count"] == 1
    assert result["contradicted"]["pct"] == 25.0
    assert result["supported"]["count"] == 3
    assert result["supported"]["pct"] == 75.0


# ---------------------------------------------------------------------------
# Integration: queue sort order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_sort_order(db_session: AsyncSession):
    """load_dashboard_context returns claims sorted contradicted-first."""
    from evidenceengine.api.services.dashboard_queries import load_dashboard_context

    packet, report_doc = await make_packet(db_session)
    run = await make_run(db_session, packet.id)

    # Create claims in reverse severity order
    claim_supported = await make_claim(
        db_session, packet.id, report_doc.id, run.id, claim_text="Supported claim."
    )
    claim_contradicted = await make_claim(
        db_session, packet.id, report_doc.id, run.id, claim_text="Contradicted claim."
    )
    claim_needs_review = await make_claim(
        db_session, packet.id, report_doc.id, run.id, claim_text="Needs review claim."
    )

    await make_verdict(db_session, claim_supported.id, run.id, verdict_type="supported")
    await make_verdict(db_session, claim_contradicted.id, run.id, verdict_type="contradicted")
    await make_verdict(db_session, claim_needs_review.id, run.id, verdict_type="needs_review")

    context = await load_dashboard_context(packet.id, run.id, db_session)

    assert context is not None
    claim_texts = [c.claim_text for c in context["claims"]]
    assert claim_texts[0] == "Contradicted claim."
    assert claim_texts[1] == "Needs review claim."
    assert claim_texts[2] == "Supported claim."


# ---------------------------------------------------------------------------
# Integration: queue filter by verdict_type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_filter_verdict(db_session: AsyncSession, client: AsyncClient):
    """GET /dashboard/{pid}/{rid}/queue?verdict_filter=contradicted only returns contradicted claims."""
    packet, report_doc = await make_packet(db_session)
    run = await make_run(db_session, packet.id)

    claim_a = await make_claim(
        db_session, packet.id, report_doc.id, run.id, claim_text="Claim A contradicted."
    )
    claim_b = await make_claim(
        db_session, packet.id, report_doc.id, run.id, claim_text="Claim B supported."
    )

    await make_verdict(db_session, claim_a.id, run.id, verdict_type="contradicted")
    await make_verdict(db_session, claim_b.id, run.id, verdict_type="supported")

    response = await client.get(
        f"/dashboard/{packet.id}/{run.id}/queue?verdict_filter=contradicted"
    )

    assert response.status_code == 200, response.text
    assert "Claim A contradicted." in response.text
    assert "Claim B supported." not in response.text


# ---------------------------------------------------------------------------
# Integration: queue filter by confidence band
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_filter_confidence(db_session: AsyncSession, client: AsyncClient):
    """GET /dashboard/{pid}/{rid}/queue?confidence=high returns only high-confidence claims."""
    packet, report_doc = await make_packet(db_session)
    run = await make_run(db_session, packet.id)

    high_claim = await make_claim(
        db_session, packet.id, report_doc.id, run.id, claim_text="High confidence claim."
    )
    low_claim = await make_claim(
        db_session, packet.id, report_doc.id, run.id, claim_text="Low confidence claim."
    )

    await make_verdict(db_session, high_claim.id, run.id, verdict_type="supported", confidence_score=0.9)
    await make_verdict(db_session, low_claim.id, run.id, verdict_type="supported", confidence_score=0.3)

    response = await client.get(
        f"/dashboard/{packet.id}/{run.id}/queue?confidence=high"
    )

    assert response.status_code == 200, response.text
    assert "High confidence claim." in response.text
    assert "Low confidence claim." not in response.text


# ---------------------------------------------------------------------------
# Integration: claim detail view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_detail_response(db_session: AsyncSession, client: AsyncClient):
    """GET /dashboard/{pid}/{rid}/claims/{cid} returns 200 HTML with claim text."""
    packet, report_doc = await make_packet(db_session)
    run = await make_run(db_session, packet.id)
    claim = await make_claim(
        db_session,
        packet.id,
        report_doc.id,
        run.id,
        claim_text="Unique claim detail text for testing.",
    )
    await make_verdict(db_session, claim.id, run.id, verdict_type="supported")

    response = await client.get(f"/dashboard/{packet.id}/{run.id}/claims/{claim.id}")

    assert response.status_code == 200, response.text
    assert "Unique claim detail text for testing." in response.text


# ---------------------------------------------------------------------------
# Integration: review persist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_persist(db_session: AsyncSession, client: AsyncClient):
    """POST /api/runs/{rid}/claims/{cid}/review with action=approve creates ReviewDecision in DB."""
    from evidenceengine.models.review import ReviewDecision

    packet, report_doc = await make_packet(db_session)
    run = await make_run(db_session, packet.id)
    claim = await make_claim(db_session, packet.id, report_doc.id, run.id)
    await make_verdict(db_session, claim.id, run.id, verdict_type="supported")

    response = await client.post(
        f"/api/runs/{run.id}/claims/{claim.id}/review",
        data={"action": "approve"},
    )

    assert response.status_code == 200, response.text

    # Verify DB state
    result = await db_session.execute(
        select(ReviewDecision).where(ReviewDecision.claim_id == claim.id)
    )
    decisions = result.scalars().all()
    assert len(decisions) == 1
    assert decisions[0].action == "approve"
    assert decisions[0].reviewer_id == "reviewer-stub"


# ---------------------------------------------------------------------------
# Integration: review overwrite (last-write-wins)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_overwrite(db_session: AsyncSession, client: AsyncClient):
    """Second POST with action=reject replaces approve decision."""
    from evidenceengine.models.review import ReviewDecision

    packet, report_doc = await make_packet(db_session)
    run = await make_run(db_session, packet.id)
    claim = await make_claim(db_session, packet.id, report_doc.id, run.id)
    await make_verdict(db_session, claim.id, run.id, verdict_type="supported")

    # First review
    r1 = await client.post(
        f"/api/runs/{run.id}/claims/{claim.id}/review",
        data={"action": "approve"},
    )
    assert r1.status_code == 200, r1.text

    # Second review — should overwrite
    r2 = await client.post(
        f"/api/runs/{run.id}/claims/{claim.id}/review",
        data={"action": "reject"},
    )
    assert r2.status_code == 200, r2.text

    # Verify only one decision remains and it is "reject"
    result = await db_session.execute(
        select(ReviewDecision).where(ReviewDecision.claim_id == claim.id)
    )
    decisions = result.scalars().all()
    assert len(decisions) == 1
    assert decisions[0].action == "reject"


# ---------------------------------------------------------------------------
# xfail: source badge rendering (templates not yet created in Plan 01)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_badge_rendering(db_session: AsyncSession, client: AsyncClient):
    """claim with retrieval_method='web_fallback' span returns HTML containing 'Web fallback'."""
    from evidenceengine.models.evidence import EvidenceSpan
    from evidenceengine.models.verdict import Verdict, VerdictEvidence
    from sqlalchemy import select as sa_select

    packet, report_doc = await make_packet(db_session)
    run = await make_run(db_session, packet.id)
    claim = await make_claim(db_session, packet.id, report_doc.id, run.id)
    await make_verdict(db_session, claim.id, run.id, verdict_type="supported")

    # Add a web_fallback evidence span
    span = EvidenceSpan(
        claim_id=claim.id,
        source_document_id=report_doc.id,
        run_version_id=run.id,
        span_text="Web fallback evidence.",
        retrieval_method="web_fallback",
        char_start=0,
        char_end=22,
    )
    db_session.add(span)
    await db_session.flush()

    # Link the span to the verdict via VerdictEvidence junction table
    result = await db_session.execute(
        sa_select(Verdict).where(Verdict.claim_id == claim.id)
    )
    verdict = result.scalar_one()
    ve = VerdictEvidence(verdict_id=verdict.id, evidence_span_id=span.id)
    db_session.add(ve)
    await db_session.flush()

    response = await client.get(f"/dashboard/{packet.id}/{run.id}/claims/{claim.id}")

    assert response.status_code == 200, response.text
    assert "Web fallback" in response.text
