"""Integration tests for classification pipeline and API endpoint.

TDD RED phase: written BEFORE implementation.
Uses real test DB with transaction rollback.
Mocks AsyncOpenAI to avoid real LLM calls.
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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


async def make_packet_with_claim(
    db: AsyncSession,
    n_claims: int = 1,
    evidence_per_claim: int = 2,
    claim_status: str = "pending",
):
    """Helper: create DocumentPacket + report SourceDocument + RunVersion + Claim(s) + EvidenceSpan(s).

    Returns: (packet, report_doc, run_version, claims, evidence_spans)
    """
    from evidenceengine.models.document import DocumentPacket, SourceDocument
    from evidenceengine.models.run import RunVersion
    from evidenceengine.models.claim import Claim
    from evidenceengine.models.evidence import EvidenceSpan

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

    run_version = RunVersion(
        packet_id=packet.id,
        status="retrieval_complete",
        pipeline_config={},
    )
    db.add(run_version)
    await db.flush()

    claims = []
    all_spans = []

    for _ in range(n_claims):
        claim = Claim(
            packet_id=packet.id,
            source_document_id=report_doc.id,
            run_version_id=run_version.id,
            claim_text="Carbon emissions increased by 12% during this period.",
            status=claim_status,
        )
        db.add(claim)
        await db.flush()

        spans = []
        for i in range(evidence_per_claim):
            span = EvidenceSpan(
                claim_id=claim.id,
                source_document_id=report_doc.id,
                run_version_id=run_version.id,
                span_text=f"Evidence text {i + 1}: Carbon emissions rose significantly.",
                char_start=i * 100,
                char_end=(i * 100) + 80,
                relevance_score=0.8 - (i * 0.1),
                retrieval_method="cross_encoder",
                retrieval_rank=i + 1,
            )
            db.add(span)
            spans.append(span)

        await db.flush()
        claims.append(claim)
        all_spans.extend(spans)

    return packet, report_doc, run_version, claims, all_spans


def make_mock_openai(verdict_type="supported", confidence_score=0.9, reasoning="Clear evidence supports the claim."):
    """Create a mock AsyncOpenAI client that returns a structured verdict response."""
    from evidenceengine.classification.schemas import VerdictClassificationResponse

    mock_parsed = VerdictClassificationResponse(
        reasoning=reasoning,
        verdict_type=verdict_type,
        confidence_score=confidence_score,
    )
    mock_message = MagicMock()
    mock_message.parsed = mock_parsed
    mock_message.refusal = None

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=mock_message)]

    mock_beta = MagicMock()
    mock_beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    mock_client = MagicMock()
    mock_client.beta = mock_beta

    return mock_client


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_single_claim_creates_verdict_and_evidence_rows(db_session):
    """Single claim with 2 EvidenceSpans: 1 Verdict + 2 VerdictEvidence rows in DB."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.verdict import Verdict, VerdictEvidence

    _, _, run_version, claims, spans = await make_packet_with_claim(db_session, n_claims=1, evidence_per_claim=2)

    mock_client = make_mock_openai(verdict_type="supported", confidence_score=0.9, reasoning="Supported by evidence.")

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1

    # Reload verdict from DB
    result = await db_session.execute(
        select(Verdict).where(Verdict.run_version_id == run_version.id)
    )
    db_verdicts = result.scalars().all()
    assert len(db_verdicts) == 1

    # Check VerdictEvidence rows
    ve_result = await db_session.execute(
        select(VerdictEvidence).where(VerdictEvidence.verdict_id == db_verdicts[0].id)
    )
    ve_rows = ve_result.scalars().all()
    assert len(ve_rows) == 2


@pytest.mark.asyncio
async def test_pipeline_verdict_fields_match_llm_response(db_session):
    """Verdict.verdict_type, reasoning, confidence_score match mocked LLM response."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.verdict import Verdict

    _, _, run_version, _, _ = await make_packet_with_claim(db_session, n_claims=1, evidence_per_claim=2)

    mock_client = make_mock_openai(
        verdict_type="contradicted",
        confidence_score=0.85,
        reasoning="Evidence directly contradicts the claim.",
    )

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.verdict_type == "contradicted"
    assert v.confidence_score == 0.85
    assert v.reasoning == "Evidence directly contradicts the claim."


@pytest.mark.asyncio
async def test_pipeline_verdict_evidence_weight_matches_relevance_score(db_session):
    """VerdictEvidence.weight == EvidenceSpan.relevance_score for each linked span."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.verdict import Verdict, VerdictEvidence
    from evidenceengine.models.evidence import EvidenceSpan

    _, _, run_version, claims, spans = await make_packet_with_claim(db_session, n_claims=1, evidence_per_claim=2)

    mock_client = make_mock_openai()

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        await classify_verdicts_for_run(str(run_version.id), db_session)

    verdict_result = await db_session.execute(
        select(Verdict).where(Verdict.run_version_id == run_version.id)
    )
    verdict = verdict_result.scalar_one()

    ve_result = await db_session.execute(
        select(VerdictEvidence).where(VerdictEvidence.verdict_id == verdict.id)
    )
    ve_rows = ve_result.scalars().all()
    assert len(ve_rows) == 2

    # Map evidence_span_id to weight
    span_weights = {ve.evidence_span_id: ve.weight for ve in ve_rows}

    for span in spans:
        assert span.id in span_weights
        assert abs(span_weights[span.id] - (span.relevance_score or 0.0)) < 0.001


@pytest.mark.asyncio
async def test_pipeline_zero_evidence_produces_insufficient_support(db_session):
    """Claim with zero EvidenceSpans: verdict_type=insufficient_support, no LLM call."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.verdict import Verdict

    _, _, run_version, _, _ = await make_packet_with_claim(db_session, n_claims=1, evidence_per_claim=0)

    mock_client = make_mock_openai()

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    assert verdicts[0].verdict_type == "insufficient_support"
    assert verdicts[0].confidence_score == 0.0
    # LLM was never called
    mock_client.beta.chat.completions.parse.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_unresolvable_anchor_produces_needs_review(db_session):
    """Claim with status=unresolvable_anchor: verdict_type=needs_review, no LLM call."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run

    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2, claim_status="unresolvable_anchor"
    )

    mock_client = make_mock_openai()

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    assert verdicts[0].verdict_type == "needs_review"
    assert verdicts[0].confidence_score == 0.0
    mock_client.beta.chat.completions.parse.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_low_confidence_overridden_to_needs_review(db_session):
    """Low confidence (0.4 < 0.7 threshold): verdict_type overridden to needs_review."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run

    _, _, run_version, _, _ = await make_packet_with_claim(db_session, n_claims=1, evidence_per_claim=2)

    mock_client = make_mock_openai(verdict_type="supported", confidence_score=0.4)

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    # Threshold routing should have overridden supported -> needs_review
    assert verdicts[0].verdict_type == "needs_review"
    assert verdicts[0].confidence_score == 0.4  # original score preserved


@pytest.mark.asyncio
async def test_pipeline_idempotent_no_duplicate_verdicts(db_session):
    """Calling pipeline twice with same run_version_id: second call skips existing (no IntegrityError)."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.verdict import Verdict

    _, _, run_version, _, _ = await make_packet_with_claim(db_session, n_claims=1, evidence_per_claim=2)

    mock_client = make_mock_openai()

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        verdicts_first = await classify_verdicts_for_run(str(run_version.id), db_session)

    # Second call — should not raise UniqueConstraint
    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        verdicts_second = await classify_verdicts_for_run(str(run_version.id), db_session)

    result = await db_session.execute(
        select(Verdict).where(Verdict.run_version_id == run_version.id)
    )
    all_verdicts = result.scalars().all()
    assert len(all_verdicts) == 1  # not doubled
    assert len(verdicts_second) == 0  # skipped existing


@pytest.mark.asyncio
async def test_pipeline_multiple_claims_one_verdict_each(db_session):
    """Multiple claims: one Verdict per claim produced."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.verdict import Verdict

    _, _, run_version, claims, _ = await make_packet_with_claim(db_session, n_claims=3, evidence_per_claim=2)

    mock_client = make_mock_openai()

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 3

    result = await db_session.execute(
        select(Verdict).where(Verdict.run_version_id == run_version.id)
    )
    db_verdicts = result.scalars().all()
    assert len(db_verdicts) == 3


@pytest.mark.asyncio
async def test_pipeline_empty_run_returns_empty_list(db_session):
    """Pipeline with no claims returns empty list — no errors."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run

    _, _, run_version, _, _ = await make_packet_with_claim(db_session, n_claims=0)

    mock_client = make_mock_openai()

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert verdicts == []


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def async_client(db_session):
    """HTTP client with DB dependency overridden to use test session."""
    from evidenceengine.api.app import create_app
    from evidenceengine.api.dependencies import get_db

    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_classify_endpoint_404_for_unknown_packet(async_client):
    """POST /api/packets/{unknown_id}/classify → 404."""
    unknown_id = uuid.uuid4()
    response = await async_client.post(f"/api/packets/{unknown_id}/classify")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_classify_endpoint_422_no_run_version(async_client, db_session):
    """POST /api/packets/{id}/classify with no RunVersion → 422."""
    from evidenceengine.models.document import DocumentPacket

    packet = DocumentPacket(
        status="queued",
        report_filename="report.pdf",
        report_file_path="/uploads/report.pdf",
    )
    db_session.add(packet)
    await db_session.flush()

    response = await async_client.post(f"/api/packets/{packet.id}/classify")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_classify_endpoint_returns_202(async_client, db_session):
    """POST /api/packets/{id}/classify returns 202 with run_version_id, verdict_count, claim_count."""
    _, _, run_version, _, _ = await make_packet_with_claim(db_session, n_claims=2, evidence_per_claim=2)

    # Get packet_id from run_version
    from evidenceengine.models.run import RunVersion
    result = await db_session.execute(
        select(RunVersion).where(RunVersion.id == run_version.id)
    )
    rv = result.scalar_one()
    packet_id = rv.packet_id

    mock_client = make_mock_openai()

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        response = await async_client.post(f"/api/packets/{packet_id}/classify")

    assert response.status_code == 202
    data = response.json()
    assert "run_version_id" in data
    assert "verdict_count" in data
    assert "claim_count" in data
    assert data["verdict_count"] == 2
    assert data["claim_count"] == 2


@pytest.mark.asyncio
async def test_classify_endpoint_sets_run_version_fields(async_client, db_session):
    """After successful classify: RunVersion.status, model_versions, completed_at are set."""
    from evidenceengine.models.run import RunVersion

    packet, _, run_version, _, _ = await make_packet_with_claim(db_session, n_claims=1, evidence_per_claim=2)

    mock_client = make_mock_openai()

    with patch("evidenceengine.classification.classifier.AsyncOpenAI", return_value=mock_client):
        response = await async_client.post(f"/api/packets/{packet.id}/classify")

    assert response.status_code == 202

    # Reload run_version
    result = await db_session.execute(
        select(RunVersion).where(RunVersion.id == run_version.id)
    )
    updated_rv = result.scalar_one()

    assert updated_rv.status == "classification_complete"
    assert updated_rv.completed_at is not None
    assert updated_rv.model_versions is not None
    from evidenceengine.core.config import settings
    assert updated_rv.model_versions.get("classification_model") == settings.classification_model
