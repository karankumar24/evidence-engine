"""Integration tests for classification pipeline and API endpoint.

TDD RED phase: written BEFORE implementation.
Uses real test DB with transaction rollback.
Mocks asyncio.to_thread to avoid real LLM calls.
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


def make_mock_message(verdict_type="supported", confidence_score=0.9, reasoning="Clear evidence supports the claim."):
    """Build the message object that asyncio.to_thread(_sync_request) returns."""
    from evidenceengine.classification.schemas import VerdictClassificationResponse

    mock_parsed = VerdictClassificationResponse(
        reasoning=reasoning,
        verdict_type=verdict_type,
        confidence_score=confidence_score,
    )
    mock_message = MagicMock()
    mock_message.parsed = mock_parsed
    mock_message.refusal = None
    return mock_message


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_single_claim_creates_verdict_and_evidence_rows(db_session):
    """Single claim with 2 EvidenceSpans: 1 Verdict + 2 VerdictEvidence rows in DB."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.verdict import Verdict, VerdictEvidence

    _, _, run_version, claims, spans = await make_packet_with_claim(db_session, n_claims=1, evidence_per_claim=2)

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message(verdict_type="supported", confidence_score=0.9, reasoning="Supported by evidence.")
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

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message(
            verdict_type="contradicted",
            confidence_score=0.85,
            reasoning="Evidence directly contradicts the claim.",
        )
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

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message()
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

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message()
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    assert verdicts[0].verdict_type == "insufficient_support"
    assert verdicts[0].confidence_score == 0.0
    # LLM was never called
    mock_thread.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_unresolvable_anchor_produces_needs_review(db_session):
    """Claim with status=unresolvable_anchor: verdict_type=needs_review, no LLM call."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run

    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2, claim_status="unresolvable_anchor"
    )

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message()
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    assert verdicts[0].verdict_type == "needs_review"
    assert verdicts[0].confidence_score == 0.0
    mock_thread.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_low_confidence_overridden_to_needs_review(db_session, monkeypatch):
    """Low confidence (0.4 < 0.7 threshold): verdict_type overridden to needs_review.

    Threshold-override is LLMClassifier-only behavior; nli_primary path skips it
    (Plan 03 gating). Pin the legacy backend explicitly.
    """
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings

    monkeypatch.setattr(settings, "classifier_backend", "llm_primary")
    _, _, run_version, _, _ = await make_packet_with_claim(db_session, n_claims=1, evidence_per_claim=2)

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message(verdict_type="supported", confidence_score=0.4)
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

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message()
        verdicts_first = await classify_verdicts_for_run(str(run_version.id), db_session)

    # Second call — should not raise UniqueConstraint
    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message()
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

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message()
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

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message()
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

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message()
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

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = make_mock_message()
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


def test_self_verify_threshold_band_invariant():
    """Lock the 0 < needs_review_threshold < self_verify_supported_cap < 1 band.

    These two numbers are deliberately co-designed. A change that swaps them
    (e.g. cap=0.70, threshold=0.80) would silently break the self-verify
    safety net: every uncapped SUPPORTED would pass the threshold even if
    the model was overconfident. Lock the invariant before anyone 'cleans
    up' the magic numbers.
    """
    from evidenceengine.core.config import settings

    threshold = settings.verdict_needs_review_threshold
    cap = settings.self_verify_supported_cap

    assert 0.0 < threshold, f"threshold must be > 0, got {threshold}"
    assert threshold < cap, (
        f"needs_review_threshold ({threshold}) must be strictly less than "
        f"self_verify_supported_cap ({cap}). See core/config.py band docstring."
    )
    assert cap < 1.0, f"cap must be < 1.0, got {cap}"
    # Floor the band-width: without at least 0.05 headroom between threshold
    # and cap, high-quality self-corroboration can never pass. If someone
    # tightens this, they're effectively disabling self-verify SUPPORTED.
    assert (cap - threshold) >= 0.05, (
        f"Self-verify band width is {cap - threshold:.3f}; need >=0.05 "
        "headroom for genuine corroboration to pass."
    )


# =============================================================================
# Phase 02 Plan 03: NLI-primary pipeline wiring tests.
# =============================================================================
#
# These tests exercise the new dispatch/tiebreaker/explanation/gating logic
# added to ``classify_verdicts_for_run``. They patch at the
# ``evidenceengine.classification.pipeline.*`` module level because Python
# import semantics bind the imported names inside the pipeline module — the
# place we control. Every test is hermetic (no real LLM, no real NLI) —
# failures here point at pipeline wiring, not flaky live services.


def _mock_verdict(
    verdict_type: str = "supported",
    confidence: float = 0.90,
    reasoning: str = "mocked verdict",
):
    from evidenceengine.classification.schemas import VerdictClassificationResponse

    return VerdictClassificationResponse(
        reasoning=reasoning,
        verdict_type=verdict_type,
        confidence_score=confidence,
    )


def _mock_backend(verdict):
    """Build a MagicMock whose async .classify returns ``verdict``."""
    mock = MagicMock()
    mock.classify = AsyncMock(return_value=verdict)
    return mock


@pytest.mark.asyncio
async def test_pipeline_dispatches_via_get_backend(db_session, monkeypatch):
    """get_backend() called ONCE per run, backend.classify ONCE per claim."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings

    monkeypatch.setattr(settings, "classifier_backend", "llm_primary")
    _, _, run_version, claims, _ = await make_packet_with_claim(
        db_session, n_claims=3, evidence_per_claim=2,
    )

    backend = _mock_backend(_mock_verdict("supported", 0.85))
    with patch.object(pipeline_mod, "get_backend", return_value=backend) as gb_spy, \
         patch.object(pipeline_mod, "generate_explanation", new=AsyncMock(return_value=None)):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 3
    # get_backend called exactly once per run (NOT per claim).
    assert gb_spy.call_count == 1
    # backend.classify called exactly once per claim.
    assert backend.classify.await_count == 3


@pytest.mark.asyncio
async def test_pipeline_nli_primary_tiebreaker_fires_below_threshold(db_session, monkeypatch):
    """NLI confidence 0.60 < 0.65 threshold → tiebreaker fires; telemetry ticks."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings
    from evidenceengine.models.run import RunVersion

    monkeypatch.setattr(settings, "classifier_backend", "nli_primary")
    monkeypatch.setattr(settings, "nli_tiebreaker_threshold", 0.65)

    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )

    nli_backend = _mock_backend(_mock_verdict("insufficient_support", 0.60))
    # Tiebreaker path imports LLMClassifier inside pipeline and calls .classify;
    # we patch LLMClassifier there to a higher-confidence stand-in.
    tiebreaker_verdict = _mock_verdict("supported", 0.88, "tiebreaker wins")
    tiebreaker_instance = _mock_backend(tiebreaker_verdict)

    with patch.object(pipeline_mod, "get_backend", return_value=nli_backend), \
         patch.object(pipeline_mod, "generate_explanation", new=AsyncMock(return_value=None)), \
         patch("evidenceengine.classification.llm_classifier.LLMClassifier",
               return_value=tiebreaker_instance) as llm_cls:
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    # Persisted verdict is the tiebreaker's verdict_type.
    # Confidence is bounded by the pre-existing self-verify cap (0.80) when evidence
    # comes from the same packet as the claim, which is the case in this fixture.
    assert verdicts[0].verdict_type == "supported"
    assert verdicts[0].confidence_score == pytest.approx(min(0.88, settings.self_verify_supported_cap))
    # Tiebreaker was constructed + called.
    assert llm_cls.call_count == 1
    assert tiebreaker_instance.classify.await_count == 1

    # Telemetry records the tiebreaker firing.
    refreshed = (await db_session.execute(
        select(RunVersion).where(RunVersion.id == run_version.id)
    )).scalar_one()
    telem = refreshed.pipeline_config["classification_telemetry"]
    assert telem["llm_tiebreaker_fired"] == 1
    assert telem["classifier_backend"] == "nli_primary"


@pytest.mark.asyncio
async def test_pipeline_nli_primary_tiebreaker_not_fired_above_threshold(db_session, monkeypatch):
    """NLI confidence 0.85 >= 0.65 → no tiebreaker; counter stays 0."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings
    from evidenceengine.models.run import RunVersion

    monkeypatch.setattr(settings, "classifier_backend", "nli_primary")
    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )

    nli_backend = _mock_backend(_mock_verdict("supported", 0.85))
    tiebreaker_instance = _mock_backend(_mock_verdict("contradicted", 0.99))

    with patch.object(pipeline_mod, "get_backend", return_value=nli_backend), \
         patch.object(pipeline_mod, "generate_explanation", new=AsyncMock(return_value=None)), \
         patch("evidenceengine.classification.llm_classifier.LLMClassifier",
               return_value=tiebreaker_instance) as llm_cls:
        await classify_verdicts_for_run(str(run_version.id), db_session)

    # Tiebreaker class must NOT have been constructed.
    assert llm_cls.call_count == 0
    assert tiebreaker_instance.classify.await_count == 0

    refreshed = (await db_session.execute(
        select(RunVersion).where(RunVersion.id == run_version.id)
    )).scalar_one()
    telem = refreshed.pipeline_config["classification_telemetry"]
    assert telem["llm_tiebreaker_fired"] == 0


@pytest.mark.asyncio
async def test_pipeline_nli_primary_tiebreaker_fail_open_on_chain_exhaustion(db_session, monkeypatch):
    """Tiebreaker raises RuntimeError → fail-open to needs_review (no exception)."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings
    from evidenceengine.models.run import RunVersion

    monkeypatch.setattr(settings, "classifier_backend", "nli_primary")
    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )

    nli_backend = _mock_backend(_mock_verdict("insufficient_support", 0.50))
    tiebreaker_instance = MagicMock()
    tiebreaker_instance.classify = AsyncMock(
        side_effect=RuntimeError("All models in fallback chain exhausted"),
    )

    with patch.object(pipeline_mod, "get_backend", return_value=nli_backend), \
         patch.object(pipeline_mod, "generate_explanation", new=AsyncMock(return_value=None)), \
         patch("evidenceengine.classification.llm_classifier.LLMClassifier",
               return_value=tiebreaker_instance):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    assert verdicts[0].verdict_type == "needs_review"
    assert abs(verdicts[0].confidence_score - 0.50) < 1e-9

    # llm_tiebreaker_fired STILL ticks to 1 (attempt counts per Pattern 4).
    refreshed = (await db_session.execute(
        select(RunVersion).where(RunVersion.id == run_version.id)
    )).scalar_one()
    telem = refreshed.pipeline_config["classification_telemetry"]
    assert telem["llm_tiebreaker_fired"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["nli_primary", "llm_primary"])
async def test_pipeline_explanation_fires_for_both_backends(
    db_session, monkeypatch, backend_name,
):
    """Explanation runs on BOTH backends; the string ends up in reasoning."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings

    monkeypatch.setattr(settings, "classifier_backend", backend_name)
    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )

    # Use a verdict that WILL NOT trigger the tiebreaker (nli_primary) and
    # WILL NOT trigger the 0.70 override (llm_primary) — 0.85 clears both.
    backend = _mock_backend(_mock_verdict("supported", 0.85, "base reasoning"))

    with patch.object(pipeline_mod, "get_backend", return_value=backend), \
         patch.object(pipeline_mod, "generate_explanation",
                      new=AsyncMock(return_value="explanation text")):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    assert "[explanation: explanation text]" in verdicts[0].reasoning


@pytest.mark.asyncio
async def test_pipeline_explanation_failure_does_not_block_verdict(db_session, monkeypatch):
    """generate_explanation returning None → verdict still persisted; counter ticks."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings
    from evidenceengine.models.run import RunVersion

    monkeypatch.setattr(settings, "classifier_backend", "nli_primary")
    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )

    backend = _mock_backend(_mock_verdict("supported", 0.85, "nli reasoning"))
    with patch.object(pipeline_mod, "get_backend", return_value=backend), \
         patch.object(pipeline_mod, "generate_explanation",
                      new=AsyncMock(return_value=None)):
        verdicts = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(verdicts) == 1
    # Reasoning has no [explanation: …] suffix and ends with the backend's text
    # (a pre-existing [Self-verify cap …] prefix may apply when evidence shares the
    # claim's packet — this fixture triggers that path).
    assert "[explanation:" not in verdicts[0].reasoning
    assert verdicts[0].reasoning.endswith("nli reasoning")

    refreshed = (await db_session.execute(
        select(RunVersion).where(RunVersion.id == run_version.id)
    )).scalar_one()
    telem = refreshed.pipeline_config["classification_telemetry"]
    assert telem["explanation_failed"] == 1
    assert telem["explanation_generated"] == 0


@pytest.mark.asyncio
async def test_pipeline_nli_primary_skips_nli_second_opinion(db_session, monkeypatch):
    """classifier_backend=nli_primary → legacy nli_second_opinion.nli_judgment NOT called."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings

    monkeypatch.setattr(settings, "classifier_backend", "nli_primary")
    monkeypatch.setattr(settings, "nli_second_opinion_enabled", True)

    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )

    backend = _mock_backend(_mock_verdict("supported", 0.85))
    with patch.object(pipeline_mod, "get_backend", return_value=backend), \
         patch.object(pipeline_mod, "generate_explanation", new=AsyncMock(return_value=None)), \
         patch("evidenceengine.classification.nli_second_opinion.nli_judgment") as nli_spy:
        await classify_verdicts_for_run(str(run_version.id), db_session)

    assert nli_spy.call_count == 0


@pytest.mark.asyncio
async def test_pipeline_llm_primary_still_runs_nli_second_opinion(db_session, monkeypatch):
    """classifier_backend=llm_primary → legacy nli_second_opinion.nli_judgment IS called per claim."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings

    monkeypatch.setattr(settings, "classifier_backend", "llm_primary")
    monkeypatch.setattr(settings, "nli_second_opinion_enabled", True)

    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=2, evidence_per_claim=2,
    )

    backend = _mock_backend(_mock_verdict("supported", 0.90))
    with patch.object(pipeline_mod, "get_backend", return_value=backend), \
         patch.object(pipeline_mod, "generate_explanation", new=AsyncMock(return_value=None)), \
         patch("evidenceengine.classification.nli_second_opinion.nli_judgment",
               return_value=None) as nli_spy:
        await classify_verdicts_for_run(str(run_version.id), db_session)

    assert nli_spy.call_count == 2


@pytest.mark.asyncio
async def test_pipeline_nli_primary_skips_apply_confidence_threshold(db_session, monkeypatch):
    """nli_primary + 0.66 confidence → verdict NOT forced to needs_review. llm_primary IS."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings

    # Use 0.66: above tiebreaker (0.65) so tiebreaker does NOT fire; below
    # legacy 0.70 override so llm_primary WOULD demote to needs_review.
    backend = _mock_backend(_mock_verdict("supported", 0.66))

    # --- nli_primary path ---
    monkeypatch.setattr(settings, "classifier_backend", "nli_primary")
    monkeypatch.setattr(settings, "nli_tiebreaker_threshold", 0.65)
    _, _, run_a, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )
    with patch.object(pipeline_mod, "get_backend", return_value=backend), \
         patch.object(pipeline_mod, "generate_explanation", new=AsyncMock(return_value=None)):
        vs_a = await classify_verdicts_for_run(str(run_a.id), db_session)
    assert vs_a[0].verdict_type == "supported"

    # --- llm_primary path (same scenario) ---
    monkeypatch.setattr(settings, "classifier_backend", "llm_primary")
    _, _, run_b, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )
    with patch.object(pipeline_mod, "get_backend", return_value=backend), \
         patch.object(pipeline_mod, "generate_explanation", new=AsyncMock(return_value=None)), \
         patch("evidenceengine.classification.nli_second_opinion.nli_judgment",
               return_value=None):
        vs_b = await classify_verdicts_for_run(str(run_b.id), db_session)
    assert vs_b[0].verdict_type == "needs_review"


@pytest.mark.asyncio
async def test_pipeline_idempotency_preserved_nli_primary(db_session, monkeypatch):
    """Re-run skips existing verdicts — backend.classify / tiebreaker / explanation unchanged."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings
    from evidenceengine.models.run import RunVersion

    monkeypatch.setattr(settings, "classifier_backend", "nli_primary")
    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )

    # Low-confidence NLI → tiebreaker fires on FIRST run; must NOT fire on rerun.
    backend = _mock_backend(_mock_verdict("insufficient_support", 0.50))
    tiebreaker_instance = _mock_backend(_mock_verdict("supported", 0.85))
    explanation_spy = AsyncMock(return_value="exp text")

    with patch.object(pipeline_mod, "get_backend", return_value=backend), \
         patch.object(pipeline_mod, "generate_explanation", new=explanation_spy), \
         patch("evidenceengine.classification.llm_classifier.LLMClassifier",
               return_value=tiebreaker_instance):
        first = await classify_verdicts_for_run(str(run_version.id), db_session)
        second = await classify_verdicts_for_run(str(run_version.id), db_session)

    assert len(first) == 1
    assert len(second) == 0
    # backend.classify called ONLY during the first run.
    assert backend.classify.await_count == 1
    # Tiebreaker called ONLY during the first run.
    assert tiebreaker_instance.classify.await_count == 1
    # Explanation called ONLY during the first run.
    assert explanation_spy.await_count == 1

    refreshed = (await db_session.execute(
        select(RunVersion).where(RunVersion.id == run_version.id)
    )).scalar_one()
    telem = refreshed.pipeline_config["classification_telemetry"]
    assert telem["llm_tiebreaker_fired"] == 0  # second run recomputed telem from scratch


@pytest.mark.asyncio
async def test_pipeline_telemetry_keys_nli_primary(db_session, monkeypatch):
    """classification_telemetry carries ALL additive keys alongside every pre-existing key."""
    from evidenceengine.classification import pipeline as pipeline_mod
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings
    from evidenceengine.models.run import RunVersion

    monkeypatch.setattr(settings, "classifier_backend", "nli_primary")
    _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=1, evidence_per_claim=2,
    )

    backend = _mock_backend(_mock_verdict("supported", 0.85))
    with patch.object(pipeline_mod, "get_backend", return_value=backend), \
         patch.object(pipeline_mod, "generate_explanation",
                      new=AsyncMock(return_value="exp")):
        await classify_verdicts_for_run(str(run_version.id), db_session)

    refreshed = (await db_session.execute(
        select(RunVersion).where(RunVersion.id == run_version.id)
    )).scalar_one()
    telem = refreshed.pipeline_config["classification_telemetry"]

    # Additive new keys (CLF-08).
    for k in (
        "classifier_backend",
        "nli_verdict_counts",
        "llm_tiebreaker_fired",
        "explanation_generated",
        "explanation_failed",
    ):
        assert k in telem, f"missing new telemetry key: {k}"
    # Pre-existing keys must ALL still be present (not renamed, not removed).
    for k in (
        "claims_total",
        "verdicts_produced",
        "self_verify_claims",
        "self_verify_rate",
        "self_verify_caps_applied",
        "threshold_overrides_to_review",
        "chain_exhausted_count",
        "model_refusal_count",
        "bypass_unresolvable_anchor",
        "nli_second_opinion_overrides",
    ):
        assert k in telem, f"existing telemetry key disappeared: {k}"
    # Types sanity.
    assert isinstance(telem["nli_verdict_counts"], dict)
    assert telem["classifier_backend"] == "nli_primary"
    assert telem["explanation_generated"] == 1
