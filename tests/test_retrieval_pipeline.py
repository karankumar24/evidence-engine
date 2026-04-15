"""Integration tests for retrieval pipeline, recall logger, and API endpoint.

TDD RED phase: written BEFORE implementation.
Uses real test DB (db_session from conftest) with transaction rollback.
Mocks reranker.rerank and bm25_retriever functions to avoid disk I/O and model inference.
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

SAMPLE_PARSED_CONTENT = {
    "blocks": [
        {
            "text": "Carbon emissions increased by 12% in this period, according to climate data.",
            "block_type": "paragraph",
            "position": {"page": 3, "paragraph": 2, "char_start": 0, "char_end": 74, "section_header": "Results"},
        },
        {
            "text": "Sea ice extent has declined significantly in the Arctic region over the last decade.",
            "block_type": "paragraph",
            "position": {"page": 4, "paragraph": 1, "char_start": 75, "char_end": 157, "section_header": "Discussion"},
        },
    ]
}

MOCK_RERANKED = [
    {"corpus_id": 0, "score": 8.5, "text": "Carbon emissions increased by 12% in this period, according to climate data."},
    {"corpus_id": 1, "score": 6.2, "text": "Sea ice extent has declined significantly in the Arctic region over the last decade."},
    {"corpus_id": 0, "score": 4.1, "text": "Carbon emissions increased by 12% in this period, according to climate data."},
    {"corpus_id": 1, "score": 3.0, "text": "Sea ice extent has declined significantly in the Arctic region over the last decade."},
    {"corpus_id": 0, "score": 1.5, "text": "Carbon emissions increased by 12% in this period, according to climate data."},
]


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
    with_source_doc: bool = True,
    resolution_status: str = "resolved",
    n_claims: int = 1,
    n_resolved_per_claim: int = 1,
):
    """Helper: create DocumentPacket + report SourceDocument + source SourceDocument
    + RunVersion + Claim(s) + CitationAnchor(s) in DB.

    Returns: (packet, report_doc, source_doc, run_version, claims, anchors)
    """
    from evidenceengine.models.document import DocumentPacket, SourceDocument
    from evidenceengine.models.run import RunVersion
    from evidenceengine.models.claim import Claim, CitationAnchor

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

    source_doc = None
    if with_source_doc:
        source_doc = SourceDocument(
            packet_id=packet.id,
            is_report=False,
            filename="source.pdf",
            file_path="/uploads/source.pdf",
            file_type="pdf",
            parsed_content=SAMPLE_PARSED_CONTENT,
        )
        db.add(source_doc)

    run_version = RunVersion(
        packet_id=packet.id,
        status="extraction_complete",
        pipeline_config={},
    )
    db.add(run_version)
    await db.flush()

    claims = []
    anchors = []

    for _ in range(n_claims):
        claim = Claim(
            packet_id=packet.id,
            source_document_id=report_doc.id,
            run_version_id=run_version.id,
            claim_text="Carbon emissions increased by 12% during this period [1].",
            status="completed",
        )
        db.add(claim)
        await db.flush()

        for _ in range(n_resolved_per_claim):
            anchor = CitationAnchor(
                claim_id=claim.id,
                raw_marker="[1]",
                citation_style="numeric",
                target_document_id=source_doc.id if source_doc else None,
                resolution_status=resolution_status,
            )
            db.add(anchor)
            anchors.append(anchor)

        claims.append(claim)

    await db.flush()
    return packet, report_doc, source_doc, run_version, claims, anchors


# Mock helpers
def mock_bm25_functions(monkeypatch):
    """Patch BM25 functions to avoid real disk I/O."""
    mock_retriever = MagicMock()

    def fake_load_or_build(spans, index_dir):
        return mock_retriever

    def fake_query_index(retriever, query, k=10):
        texts = [b["text"] for b in SAMPLE_PARSED_CONTENT["blocks"]]
        scores = [5.0] * len(texts)
        return texts[:k], scores[:k]

    monkeypatch.setattr("evidenceengine.retrieval.pipeline.load_or_build_index", fake_load_or_build)
    monkeypatch.setattr("evidenceengine.retrieval.pipeline.query_index", fake_query_index)
    return mock_retriever


def mock_reranker(monkeypatch):
    """Patch rerank to avoid real model inference."""
    async def fake_rerank(query, passages):
        return MOCK_RERANKED[:len(passages)] if passages else []

    monkeypatch.setattr("evidenceengine.retrieval.pipeline.rerank", fake_rerank)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_creates_evidence_span_rows(db_session, monkeypatch):
    """Pipeline creates EvidenceSpan rows (up to retrieval_top_k_final=5)."""
    from evidenceengine.models.evidence import EvidenceSpan
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run

    packet, _, source_doc, run_version, claims, _ = await make_packet_with_claim(db_session)
    mock_bm25_functions(monkeypatch)
    mock_reranker(monkeypatch)

    spans = await retrieve_evidence_for_run(str(run_version.id), db_session)
    assert len(spans) > 0
    assert len(spans) <= 5  # retrieval_top_k_final default


@pytest.mark.asyncio
async def test_pipeline_sets_retrieval_method_cross_encoder(db_session, monkeypatch):
    """Persisted EvidenceSpan rows have retrieval_method='cross_encoder'."""
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run

    _, _, source_doc, run_version, claims, _ = await make_packet_with_claim(db_session)
    mock_bm25_functions(monkeypatch)
    mock_reranker(monkeypatch)

    spans = await retrieve_evidence_for_run(str(run_version.id), db_session)
    for span in spans:
        assert span.retrieval_method == "cross_encoder"


@pytest.mark.asyncio
async def test_pipeline_sets_retrieval_rank_1_based(db_session, monkeypatch):
    """First span has retrieval_rank=1, second has retrieval_rank=2."""
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run

    _, _, source_doc, run_version, claims, _ = await make_packet_with_claim(db_session)
    mock_bm25_functions(monkeypatch)
    mock_reranker(monkeypatch)

    spans = await retrieve_evidence_for_run(str(run_version.id), db_session)
    if len(spans) >= 2:
        assert spans[0].retrieval_rank == 1
        assert spans[1].retrieval_rank == 2


@pytest.mark.asyncio
async def test_pipeline_scoped_to_cited_document(db_session, monkeypatch):
    """Spans come only from the specifically cited source document, not all docs."""
    from evidenceengine.models.document import SourceDocument
    from evidenceengine.models.claim import Claim, CitationAnchor
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run

    # Create two source documents — claim only cites doc_A
    packet, report_doc, doc_a, run_version, claims, _ = await make_packet_with_claim(db_session)

    doc_b = SourceDocument(
        packet_id=packet.id,
        is_report=False,
        filename="source_b.pdf",
        file_path="/uploads/source_b.pdf",
        file_type="pdf",
        parsed_content={"blocks": [{"text": "Completely unrelated text from doc B.", "block_type": "paragraph", "position": {"page": 1, "paragraph": 1, "char_start": 0, "char_end": 36, "section_header": None}}]},
    )
    db_session.add(doc_b)
    await db_session.flush()

    # Track which docs load_or_build_index is called with
    called_doc_ids = []

    def fake_load_or_build(spans, index_dir):
        called_doc_ids.append(index_dir)
        return MagicMock()

    def fake_query_index(retriever, query, k=10):
        texts = ["Carbon emissions increased."]
        return texts, [5.0]

    async def fake_rerank(query, passages):
        return [{"corpus_id": 0, "score": 8.5, "text": passages[0]}] if passages else []

    monkeypatch.setattr("evidenceengine.retrieval.pipeline.load_or_build_index", fake_load_or_build)
    monkeypatch.setattr("evidenceengine.retrieval.pipeline.query_index", fake_query_index)
    monkeypatch.setattr("evidenceengine.retrieval.pipeline.rerank", fake_rerank)

    spans = await retrieve_evidence_for_run(str(run_version.id), db_session)

    # Only doc_a should be indexed (claim only has anchor pointing to doc_a)
    assert any(str(doc_a.id) in path for path in called_doc_ids)
    assert not any(str(doc_b.id) in path for path in called_doc_ids)


@pytest.mark.asyncio
async def test_unresolvable_claim_skipped(db_session, monkeypatch):
    """Claim with unresolvable anchor → no EvidenceSpan rows; still counts in recall denominator."""
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run

    _, _, _, run_version, claims, _ = await make_packet_with_claim(
        db_session, resolution_status="unresolvable"
    )
    mock_bm25_functions(monkeypatch)
    mock_reranker(monkeypatch)

    spans = await retrieve_evidence_for_run(str(run_version.id), db_session)
    assert spans == []


@pytest.mark.asyncio
async def test_multiple_anchors_per_claim(db_session, monkeypatch):
    """Claim with 2 resolved anchors → EvidenceSpan rows from BOTH documents."""
    from evidenceengine.models.document import SourceDocument
    from evidenceengine.models.claim import CitationAnchor
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run

    packet, report_doc, doc_a, run_version, claims, _ = await make_packet_with_claim(db_session)

    # Create second source doc
    doc_b = SourceDocument(
        packet_id=packet.id,
        is_report=False,
        filename="source_b.pdf",
        file_path="/uploads/source_b.pdf",
        file_type="pdf",
        parsed_content={"blocks": [{"text": "Different evidence text in second source document.", "block_type": "paragraph", "position": {"page": 1, "paragraph": 1, "char_start": 0, "char_end": 49, "section_header": None}}]},
    )
    db_session.add(doc_b)

    # Add second anchor to same claim pointing to doc_b
    claim = claims[0]
    anchor_b = CitationAnchor(
        claim_id=claim.id,
        raw_marker="[2]",
        citation_style="numeric",
        target_document_id=doc_b.id,
        resolution_status="resolved",
    )
    db_session.add(anchor_b)
    await db_session.flush()

    # Track which source_document_ids spans come from
    source_doc_ids_used = []

    def fake_load_or_build(spans, index_dir):
        source_doc_ids_used.append(index_dir)
        return MagicMock()

    def fake_query_index(retriever, query, k=10):
        return ["Some evidence text here."], [5.0]

    async def fake_rerank(query, passages):
        return [{"corpus_id": 0, "score": 8.5, "text": passages[0]}] if passages else []

    monkeypatch.setattr("evidenceengine.retrieval.pipeline.load_or_build_index", fake_load_or_build)
    monkeypatch.setattr("evidenceengine.retrieval.pipeline.query_index", fake_query_index)
    monkeypatch.setattr("evidenceengine.retrieval.pipeline.rerank", fake_rerank)

    spans = await retrieve_evidence_for_run(str(run_version.id), db_session)

    # Should have indexed both documents
    assert len(source_doc_ids_used) == 2
    assert any(str(doc_a.id) in d for d in source_doc_ids_used)
    assert any(str(doc_b.id) in d for d in source_doc_ids_used)


@pytest.mark.asyncio
async def test_index_cache_reused_within_run(db_session, monkeypatch):
    """Two claims citing same source_document_id → load_or_build_index called only once."""
    from evidenceengine.models.claim import Claim, CitationAnchor
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run

    packet, report_doc, source_doc, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=0
    )

    # Add two claims both citing the same source_doc
    for _ in range(2):
        claim = Claim(
            packet_id=packet.id,
            source_document_id=report_doc.id,
            run_version_id=run_version.id,
            claim_text="Carbon emissions rose by 12% [1].",
            status="completed",
        )
        db_session.add(claim)
        await db_session.flush()
        anchor = CitationAnchor(
            claim_id=claim.id,
            raw_marker="[1]",
            citation_style="numeric",
            target_document_id=source_doc.id,
            resolution_status="resolved",
        )
        db_session.add(anchor)

    await db_session.flush()

    build_call_count = []

    def fake_load_or_build(spans, index_dir):
        build_call_count.append(index_dir)
        return MagicMock()

    def fake_query_index(retriever, query, k=10):
        return ["Evidence text here."], [5.0]

    async def fake_rerank(query, passages):
        return [{"corpus_id": 0, "score": 8.5, "text": passages[0]}] if passages else []

    monkeypatch.setattr("evidenceengine.retrieval.pipeline.load_or_build_index", fake_load_or_build)
    monkeypatch.setattr("evidenceengine.retrieval.pipeline.query_index", fake_query_index)
    monkeypatch.setattr("evidenceengine.retrieval.pipeline.rerank", fake_rerank)

    await retrieve_evidence_for_run(str(run_version.id), db_session)

    # Index built only once — cache reused for second claim
    assert len(build_call_count) == 1


@pytest.mark.asyncio
async def test_recall_logger_writes_to_run_version(db_session, monkeypatch):
    """After pipeline completes, RunVersion.pipeline_config['retrieval_metrics'] exists."""
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run
    from evidenceengine.models.run import RunVersion as RunVersionModel
    from sqlalchemy import select

    _, _, _, run_version, _, _ = await make_packet_with_claim(db_session)
    mock_bm25_functions(monkeypatch)
    mock_reranker(monkeypatch)

    await retrieve_evidence_for_run(str(run_version.id), db_session)

    # Reload run_version from DB to get updated pipeline_config
    result = await db_session.execute(
        select(RunVersionModel).where(RunVersionModel.id == run_version.id)
    )
    updated_run = result.scalar_one()
    assert updated_run.pipeline_config is not None
    assert "retrieval_metrics" in updated_run.pipeline_config
    metrics = updated_run.pipeline_config["retrieval_metrics"]
    assert "recall_at_k" in metrics
    assert isinstance(metrics["recall_at_k"], float)


@pytest.mark.asyncio
async def test_recall_at_k_correct_with_unresolvable(db_session, monkeypatch):
    """3 total claims, 1 unresolvable (no spans), 2 retrieved → recall_at_k = 2/3."""
    from evidenceengine.models.claim import Claim, CitationAnchor
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run
    from evidenceengine.models.run import RunVersion as RunVersionModel
    from sqlalchemy import select

    # Make packet with no claims initially
    packet, report_doc, source_doc, run_version, _, _ = await make_packet_with_claim(
        db_session, n_claims=0
    )

    # Add 2 resolved claims
    for _ in range(2):
        claim = Claim(
            packet_id=packet.id,
            source_document_id=report_doc.id,
            run_version_id=run_version.id,
            claim_text="Carbon rose by 12% [1].",
            status="completed",
        )
        db_session.add(claim)
        await db_session.flush()
        anchor = CitationAnchor(
            claim_id=claim.id,
            raw_marker="[1]",
            citation_style="numeric",
            target_document_id=source_doc.id,
            resolution_status="resolved",
        )
        db_session.add(anchor)

    # Add 1 unresolvable claim
    unresolvable_claim = Claim(
        packet_id=packet.id,
        source_document_id=report_doc.id,
        run_version_id=run_version.id,
        claim_text="Unresolvable claim without citation.",
        status="completed",
    )
    db_session.add(unresolvable_claim)
    await db_session.flush()
    unresolvable_anchor = CitationAnchor(
        claim_id=unresolvable_claim.id,
        raw_marker="[99]",
        citation_style="numeric",
        target_document_id=None,
        resolution_status="unresolvable",
    )
    db_session.add(unresolvable_anchor)
    await db_session.flush()

    mock_bm25_functions(monkeypatch)
    mock_reranker(monkeypatch)

    await retrieve_evidence_for_run(str(run_version.id), db_session)

    result = await db_session.execute(
        select(RunVersionModel).where(RunVersionModel.id == run_version.id)
    )
    updated_run = result.scalar_one()
    metrics = updated_run.pipeline_config["retrieval_metrics"]
    assert metrics["claim_count"] == 3
    assert metrics["retrieved_count"] == 2
    # recall_at_k = 2/3 ≈ 0.667
    assert abs(metrics["recall_at_k"] - 2 / 3) < 0.01


@pytest.mark.asyncio
async def test_recall_logger_claim_count_includes_unresolvable(db_session, monkeypatch):
    """recall log claim_count includes unresolvable claims."""
    from evidenceengine.retrieval.pipeline import retrieve_evidence_for_run
    from evidenceengine.models.run import RunVersion as RunVersionModel
    from sqlalchemy import select

    _, _, _, run_version, _, _ = await make_packet_with_claim(
        db_session, resolution_status="unresolvable"
    )
    mock_bm25_functions(monkeypatch)
    mock_reranker(monkeypatch)

    await retrieve_evidence_for_run(str(run_version.id), db_session)

    result = await db_session.execute(
        select(RunVersionModel).where(RunVersionModel.id == run_version.id)
    )
    updated_run = result.scalar_one()
    assert updated_run.pipeline_config["retrieval_metrics"]["claim_count"] == 1


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
async def test_retrieval_endpoint_returns_202(async_client, db_session, monkeypatch):
    """POST /api/packets/{id}/retrieve with valid packet → 202 with run_version_id and evidence_count."""
    from evidenceengine.models.document import DocumentPacket, SourceDocument
    from evidenceengine.models.run import RunVersion
    from evidenceengine.models.claim import Claim, CitationAnchor

    # Set up test data
    packet = DocumentPacket(
        status="queued",
        report_filename="report.pdf",
        report_file_path="/uploads/report.pdf",
    )
    db_session.add(packet)
    await db_session.flush()

    report_doc = SourceDocument(
        packet_id=packet.id,
        is_report=True,
        filename="report.pdf",
        file_path="/uploads/report.pdf",
        file_type="pdf",
        parsed_content={"blocks": []},
    )
    db_session.add(report_doc)

    source_doc = SourceDocument(
        packet_id=packet.id,
        is_report=False,
        filename="source.pdf",
        file_path="/uploads/source.pdf",
        file_type="pdf",
        parsed_content=SAMPLE_PARSED_CONTENT,
    )
    db_session.add(source_doc)

    run_version = RunVersion(
        packet_id=packet.id,
        status="extraction_complete",
        pipeline_config={},
    )
    db_session.add(run_version)
    await db_session.flush()

    claim = Claim(
        packet_id=packet.id,
        source_document_id=report_doc.id,
        run_version_id=run_version.id,
        claim_text="Carbon rose by 12% [1].",
        status="completed",
    )
    db_session.add(claim)
    await db_session.flush()

    anchor = CitationAnchor(
        claim_id=claim.id,
        raw_marker="[1]",
        citation_style="numeric",
        target_document_id=source_doc.id,
        resolution_status="resolved",
    )
    db_session.add(anchor)
    await db_session.flush()

    # Mock retrieval functions
    monkeypatch.setattr("evidenceengine.retrieval.pipeline.load_or_build_index", lambda spans, path: MagicMock())
    monkeypatch.setattr("evidenceengine.retrieval.pipeline.query_index", lambda r, q, k=10: (["Carbon emissions text."], [5.0]))

    async def fake_rerank(query, passages):
        return [{"corpus_id": 0, "score": 8.5, "text": passages[0]}] if passages else []

    monkeypatch.setattr("evidenceengine.retrieval.pipeline.rerank", fake_rerank)

    response = await async_client.post(f"/api/packets/{packet.id}/retrieve")
    assert response.status_code == 202
    data = response.json()
    assert "run_version_id" in data
    assert "evidence_count" in data


@pytest.mark.asyncio
async def test_retrieval_endpoint_404_for_unknown_packet(async_client):
    """POST /api/packets/{unknown_id}/retrieve → 404."""
    unknown_id = uuid.uuid4()
    response = await async_client.post(f"/api/packets/{unknown_id}/retrieve")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_retrieval_endpoint_requires_completed_run(async_client, db_session):
    """Packet with no RunVersion → 422."""
    from evidenceengine.models.document import DocumentPacket

    packet = DocumentPacket(
        status="queued",
        report_filename="report.pdf",
        report_file_path="/uploads/report.pdf",
    )
    db_session.add(packet)
    await db_session.flush()

    response = await async_client.post(f"/api/packets/{packet.id}/retrieve")
    assert response.status_code == 422
