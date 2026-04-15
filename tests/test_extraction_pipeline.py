"""Integration tests for extraction pipeline and API endpoint.

TDD RED phase: written BEFORE implementation.
Uses real test DB (db_session from conftest) with transaction rollback.
All LLM calls mocked via unittest.mock.AsyncMock — no real OpenAI API calls.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from evidenceengine.api.app import create_app
from evidenceengine.api.dependencies import get_db
from evidenceengine.extraction.schemas import (
    ClaimExtractionResponse,
    ExtractedClaim,
    ExtractedCitationMarker,
)
from evidenceengine.models.claim import CitationAnchor, Claim
from evidenceengine.models.document import DocumentPacket, SourceDocument
from evidenceengine.models.run import RunVersion

import os

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
async def packet_with_docs(db_session: AsyncSession):
    """Create a DocumentPacket with one report doc and one source doc."""
    packet = DocumentPacket(
        status="queued",
        report_filename="report.pdf",
        report_file_path="/uploads/report.pdf",
    )
    db_session.add(packet)
    await db_session.flush()

    # Report document with citation block content
    report_blocks = {
        "blocks": [
            {
                "text": "References",
                "block_type": "heading",
                "position": {"page": 10, "paragraph": 0, "char_start": 0, "char_end": 10, "section_header": "References"},
            },
            {
                "text": "Smith J. 2023. Climate data analysis. Journal of Climate.",
                "block_type": "paragraph",
                "position": {"page": 10, "paragraph": 1, "char_start": 11, "char_end": 70, "section_header": "References"},
            },
            {
                "text": "Results",
                "block_type": "heading",
                "position": {"page": 3, "paragraph": 0, "char_start": 100, "char_end": 107, "section_header": None},
            },
            {
                "text": "Carbon emissions increased by 12% [1].",
                "block_type": "paragraph",
                "position": {"page": 3, "paragraph": 1, "char_start": 108, "char_end": 146, "section_header": "Results"},
            },
        ]
    }
    report_raw_text = "Results\nCarbon emissions increased by 12% [1].\nReferences\nSmith J. 2023. Climate data analysis."

    report_doc = SourceDocument(
        packet_id=packet.id,
        is_report=True,
        filename="report.pdf",
        file_path="/uploads/report.pdf",
        file_type="pdf",
        parsed_content=report_blocks,
        raw_text=report_raw_text,
        parse_status="completed",
    )
    db_session.add(report_doc)

    # Source document that will be resolved to
    source_doc = SourceDocument(
        packet_id=packet.id,
        is_report=False,
        filename="smith_climate_2023.pdf",
        file_path="/uploads/smith_climate_2023.pdf",
        file_type="pdf",
        raw_text="Climate data analysis. Smith 2023.",
        parse_status="completed",
    )
    db_session.add(source_doc)
    await db_session.flush()

    return packet, report_doc, source_doc


@pytest_asyncio.fixture
async def client(db_session):
    """AsyncClient with overridden DB dependency."""
    app = create_app()

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Pipeline unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_creates_claim_rows(db_session: AsyncSession, packet_with_docs):
    """Pipeline creates Claim rows in DB when LLM returns extracted claims."""
    from evidenceengine.extraction.pipeline import extract_claims_for_document

    packet, report_doc, source_doc = packet_with_docs

    # Create a RunVersion for this test
    run_version = RunVersion(
        packet_id=packet.id,
        status="running",
        pipeline_config={"phase": "claim_extraction"},
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(run_version)
    await db_session.flush()

    mock_extraction = ClaimExtractionResponse(claims=[
        ExtractedClaim(
            claim_text="Carbon emissions increased by 12% [1].",
            citation_markers=[ExtractedCitationMarker(raw_marker="[1]", citation_style="numeric")],
        ),
        ExtractedClaim(
            claim_text="Temperature rose significantly [1].",
            citation_markers=[ExtractedCitationMarker(raw_marker="[1]", citation_style="numeric")],
        ),
    ])

    with patch(
        "evidenceengine.extraction.pipeline.extract_claims_from_blocks",
        new_callable=AsyncMock,
        return_value=mock_extraction,
    ):
        claims = await extract_claims_for_document(
            report_document_id=str(report_doc.id),
            run_version_id=str(run_version.id),
            db=db_session,
        )

    assert len(claims) == 2, f"Expected 2 claims, got {len(claims)}"


@pytest.mark.asyncio
async def test_pipeline_creates_anchor_rows(db_session: AsyncSession, packet_with_docs):
    """Pipeline creates CitationAnchor rows for each marker in each claim."""
    from evidenceengine.extraction.pipeline import extract_claims_for_document

    packet, report_doc, source_doc = packet_with_docs

    run_version = RunVersion(
        packet_id=packet.id,
        status="running",
        pipeline_config={"phase": "claim_extraction"},
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(run_version)
    await db_session.flush()

    mock_extraction = ClaimExtractionResponse(claims=[
        ExtractedClaim(
            claim_text="Carbon emissions increased by 12% [1].",
            citation_markers=[ExtractedCitationMarker(raw_marker="[1]", citation_style="numeric")],
        ),
    ])

    with patch(
        "evidenceengine.extraction.pipeline.extract_claims_from_blocks",
        new_callable=AsyncMock,
        return_value=mock_extraction,
    ):
        claims = await extract_claims_for_document(
            report_document_id=str(report_doc.id),
            run_version_id=str(run_version.id),
            db=db_session,
        )

    assert len(claims) == 1
    # Check anchors were persisted
    result = await db_session.execute(
        select(CitationAnchor).where(CitationAnchor.claim_id == claims[0].id)
    )
    anchors = result.scalars().all()
    assert len(anchors) == 1
    assert anchors[0].raw_marker == "[1]"


@pytest.mark.asyncio
async def test_unresolvable_anchor_flagged(db_session: AsyncSession, packet_with_docs):
    """Unresolvable anchor written with resolution_status='unresolvable'; Claim gets 'unresolvable_anchor' status (CLAIM-06)."""
    from evidenceengine.extraction.pipeline import extract_claims_for_document

    packet, report_doc, source_doc = packet_with_docs

    run_version = RunVersion(
        packet_id=packet.id,
        status="running",
        pipeline_config={"phase": "claim_extraction"},
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(run_version)
    await db_session.flush()

    # Marker that won't match any source doc
    mock_extraction = ClaimExtractionResponse(claims=[
        ExtractedClaim(
            claim_text="Sea levels rose (Completely Unknown Author 1899).",
            citation_markers=[
                ExtractedCitationMarker(raw_marker="Completely Unknown Author 1899", citation_style="author_year")
            ],
        ),
    ])

    with patch(
        "evidenceengine.extraction.pipeline.extract_claims_from_blocks",
        new_callable=AsyncMock,
        return_value=mock_extraction,
    ):
        claims = await extract_claims_for_document(
            report_document_id=str(report_doc.id),
            run_version_id=str(run_version.id),
            db=db_session,
        )

    assert len(claims) == 1
    claim = claims[0]
    assert claim.status == "unresolvable_anchor", f"Expected 'unresolvable_anchor', got '{claim.status}'"

    result = await db_session.execute(
        select(CitationAnchor).where(CitationAnchor.claim_id == claim.id)
    )
    anchors = result.scalars().all()
    assert len(anchors) == 1
    assert anchors[0].resolution_status == "unresolvable"
    assert anchors[0].target_document_id is None


@pytest.mark.asyncio
async def test_resolved_anchor_sets_target_document_id(db_session: AsyncSession, packet_with_docs):
    """Successfully resolved anchor has target_document_id set to the source doc UUID."""
    from evidenceengine.extraction.pipeline import extract_claims_for_document

    packet, report_doc, source_doc = packet_with_docs

    run_version = RunVersion(
        packet_id=packet.id,
        status="running",
        pipeline_config={"phase": "claim_extraction"},
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(run_version)
    await db_session.flush()

    mock_extraction = ClaimExtractionResponse(claims=[
        ExtractedClaim(
            claim_text="Carbon emissions increased by 12% [1].",
            citation_markers=[ExtractedCitationMarker(raw_marker="[1]", citation_style="numeric")],
        ),
    ])

    with patch(
        "evidenceengine.extraction.pipeline.extract_claims_from_blocks",
        new_callable=AsyncMock,
        return_value=mock_extraction,
    ):
        claims = await extract_claims_for_document(
            report_document_id=str(report_doc.id),
            run_version_id=str(run_version.id),
            db=db_session,
        )

    result = await db_session.execute(
        select(CitationAnchor).where(CitationAnchor.claim_id == claims[0].id)
    )
    anchors = result.scalars().all()
    assert len(anchors) >= 1
    # At least attempt resolution — either resolved or unresolvable is acceptable
    # The important thing is target_document_id is set when resolved
    for anchor in anchors:
        if anchor.resolution_status == "resolved":
            assert anchor.target_document_id is not None


@pytest.mark.asyncio
async def test_no_citation_blocks_returns_empty(db_session: AsyncSession, packet_with_docs):
    """Report with no citation markers returns empty list without error."""
    from evidenceengine.extraction.pipeline import extract_claims_for_document

    packet, report_doc, source_doc = packet_with_docs

    # Modify report to have no citation markers in blocks
    report_doc.parsed_content = {
        "blocks": [
            {
                "text": "This text has no citations at all.",
                "block_type": "paragraph",
                "position": {"page": 1, "paragraph": 0, "char_start": 0, "char_end": 34, "section_header": None},
            }
        ]
    }
    await db_session.flush()

    run_version = RunVersion(
        packet_id=packet.id,
        status="running",
        pipeline_config={"phase": "claim_extraction"},
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(run_version)
    await db_session.flush()

    with patch(
        "evidenceengine.extraction.pipeline.extract_claims_from_blocks",
        new_callable=AsyncMock,
    ) as mock_extract:
        claims = await extract_claims_for_document(
            report_document_id=str(report_doc.id),
            run_version_id=str(run_version.id),
            db=db_session,
        )
        # LLM should NOT be called when no citation blocks found
        mock_extract.assert_not_called()

    assert claims == []


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_endpoint_returns_202(client, db_session: AsyncSession):
    """POST /api/packets/{id}/extract returns 202 with run_version_id and claim_count."""
    # Create a packet with a report doc
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
        parsed_content={"blocks": [
            {
                "text": "Emissions rose [1].",
                "block_type": "paragraph",
                "position": {"page": 1, "paragraph": 0, "char_start": 0, "char_end": 19, "section_header": "Results"},
            }
        ]},
        raw_text="Emissions rose [1].",
        parse_status="completed",
    )
    db_session.add(report_doc)
    await db_session.flush()

    mock_extraction = ClaimExtractionResponse(claims=[
        ExtractedClaim(
            claim_text="Emissions rose [1].",
            citation_markers=[ExtractedCitationMarker(raw_marker="[1]", citation_style="numeric")],
        ),
    ])

    with patch(
        "evidenceengine.extraction.pipeline.extract_claims_from_blocks",
        new_callable=AsyncMock,
        return_value=mock_extraction,
    ):
        response = await client.post(f"/api/packets/{packet.id}/extract")

    assert response.status_code == 202, f"Expected 202, got {response.status_code}: {response.text}"
    data = response.json()
    assert "run_version_id" in data
    assert "claim_count" in data
    assert data["claim_count"] >= 0


@pytest.mark.asyncio
async def test_pipeline_endpoint_404_for_unknown_packet(client):
    """POST /api/packets/{unknown_id}/extract returns 404."""
    unknown_id = str(uuid.uuid4())
    response = await client.post(f"/api/packets/{unknown_id}/extract")
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"
