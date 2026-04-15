"""Integration tests for POST /api/packets and GET /api/packets/:id endpoints."""

import os
import tempfile
import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from evidenceengine.api.app import create_app
from evidenceengine.api.dependencies import get_db, get_file_store
from evidenceengine.models.base import Base
from evidenceengine.storage.file_store import FileStore

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# Use the test database (same PostgreSQL, isolated via transaction rollback)
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine",
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest_asyncio.fixture
async def test_engine():
    """Function-scoped engine to avoid event loop issues."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine):
    """Yield a session that rolls back after each test."""
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def temp_upload_dir(tmp_path):
    """Temporary directory for file uploads."""
    return str(tmp_path / "uploads")


@pytest_asyncio.fixture
async def client(db_session, temp_upload_dir):
    """AsyncClient with overridden DB and file store dependencies."""
    app = create_app()

    async def override_get_db():
        yield db_session

    def override_get_file_store():
        return FileStore(temp_upload_dir)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_file_store] = override_get_file_store

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# --------------------------------------------------------------------------- #
# Test 1: POST /api/packets with valid PDF report returns 201
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_upload_pdf_report_returns_201(client):
    simple_pdf = os.path.join(FIXTURES, "simple_report.pdf")
    with open(simple_pdf, "rb") as f:
        response = await client.post(
            "/api/packets/",
            files={"report": ("simple_report.pdf", f, "application/pdf")},
        )
    assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
    data = response.json()
    assert "id" in data
    assert data["status"] == "completed"
    assert data["report_filename"] == "simple_report.pdf"
    assert isinstance(data["source_documents"], list)
    assert len(data["source_documents"]) >= 1  # report is included


# --------------------------------------------------------------------------- #
# Test 2: POST /api/packets with valid DOCX report returns 201
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_upload_docx_report_returns_201(client):
    simple_docx = os.path.join(FIXTURES, "simple_report.docx")
    with open(simple_docx, "rb") as f:
        response = await client.post(
            "/api/packets/",
            files={"report": ("simple_report.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
    assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "completed"


# --------------------------------------------------------------------------- #
# Test 3: POST /api/packets with invalid file type returns 422
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_upload_invalid_file_type_returns_422(client):
    # Create a fake PDF that's actually plain text
    fake_content = b"This is not a real PDF file\n" * 20
    response = await client.post(
        "/api/packets/",
        files={"report": ("fake.pdf", fake_content, "application/pdf")},
    )
    assert response.status_code == 422, f"Expected 422, got {response.status_code}: {response.text}"
    data = response.json()
    # Should have error details
    detail = data.get("detail", {})
    assert "errors" in detail or isinstance(detail, list)


# --------------------------------------------------------------------------- #
# Test 4: POST /api/packets with report only (no sources) returns 201
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_upload_report_only_returns_201(client):
    """Sources are optional (INGEST-02: 'one or more')."""
    simple_pdf = os.path.join(FIXTURES, "simple_report.pdf")
    with open(simple_pdf, "rb") as f:
        response = await client.post(
            "/api/packets/",
            files={"report": ("simple_report.pdf", f, "application/pdf")},
        )
    assert response.status_code == 201


# --------------------------------------------------------------------------- #
# Test 5: POST /api/packets with report + source returns 201, both docs present
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_upload_report_and_sources(client):
    simple_pdf = os.path.join(FIXTURES, "simple_report.pdf")
    simple_docx = os.path.join(FIXTURES, "simple_report.docx")
    with open(simple_pdf, "rb") as report_f, open(simple_docx, "rb") as source_f:
        response = await client.post(
            "/api/packets/",
            files=[
                ("report", ("report.pdf", report_f, "application/pdf")),
                ("sources", ("source.docx", source_f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ],
        )
    assert response.status_code == 201
    data = response.json()
    assert len(data["source_documents"]) == 2  # report + 1 source
    # Verify one is marked as report
    is_report_flags = [d["is_report"] for d in data["source_documents"]]
    assert True in is_report_flags


# --------------------------------------------------------------------------- #
# Test 6: GET /api/packets/:id returns packet with all documents
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_get_packet_returns_packet(client):
    # First upload
    simple_pdf = os.path.join(FIXTURES, "simple_report.pdf")
    with open(simple_pdf, "rb") as f:
        upload_response = await client.post(
            "/api/packets/",
            files={"report": ("simple_report.pdf", f, "application/pdf")},
        )
    assert upload_response.status_code == 201
    packet_id = upload_response.json()["id"]

    # Then fetch
    get_response = await client.get(f"/api/packets/{packet_id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["id"] == packet_id
    assert data["status"] == "completed"
    assert len(data["source_documents"]) >= 1


# --------------------------------------------------------------------------- #
# Test 7: GET /api/packets/:id for non-existent ID returns 404
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_get_packet_not_found(client):
    fake_id = str(uuid.uuid4())
    response = await client.get(f"/api/packets/{fake_id}")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Test 8: Uploaded files are saved to disk
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_files_saved_to_disk(client, temp_upload_dir):
    simple_pdf = os.path.join(FIXTURES, "simple_report.pdf")
    with open(simple_pdf, "rb") as f:
        response = await client.post(
            "/api/packets/",
            files={"report": ("simple_report.pdf", f, "application/pdf")},
        )
    assert response.status_code == 201
    packet_id = response.json()["id"]

    # Verify file exists on disk
    expected_path = os.path.join(temp_upload_dir, packet_id, "simple_report.pdf")
    assert os.path.exists(expected_path), f"Expected file at {expected_path}"


# --------------------------------------------------------------------------- #
# Test 9: SourceDocument has raw_text and parsed_content populated
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_source_document_has_parsed_content(client, db_session):
    from sqlalchemy.future import select

    from evidenceengine.models.document import SourceDocument

    simple_pdf = os.path.join(FIXTURES, "simple_report.pdf")
    with open(simple_pdf, "rb") as f:
        response = await client.post(
            "/api/packets/",
            files={"report": ("simple_report.pdf", f, "application/pdf")},
        )
    assert response.status_code == 201
    packet_id = response.json()["id"]

    # Check DB record has raw_text and parsed_content
    from sqlalchemy import text as sql_text
    from evidenceengine.models.document import DocumentPacket
    import uuid as uuid_mod

    result = await db_session.execute(
        select(SourceDocument).where(
            SourceDocument.packet_id == uuid_mod.UUID(packet_id)
        )
    )
    docs = result.scalars().all()
    assert len(docs) >= 1
    for doc in docs:
        assert doc.raw_text is not None and len(doc.raw_text) > 0, "raw_text must be populated"
        assert doc.parsed_content is not None, "parsed_content must be populated"
        # Verify offset integrity: spot-check first block
        if doc.parsed_content and doc.parsed_content.get("blocks"):
            first_block = doc.parsed_content["blocks"][0]
            pos = first_block["position"]
            expected_text = first_block["text"]
            extracted = doc.raw_text[pos["char_start"]:pos["char_end"]]
            assert extracted == expected_text, (
                f"Stored offset integrity failed: raw_text[{pos['char_start']}:{pos['char_end']}]={extracted!r} "
                f"!= block.text={expected_text!r}"
            )
