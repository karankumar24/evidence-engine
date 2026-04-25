"""Document packet API endpoints.

POST /api/packets   — Upload report + sources, validate, parse, store
GET  /api/packets/:id — Retrieve packet status and document details
"""

import uuid as uuid_module
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from evidenceengine.api.dependencies import get_db, get_file_store
from evidenceengine.core.config import settings
from evidenceengine.ingestion.validation import (
    serialize_parsed_document as _serialize_parsed_document,
    validate_and_parse_file,
    validate_file_type,
)
from evidenceengine.models.document import DocumentPacket, SourceDocument
from evidenceengine.schemas.common import ErrorDetail, ErrorResponse
from evidenceengine.schemas.document import PacketResponse
from evidenceengine.storage.file_store import FileStore

router = APIRouter(prefix="/api/packets", tags=["packets"])

_MAX_FILE_BYTES = settings.max_file_size_mb * 1024 * 1024


@router.post(
    "/",
    response_model=PacketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document packet",
)
async def create_packet(
    report: Annotated[UploadFile, File(description="Report document (PDF or DOCX)")],
    sources: Annotated[
        list[UploadFile],
        File(description="Source documents (PDF or DOCX, optional)"),
    ] = [],
    db: AsyncSession = Depends(get_db),
    file_store: FileStore = Depends(get_file_store),
) -> PacketResponse:
    """Upload a report and optional source documents.

    Validates file types, parses all files, and stores results atomically.
    If any file fails validation or parsing, the entire request is rejected with details.
    """
    packet_id = str(uuid_module.uuid4())
    all_files = [(report, "report")] + [(s, "source") for s in sources]
    errors: list[ErrorDetail] = []

    # Step 1: Read all file contents and check sizes
    file_contents: list[tuple[UploadFile, str, bytes]] = []
    for upload_file, role in all_files:
        content = await upload_file.read()
        if len(content) > _MAX_FILE_BYTES:
            errors.append(
                ErrorDetail(
                    file=upload_file.filename or "unknown",
                    role=role,
                    error=f"File exceeds maximum size of {settings.max_file_size_mb}MB",
                )
            )
        else:
            file_contents.append((upload_file, role, content))

    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                message="One or more files exceed the size limit",
                errors=errors,
            ).model_dump(),
        )

    # Step 2: Validate file types for ALL files first
    for upload_file, role, content in file_contents:
        filename = upload_file.filename or "unknown"
        try:
            validate_file_type(content)
        except ValueError as e:
            errors.append(ErrorDetail(file=filename, role=role, error=str(e)))

    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                message="One or more files have unsupported file types",
                errors=errors,
            ).model_dump(),
        )

    # Step 3: Save all raw files to disk
    saved_paths: dict[str, str] = {}
    for upload_file, role, content in file_contents:
        filename = upload_file.filename or "unknown"
        file_path = file_store.save(packet_id, filename, content)
        saved_paths[filename] = file_path

    # Step 4: Parse all files (parse-then-accept pattern)
    parsed_results: dict[str, object] = {}
    for upload_file, role, content in file_contents:
        filename = upload_file.filename or "unknown"
        file_path = saved_paths[filename]
        try:
            parsed = validate_and_parse_file(file_path, content)
            parsed_results[filename] = parsed
        except ValueError as e:
            errors.append(ErrorDetail(file=filename, role=role, error=str(e)))

    if errors:
        # Clean up saved files — atomicity guarantee
        file_store.delete_packet(packet_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                message="One or more files failed to parse",
                errors=errors,
            ).model_dump(),
        )

    # Step 5: Create database records
    report_filename = report.filename or "report"
    report_path = saved_paths.get(report_filename, "")

    packet = DocumentPacket(
        id=uuid_module.UUID(packet_id),
        status="completed",
        report_filename=report_filename,
        report_file_path=report_path,
    )
    db.add(packet)
    await db.flush()  # Get the packet ID assigned

    for upload_file, role, content in file_contents:
        filename = upload_file.filename or "unknown"
        file_path = saved_paths[filename]
        parsed = parsed_results[filename]
        is_report = role == "report"

        # Serialize parsed blocks to JSON-serialisable dicts
        parsed_content = _serialize_parsed_document(parsed)

        source_doc = SourceDocument(
            packet_id=packet.id,
            is_report=is_report,
            filename=filename,
            file_path=file_path,
            file_type=_detect_file_type_str(filename),
            file_size_bytes=len(content),
            parsed_content=parsed_content,
            raw_text=(parsed.raw_text or "").replace("\x00", "") or None,
            markdown_text=(parsed.markdown_text or "").replace("\x00", "") or None,
            total_pages=parsed.total_pages,
            parse_status="completed",
        )
        db.add(source_doc)

    await db.flush()
    await db.refresh(packet)

    # Reload with eager source_documents for the response
    result = await db.execute(
        select(DocumentPacket)
        .options(selectinload(DocumentPacket.source_documents))
        .where(DocumentPacket.id == packet.id)
    )
    packet = result.scalar_one()

    return PacketResponse.model_validate(packet)


@router.get(
    "/{packet_id}",
    response_model=PacketResponse,
    summary="Get packet status and documents",
)
async def get_packet(
    packet_id: uuid_module.UUID,
    db: AsyncSession = Depends(get_db),
) -> PacketResponse:
    """Retrieve a document packet by ID with all source documents."""
    result = await db.execute(
        select(DocumentPacket)
        .options(selectinload(DocumentPacket.source_documents))
        .where(DocumentPacket.id == packet_id)
    )
    packet = result.scalar_one_or_none()

    if packet is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Packet {packet_id} not found",
        )

    return PacketResponse.model_validate(packet)


def _detect_file_type_str(filename: str) -> str:
    """Detect file type string from filename extension."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        return "pdf"
    if ext in ("docx", "doc"):
        return "docx"
    return ext
