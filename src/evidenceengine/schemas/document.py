"""Pydantic schemas for DocumentPacket and SourceDocument API responses."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SourceDocumentResponse(BaseModel):
    """Response schema for a single source document."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: str
    is_report: bool
    parse_status: str
    total_pages: int | None
    parse_error: str | None
    created_at: datetime


class PacketResponse(BaseModel):
    """Response schema for a document packet (report + sources)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: str
    report_filename: str
    source_documents: list[SourceDocumentResponse]
    created_at: datetime
    updated_at: datetime


class PacketListResponse(BaseModel):
    """Response schema for a list of packets."""

    packets: list[PacketResponse]
    total: int
