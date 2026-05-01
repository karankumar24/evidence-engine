"""DocumentPacket and SourceDocument ORM models."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evidenceengine.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from evidenceengine.models.run import RunVersion


class DocumentPacket(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Top-level container for a verification request: one report + N source documents."""

    __tablename__ = "document_packets"

    status: Mapped[str] = mapped_column(String(50), default="queued", nullable=False)
    report_filename: Mapped[str] = mapped_column(Text, nullable=False)
    report_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    # Ownership: the session that uploaded this packet. Nullable so legacy rows
    # migrate cleanly; NULL acts as "unclaimed" — invisible to every new
    # visitor since no cookie will ever match. Indexed for dashboard queries.
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # Demo flag: when True the packet is visible to EVERY visitor regardless
    # of session_id. Used to ship a canned example so the dashboard isn't
    # empty on first-visit.
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    source_documents: Mapped[list["SourceDocument"]] = relationship(
        "SourceDocument",
        back_populates="packet",
        cascade="all, delete-orphan",
    )
    run_versions: Mapped[list["RunVersion"]] = relationship(
        "RunVersion",
        back_populates="packet",
        cascade="all, delete-orphan",
    )


class SourceDocument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A single uploaded file (report or source) within a DocumentPacket."""

    __tablename__ = "source_documents"

    packet_id: Mapped[uuid.UUID] = mapped_column(
        nullable=False,
        index=True,
    )
    is_report: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)  # "pdf" or "docx"
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # Parsed content
    parsed_content: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    markdown_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parse_status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    parse_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Structured paper metadata extracted from PDF at parse time.
    # Used for citation anchor resolution: matching "[1]" to uploaded papers
    # by title+author instead of filename (filename matching fails for generic names).
    # Schema: {"title": str, "authors": [str, ...], "year": str}
    paper_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    packet: Mapped["DocumentPacket"] = relationship(
        "DocumentPacket",
        back_populates="source_documents",
        foreign_keys=[packet_id],
    )

    __table_args__ = (
        __import__("sqlalchemy").ForeignKeyConstraint(
            ["packet_id"],
            ["document_packets.id"],
            name="fk_source_documents_packet_id_document_packets",
        ),
    )
