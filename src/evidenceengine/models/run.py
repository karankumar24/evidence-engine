"""RunVersion ORM model — tracks pipeline execution runs."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evidenceengine.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from evidenceengine.models.document import DocumentPacket


class RunVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Records a single pipeline run for a packet — config, status, and timing."""

    __tablename__ = "run_versions"
    __table_args__ = (
        Index("ix_run_versions_created_at", "created_at"),
    )

    packet_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_packets.id", name="fk_run_versions_packet_id_document_packets"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(30), default="queued", nullable=False)
    pipeline_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_versions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    packet: Mapped["DocumentPacket"] = relationship(
        "DocumentPacket",
        back_populates="run_versions",
    )
