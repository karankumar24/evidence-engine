"""Claim and CitationAnchor ORM models — schema stubs for Phase 2."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evidenceengine.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from evidenceengine.models.evidence import EvidenceSpan
    from evidenceengine.models.review import ReviewDecision
    from evidenceengine.models.verdict import Verdict


class Claim(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A factual claim extracted from the report — extracted in Phase 2."""

    __tablename__ = "claims"

    packet_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_packets.id", name="fk_claims_packet_id_document_packets"),
        nullable=False,
        index=True,
    )
    source_document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "source_documents.id",
            name="fk_claims_source_document_id_source_documents",
        ),
        nullable=False,
        index=True,
    )
    run_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run_versions.id", name="fk_claims_run_version_id_run_versions"),
        nullable=False,
        index=True,
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    section_header: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paragraph_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)

    # Relationships
    citation_anchors: Mapped[list["CitationAnchor"]] = relationship(
        "CitationAnchor",
        back_populates="claim",
        cascade="all, delete-orphan",
    )
    evidence_spans: Mapped[list["EvidenceSpan"]] = relationship(
        "EvidenceSpan",
        back_populates="claim",
        cascade="all, delete-orphan",
    )
    verdicts: Mapped[list["Verdict"]] = relationship(
        "Verdict",
        back_populates="claim",
        cascade="all, delete-orphan",
    )
    review_decisions: Mapped[list["ReviewDecision"]] = relationship(
        "ReviewDecision",
        back_populates="claim",
        cascade="all, delete-orphan",
    )


class CitationAnchor(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A citation marker found in a claim, resolved to a source document."""

    __tablename__ = "citation_anchors"

    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claims.id", name="fk_citation_anchors_claim_id_claims"),
        nullable=False,
        index=True,
    )
    raw_marker: Mapped[str] = mapped_column(Text, nullable=False)  # e.g. "[1]", "(Smith 2023)"
    citation_style: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # "numeric", "author_year", "footnote"
    target_document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "source_documents.id",
            name="fk_citation_anchors_target_document_id_source_documents",
        ),
        nullable=True,
    )
    resolution_status: Mapped[str] = mapped_column(
        String(30), default="pending", nullable=False
    )  # pending/resolved/unresolvable

    # Relationships
    claim: Mapped["Claim"] = relationship("Claim", back_populates="citation_anchors")
