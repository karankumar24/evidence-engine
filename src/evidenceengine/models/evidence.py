"""EvidenceSpan ORM model — schema stub for Phase 3."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evidenceengine.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from evidenceengine.models.claim import Claim
    from evidenceengine.models.verdict import VerdictEvidence


class EvidenceSpan(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A span of text from a source document used as evidence for a claim."""

    __tablename__ = "evidence_spans"

    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claims.id", name="fk_evidence_spans_claim_id_claims"),
        nullable=False,
        index=True,
    )
    source_document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "source_documents.id",
            name="fk_evidence_spans_source_document_id_source_documents",
        ),
        nullable=False,
        index=True,
    )
    run_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run_versions.id", name="fk_evidence_spans_run_version_id_run_versions"),
        nullable=False,
        index=True,
    )
    span_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paragraph_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    section_header: Mapped[str | None] = mapped_column(Text, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_method: Mapped[str] = mapped_column(
        String(30), nullable=False, default="bm25"
    )  # "bm25", "cross_encoder", etc.
    retrieval_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    claim: Mapped["Claim"] = relationship("Claim", back_populates="evidence_spans")
    verdict_evidence: Mapped[list["VerdictEvidence"]] = relationship(
        "VerdictEvidence",
        back_populates="evidence_span",
    )
