"""Verdict and VerdictEvidence ORM models — schema stubs for Phase 4."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evidenceengine.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from evidenceengine.models.claim import Claim
    from evidenceengine.models.evidence import EvidenceSpan
    from evidenceengine.models.review import ReviewDecision


class Verdict(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The verdict for a claim: supported, contradicted, insufficient, or needs review."""

    __tablename__ = "verdicts"
    __table_args__ = (
        UniqueConstraint(
            "claim_id",
            "run_version_id",
            name="uq_verdicts_claim_id_run_version_id",
        ),
    )

    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claims.id", name="fk_verdicts_claim_id_claims"),
        nullable=False,
        index=True,
    )
    run_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run_versions.id", name="fk_verdicts_run_version_id_run_versions"),
        nullable=False,
        index=True,
    )
    verdict_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # supported/contradicted/insufficient_support/needs_review
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Relationships
    claim: Mapped["Claim"] = relationship("Claim", back_populates="verdicts")
    verdict_evidence: Mapped[list["VerdictEvidence"]] = relationship(
        "VerdictEvidence",
        back_populates="verdict",
        cascade="all, delete-orphan",
    )
    review_decisions: Mapped[list["ReviewDecision"]] = relationship(
        "ReviewDecision",
        back_populates="verdict",
        cascade="all, delete-orphan",
    )


class VerdictEvidence(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Junction table linking a Verdict to the EvidenceSpans that support it."""

    __tablename__ = "verdict_evidence"

    verdict_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("verdicts.id", name="fk_verdict_evidence_verdict_id_verdicts"),
        nullable=False,
        index=True,
    )
    evidence_span_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "evidence_spans.id",
            name="fk_verdict_evidence_evidence_span_id_evidence_spans",
        ),
        nullable=False,
        index=True,
    )
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    verdict: Mapped["Verdict"] = relationship("Verdict", back_populates="verdict_evidence")
    evidence_span: Mapped["EvidenceSpan"] = relationship(
        "EvidenceSpan", back_populates="verdict_evidence"
    )
