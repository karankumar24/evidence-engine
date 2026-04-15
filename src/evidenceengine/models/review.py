"""ReviewDecision ORM model — schema stub for Phase 6."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evidenceengine.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from evidenceengine.models.claim import Claim
    from evidenceengine.models.verdict import Verdict


class ReviewDecision(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A human reviewer's decision on a verdict."""

    __tablename__ = "review_decisions"

    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claims.id", name="fk_review_decisions_claim_id_claims"),
        nullable=False,
        index=True,
    )
    verdict_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("verdicts.id", name="fk_review_decisions_verdict_id_verdicts"),
        nullable=False,
        index=True,
    )
    reviewer_id: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # Simple string for now, no auth in v1
    action: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # approve/reject/mark_insufficient
    verdict_at_decision: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # snapshot of verdict_type when decision was made
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    claim: Mapped["Claim"] = relationship("Claim")
    verdict: Mapped["Verdict"] = relationship("Verdict", back_populates="review_decisions")
