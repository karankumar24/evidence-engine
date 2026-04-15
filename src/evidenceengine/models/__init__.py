"""SQLAlchemy ORM models — all models must be imported here so Alembic autogenerate can detect them."""

from evidenceengine.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from evidenceengine.models.claim import CitationAnchor, Claim
from evidenceengine.models.document import DocumentPacket, SourceDocument
from evidenceengine.models.evidence import EvidenceSpan
from evidenceengine.models.review import ReviewDecision
from evidenceengine.models.run import RunVersion
from evidenceengine.models.verdict import Verdict, VerdictEvidence

__all__ = [
    "Base",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "DocumentPacket",
    "SourceDocument",
    "Claim",
    "CitationAnchor",
    "EvidenceSpan",
    "Verdict",
    "VerdictEvidence",
    "ReviewDecision",
    "RunVersion",
]
