"""EvidenceEngine — provenance-aware document verification system.

Core data models are importable directly from this package:

    from evidenceengine import (
        DocumentPacket, SourceDocument, Claim, CitationAnchor,
        EvidenceSpan, Verdict, ReviewDecision, RunVersion,
    )

These imports do not pull in FastAPI, ML pipeline, or dashboard dependencies.
Install with `pip install evidenceengine[core]` for the minimal dependency set.
"""

__version__ = "0.1.0"

from evidenceengine.models import (
    CitationAnchor,
    Claim,
    DocumentPacket,
    EvidenceSpan,
    ReviewDecision,
    RunVersion,
    SourceDocument,
    Verdict,
)

__all__ = [
    "__version__",
    "DocumentPacket",
    "SourceDocument",
    "Claim",
    "CitationAnchor",
    "EvidenceSpan",
    "Verdict",
    "ReviewDecision",
    "RunVersion",
]
