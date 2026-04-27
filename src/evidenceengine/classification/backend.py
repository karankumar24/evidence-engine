"""ClassifierBackend Protocol + get_backend() dispatcher.

Production runs ``nli_primary`` only. The historical LLM-primary rollback path
was deleted in the Phase A simplification refactor — ``get_backend`` now
returns ``NLIClassifier`` unconditionally.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evidenceengine.classification.schemas import VerdictClassificationResponse


@runtime_checkable
class ClassifierBackend(Protocol):
    """Common interface for verdict classifiers.

    Same ``(claim_text, evidence_spans)`` input contract callers have always
    used. Returns a ``VerdictClassificationResponse``.
    """

    async def classify(
        self,
        claim_text: str,
        evidence_spans: list[dict],
    ) -> VerdictClassificationResponse: ...


def get_backend() -> ClassifierBackend:
    """Return the NLI classifier backend.

    Lazy import keeps this module cheap to load and avoids circular deps.
    """
    from evidenceengine.classification.nli_classifier import NLIClassifier  # noqa: PLC0415
    return NLIClassifier()
