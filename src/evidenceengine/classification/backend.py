"""ClassifierBackend Protocol + get_backend() dispatcher.

Plan 02-01 foundation. Plan 02 lands NLIClassifier; Plan 03 wires dispatch
into pipeline.py. In this plan, get_backend() is defined but not yet called
from pipeline code — it's provided so Plan 02/03 can slot in without editing
this file.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evidenceengine.classification.schemas import VerdictClassificationResponse


@runtime_checkable
class ClassifierBackend(Protocol):
    """Common interface for verdict classifiers (LLM-based or NLI-based).

    The same (claim_text, evidence_spans) input contract that ``classify_claim``
    has always used. Returns a VerdictClassificationResponse — may set
    ``explanation`` to None (Plan 03 wires the explanation generator).
    """

    async def classify(
        self,
        claim_text: str,
        evidence_spans: list[dict],
    ) -> VerdictClassificationResponse: ...


def get_backend() -> ClassifierBackend:
    """Return the classifier backend selected by settings.classifier_backend.

    Lazy imports avoid circular dependencies AND keep Plan 01 green standalone:
    ``NLIClassifier`` is only imported on the ``nli_primary`` branch, which
    means tests that monkeypatch ``settings.classifier_backend = "llm_primary"``
    never trigger the NLI import even before Plan 02 lands.
    """
    # Lazy import so this module stays cheap to load.
    from evidenceengine.core.config import settings

    if settings.classifier_backend == "nli_primary":
        # Imported lazily — Plan 02 provides this module. Until it lands,
        # callers explicitly setting nli_primary will get an ImportError,
        # which is the correct signal that the feature isn't wired yet.
        from evidenceengine.classification.nli_classifier import NLIClassifier  # noqa: PLC0415
        return NLIClassifier()

    from evidenceengine.classification.llm_classifier import LLMClassifier  # noqa: PLC0415
    return LLMClassifier()
