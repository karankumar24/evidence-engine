"""Backward-compat shim — real implementation lives in llm_classifier.

Kept so existing tests patching ``evidenceengine.classification.classifier.*``
(specifically ``asyncio.to_thread`` in 22 locations across
``test_classifier.py``, ``test_classification_pipeline.py``, and
``test_adversarial_pipeline.py``) keep working.

DO NOT put logic here. If you're tempted to add a function, put it in
``llm_classifier.py`` (for LLM path) or ``nli_classifier.py`` (Plan 02) and
re-export it from here only if existing tests patch it by attribute.

Rationale: Plan 02-01 lifted ``classify_claim`` + ``apply_confidence_threshold``
into ``llm_classifier.py`` so ``LLMClassifier`` (Protocol impl) can live next
to its implementation. This file preserves the historical module namespace
so downstream tests — which patch ``classifier.asyncio.to_thread`` at module
scope — never notice the refactor.
"""

# Re-export asyncio so tests that patch ``classifier.asyncio.to_thread`` at
# module scope still resolve. This IS load-bearing — removing it breaks 22
# existing test patch targets.
import asyncio  # noqa: F401

# Re-export implementation symbols so existing `from
# evidenceengine.classification.classifier import classify_claim,
# apply_confidence_threshold` statements (e.g. classification/pipeline.py:18)
# keep resolving.
from evidenceengine.classification.llm_classifier import (  # noqa: F401
    SYSTEM_PROMPT,
    PROMPT_VERSION,
    apply_confidence_threshold,
    classify_claim,
)
