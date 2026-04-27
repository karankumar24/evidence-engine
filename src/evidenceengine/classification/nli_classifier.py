"""NLI-primary classifier — Phase 02 Plan 02.

Lazy, thread-safe singleton wrapping the cross-encoder/nli-deberta-v3-small model
(``NLI_MODEL_NAME`` below) loaded onto MPS (or CPU fallback). Per-span NLI inference (premise=evidence, hypothesis=claim)
is aggregated by element-wise MAX across spans, then mapped to the 4-way
VerdictClassificationResponse schema via settings-driven thresholds.

Design notes (see .planning/phases/02-nli-primary-classifier/02-RESEARCH.md):

- Premise/hypothesis direction is locked: premise=evidence_span, hypothesis=claim.
  Flipping this tanks FEVER/SciFact accuracy (Pitfall 5).
- Softmax is computed on fp32 logits (``logits.float()``) unconditionally to
  dodge the MPS fp16 NaN class (Pitfall 1).
- Tokenizer outputs are moved to the model device before the forward pass
  (Pitfall 2) — skipping this raises
  ``RuntimeError: Placeholder storage has not been allocated on MPS device!``.
- Singleton load failures fail-open: ``_load_failed`` flips, ``_ensure_loaded``
  returns ``(None, None, None)`` forever after, and ``classify`` downgrades to
  ``needs_review`` with confidence 0.0. Never raises to callers.
- ``transformers`` / ``torch`` are imported lazily inside the functions that
  need them so the module is cheap to import from tests that never touch a
  real model.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Literal

from evidenceengine.classification.schemas import VerdictClassificationResponse

logger = logging.getLogger(__name__)

# Fine-tuned on SciFact (epoch 1): 53.7% → 78.0% on SciFact dev (+24.3pp).
# Local path for production (volume-mounted at /data/models/scifact-nli).
# Falls back to base model string if local path absent (dev/CI environments).
import os as _os
NLI_MODEL_NAME = (
    _os.environ.get("NLI_MODEL_PATH")
    or ("/data/models/scifact-nli" if _os.path.exists("/data/models/scifact-nli/config.json") else "cross-encoder/nli-deberta-v3-small")
)

# Actual id2label for this model checkpoint (contradiction-first).
NLI_LABELS: tuple[str, str, str] = ("contradiction", "entailment", "neutral")

# Module-level singleton state. Tests reset these via an autouse fixture.
_model = None
_tokenizer = None
_device = None
_lock = threading.Lock()
_load_failed = False


def _ensure_loaded():
    """Double-check-locked lazy load of tokenizer + model.

    Returns ``(model, tokenizer, device)`` on success, ``(None, None, None)``
    on failure or once a prior load has failed. Never raises.
    """
    global _model, _tokenizer, _device, _load_failed

    # Fast path (no lock) — already loaded or already failed.
    if _model is not None:
        return _model, _tokenizer, _device
    if _load_failed:
        return None, None, None

    with _lock:
        # Re-check inside the lock.
        if _model is not None:
            return _model, _tokenizer, _device
        if _load_failed:
            return None, None, None
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            _device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            # Resolve model path at load time so NLI_MODEL_PATH from .env works.
            from evidenceengine.core.config import settings as _settings  # noqa: PLC0415
            _effective_model = (
                _settings.nli_model_path
                or NLI_MODEL_NAME
            )
            logger.info("Loading NLI model %s onto %s", _effective_model, _device)
            _tokenizer = AutoTokenizer.from_pretrained(_effective_model)
            _model = AutoModelForSequenceClassification.from_pretrained(_effective_model)
            _model.to(_device)
            _model.eval()
            logger.info("NLI model ready on %s", _device)
            return _model, _tokenizer, _device
        except Exception as exc:  # noqa: BLE001 — fail-open by contract
            logger.warning(
                "NLI-primary disabled: could not load %s (%s)",
                NLI_MODEL_NAME,
                exc,
            )
            _load_failed = True
            return None, None, None


def nli_probs_for_pair(
    premise: str, hypothesis: str
) -> tuple[float, float, float] | None:
    """Run one NLI inference.

    Returns ``(p_entail, p_neutral, p_contra)`` or None if the singleton
    failed to load. Premise is the evidence span; hypothesis is the claim.
    """
    import torch  # noqa: PLC0415

    model, tokenizer, device = _ensure_loaded()
    if model is None:
        return None

    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    # Move every tokenizer tensor to the model device (Pitfall 2).
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    # fp32 softmax unconditionally — MPS fp16 can NaN (Pitfall 1).
    probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
    # cross-encoder/nli-deberta-v3-small label order: 0=contradiction, 1=entailment, 2=neutral.
    # Return canonical (p_entail, p_neutral, p_contra) order expected by callers.
    return float(probs[1]), float(probs[2]), float(probs[0])


VerdictLiteral = Literal[
    "supported", "contradicted", "insufficient_support", "needs_review"
]


def nli_probs_to_verdict(
    p_entail: float, p_neutral: float, p_contra: float
) -> tuple[VerdictLiteral, float]:
    """Map 3-way NLI probabilities to a 4-way verdict + confidence.

    Rules (locked by ROADMAP + Pattern 2 in RESEARCH.md):
    - max_p < nli_min_confidence_for_verdict (0.50) → needs_review
    - p_entail >= nli_entailment_supported_threshold (0.80) → supported
    - p_contra >= nli_contradiction_contradicted_threshold (0.80) → contradicted
    - else → insufficient_support (confidence = max of the three)

    WHY max_p uses all three classes (not just entail+contra): the NLI model
    cannot distinguish "insufficient_support" (neutral-dominant, evidence exists
    but is unrelated) from "needs_review" (genuinely uncertain) — both produce
    neutral-dominant outputs. Using max_p (including neutral) means the 0.50
    threshold is never hit in practice (the model is always confident about
    SOMETHING). As a result, needs_review is only assigned by rule-based paths
    in pipeline.py (unresolvable_anchor, model load failure), not by this function.
    This is the CORRECT behavior for the current NLI-only pipeline: neutral
    dominant = insufficient_support, not needs_review.

    Confidence for supported is p_entail; for contradicted is p_contra;
    for needs_review and insufficient_support it is the max across classes.
    """
    # Local import so tests can monkeypatch settings attributes cleanly.
    from evidenceengine.core.config import settings  # noqa: PLC0415

    max_p = max(p_entail, p_neutral, p_contra)
    if max_p < settings.nli_min_confidence_for_verdict:
        return "needs_review", max_p
    if p_entail >= settings.nli_entailment_supported_threshold:
        return "supported", p_entail
    if p_contra >= settings.nli_contradiction_contradicted_threshold:
        return "contradicted", p_contra
    return "insufficient_support", max_p


def aggregate_nli(
    per_span_probs: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Element-wise MAX across spans.

    Empty input returns ``(0.0, 0.0, 0.0)`` — zero signal, which the verdict
    mapper interprets as needs_review at 0.0 confidence (max_p < min threshold).
    """
    if not per_span_probs:
        return 0.0, 0.0, 0.0
    return (
        max(p[0] for p in per_span_probs),
        max(p[1] for p in per_span_probs),
        max(p[2] for p in per_span_probs),
    )


class NLIClassifier:
    """ClassifierBackend implementation backed by cross-encoder/nli-deberta-v3-small NLI.

    Per-span inference with MAX aggregation. Leaves ``explanation=None`` on
    the response — Plan 03's explanation generator fills that field after the
    verdict is decided.
    """

    async def classify(
        self,
        claim_text: str,
        evidence_spans: list[dict],
    ) -> VerdictClassificationResponse:
        if not evidence_spans:
            return VerdictClassificationResponse(
                reasoning="No evidence spans retrieved.",
                verdict_type="insufficient_support",
                confidence_score=0.0,
            )

        def _run() -> list[tuple[float, float, float] | None]:
            # Per-span inference happens inside ONE to_thread call — we don't
            # want to spawn N worker threads for N spans.
            return [
                nli_probs_for_pair(s["span_text"], claim_text) for s in evidence_spans
            ]

        per_span = await asyncio.to_thread(_run)
        valid = [p for p in per_span if p is not None]
        if not valid:
            # Every inference returned None → singleton failed to load.
            # Fail-open to needs_review, never silently pass.
            return VerdictClassificationResponse(
                reasoning="NLI model unavailable — routed to needs_review.",
                verdict_type="needs_review",
                confidence_score=0.0,
            )

        agg = aggregate_nli(valid)
        verdict, conf = nli_probs_to_verdict(*agg)
        reasoning = (
            f"[NLI-primary] per-span MAX: entail={agg[0]:.3f} "
            f"neutral={agg[1]:.3f} contra={agg[2]:.3f} → {verdict}"
        )
        return VerdictClassificationResponse(
            reasoning=reasoning,
            verdict_type=verdict,
            confidence_score=conf,
        )
