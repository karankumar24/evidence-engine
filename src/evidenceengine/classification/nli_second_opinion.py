"""Deterministic NLI second-opinion for classifier verdicts.

Runs a locally-hosted natural language inference (NLI) model AFTER the LLM
verdict to catch the two failure modes that matter most for the trust model:

  1. LLM returned SUPPORTED but NLI strongly says CONTRADICTION → force
     needs_review. The pipeline's worst-case failure class (false support)
     gets one extra guard.

  2. LLM returned CONTRADICTED but NLI strongly says ENTAILMENT → force
     needs_review. Someone's wrong; don't publish either verdict.

NLI never escalates a verdict (never turns needs_review → supported, never
creates a contradicted verdict where the LLM didn't). It can only downgrade
to needs_review. This keeps the trust direction one-way: toward human review.

Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (184M params, ~740MB on
disk; downloaded on first use to ~/.cache/huggingface/). FEVER-tuned so it
handles fact-verification NLI better than plain MNLI. Per research, this
family is the community-standard zero-shot NLI stack that runs on CPU in
seconds per inference.

The model is loaded lazily, behind a lock, on first call. Settings flag
`nli_second_opinion_enabled` (default True) makes it trivially disable-able
if a deployment doesn't want the extra latency.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)


NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Thresholds: empirically, the FEVER-tuned DeBERTa-v3-base assigns >0.95
# probability to clear entailment and contradiction cases and leaves
# semantically ambiguous pairs in neutral territory. We want second-opinion
# only to fire when the model is CONFIDENTLY disagreeing — otherwise noise
# dominates and we'd downgrade real supported verdicts.
CONTRADICTION_TRIGGER_PROB = 0.85  # "NLI thinks it's a contradiction"
ENTAILMENT_TRIGGER_PROB = 0.85     # "NLI thinks it's supported"


NLILabel = Literal["entailment", "neutral", "contradiction"]


@dataclass(frozen=True)
class NLIJudgment:
    """Structured NLI result, ready for downstream decision logic."""

    label: NLILabel
    probability: float  # probability of `label` under the NLI model


_pipeline = None
_pipeline_lock = threading.Lock()
_load_failed = False


def _get_pipeline():
    """Lazy-load the NLI pipeline; thread-safe; fails open (returns None).

    If transformers/torch isn't installed we return None and the caller
    treats second-opinion as disabled. This keeps the classifier from
    breaking when we ship the core without ML dependencies.
    """
    global _pipeline, _load_failed
    if _pipeline is not None:
        return _pipeline
    if _load_failed:
        return None
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        if _load_failed:
            return None
        try:
            from transformers import pipeline  # noqa: PLC0415
            logger.info("Loading NLI model: %s", NLI_MODEL_NAME)
            _pipeline = pipeline(
                "text-classification",
                model=NLI_MODEL_NAME,
                top_k=None,
                device=-1,  # CPU; never assume GPU
            )
            logger.info("NLI model loaded")
            return _pipeline
        except Exception as exc:  # noqa: BLE001 — fail-open is the design
            logger.warning(
                "NLI second-opinion disabled: could not load %s (%s: %s). "
                "This is expected when running without transformers/torch.",
                NLI_MODEL_NAME, type(exc).__name__, exc,
            )
            _load_failed = True
            return None


def nli_judgment(claim_text: str, evidence_texts: list[str]) -> NLIJudgment | None:
    """Run NLI over (claim, concatenated evidence) and return the top label.

    Treats the evidence as the premise and the claim as the hypothesis, which
    is the standard FEVER framing (does the retrieved evidence entail or
    contradict the claim?).

    Returns None when the NLI model couldn't load or no evidence was given —
    the caller's contract is "None means no opinion, don't override."
    """
    if not evidence_texts:
        return None
    pipe = _get_pipeline()
    if pipe is None:
        return None

    # Concatenate evidence so NLI sees everything at once. Truncate at 4000
    # chars to stay within the 512-token model window while preserving most
    # pipelines' top-k retrieval output. Join on newlines — the model was
    # trained on newline-separated passages for FEVER.
    premise = "\n".join(evidence_texts)[:4000]
    try:
        result = pipe({"text": premise, "text_pair": claim_text})
    except Exception as exc:  # noqa: BLE001
        logger.warning("NLI inference failed: %s", exc)
        return None

    # pipeline with top_k=None returns a list of dicts; find the highest-prob label.
    top = max(result, key=lambda x: x["score"])
    label = top["label"].lower()
    if label not in ("entailment", "neutral", "contradiction"):
        logger.warning("Unexpected NLI label %r — treating as neutral", label)
        label = "neutral"
    return NLIJudgment(label=label, probability=float(top["score"]))  # type: ignore[arg-type]


def should_force_review(
    llm_verdict_type: str,
    nli: NLIJudgment | None,
) -> tuple[bool, str]:
    """Decide whether the NLI judgment warrants forcing needs_review.

    Args:
        llm_verdict_type: "supported" | "contradicted" | "insufficient_support" | "needs_review"
        nli: the NLI second opinion, or None (no opinion)

    Returns:
        (force_needs_review, reason_suffix). reason_suffix is empty when we
        don't force anything; otherwise a short string suitable for appending
        to the verdict's reasoning field for auditability.
    """
    if nli is None:
        return False, ""
    # Case 1: LLM said supported, NLI confidently says contradiction.
    if (
        llm_verdict_type == "supported"
        and nli.label == "contradiction"
        and nli.probability >= CONTRADICTION_TRIGGER_PROB
    ):
        return True, (
            f"[NLI second-opinion: high-confidence contradiction "
            f"(p={nli.probability:.2f}) — forced to needs_review]"
        )
    # Case 2: LLM said contradicted, NLI confidently says entailment.
    if (
        llm_verdict_type == "contradicted"
        and nli.label == "entailment"
        and nli.probability >= ENTAILMENT_TRIGGER_PROB
    ):
        return True, (
            f"[NLI second-opinion: high-confidence entailment "
            f"(p={nli.probability:.2f}) contradicts LLM — forced to needs_review]"
        )
    return False, ""
