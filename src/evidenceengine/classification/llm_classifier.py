"""LLM-based verdict classifier (v1.2.9 Plan 02-01 lift-and-shift).

This module holds the ACTUAL implementation of `classify_claim`, moved here
byte-identically from `classifier.py` so the new `ClassifierBackend` Protocol
can treat LLM and NLI classification as peers.

The original `classifier.py` is now a thin backward-compat shim that re-exports
`classify_claim`, `apply_confidence_threshold`, and `asyncio` so every existing
test patch target (`evidenceengine.classification.classifier.asyncio.to_thread`
etc.) keeps resolving. See plan 02-01 Pitfall 3 for rationale.

Behavior contract: for any given (claim_text, evidence_spans) input, calling
`LLMClassifier().classify(...)` MUST produce the same VerdictClassificationResponse
as calling `classify_claim(...)` directly. This parity is the rollback guarantee
for the NLI-primary migration.
"""

from __future__ import annotations

import asyncio
import logging

from evidenceengine.classification.schemas import VerdictClassificationResponse
from evidenceengine.core.config import settings
from evidenceengine.llm.fallback import sync_call_with_fallback

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are an expert evidence classifier for document verification. Your task is to determine whether evidence from a cited source supports, contradicts, or is insufficient to verify a claim from a report.

## Input
You will receive:
- CLAIM: A specific factual statement extracted from a report
- EVIDENCE SPANS: Numbered passages retrieved from the cited source document

## Classification Rules

Use exactly one of these four labels:

**supported**
Evidence EXPLICITLY and COMPLETELY confirms ALL elements of the claim.
- Every factual element in the claim must be directly addressed by the evidence.
- If any element is unconfirmed or only partially addressed → use insufficient_support instead.
- Example: Claim "Revenue grew 12% YoY" + Evidence "Annual revenue increased 12% compared to prior year" → supported

**contradicted**
Evidence ACTIVELY states the OPPOSITE of the claim.
- The contradiction must be direct and explicit, not merely a failure to confirm.
- Numeric example: Claim says "10% increase", evidence says "5% decrease" → contradicted, not insufficient_support
- Example: Claim "No layoffs occurred" + Evidence "The company announced 500 redundancies" → contradicted

**insufficient_support**
DEFAULT for uncertain cases. Evidence is absent, partial, or does not directly address the claim.
- Use this when you are not sure whether to use supported or contradicted.
- Evidence addresses a related but different topic → insufficient_support
- Evidence confirms some but not all elements → insufficient_support
- No relevant evidence spans available → insufficient_support

**needs_review**
Evidence internally conflicts (different spans say opposite things) OR the claim itself is genuinely ambiguous (cannot be evaluated as written).
- Do NOT use needs_review as a substitute for insufficient_support.
- Reserve for genuine ambiguity or internal contradiction in the evidence.

## Confidence Calibration

Score below 0.7 for uncertain cases. Score above 0.85 only when evidence is unambiguous.
A score below 0.7 will route to needs_review automatically — this is a feature, not a failure.
Be honest about uncertainty rather than artificially inflating confidence.

## Self-verification caveat (when evidence comes from the SAME document as the claim)

Each evidence span is tagged with its source: `SAME DOCUMENT AS CLAIM` means the span came
from the very same document the claim appears in; `EXTERNAL SOURCE` means the span came
from a different (cited) document. Treat these tags as ground truth — not as hints.

- If ALL evidence spans are tagged `SAME DOCUMENT AS CLAIM`, you are in self-verification
  mode. "Self-reference" is not proof. Only return **supported** when a DIFFERENT
  paragraph, in different words, independently restates the fact. If the only evidence
  is a near-duplicate of the claim's own wording, return **insufficient_support** — the
  document is not self-validating.
- If at least one span is `EXTERNAL SOURCE`, rely primarily on the external spans for
  a **supported** verdict; SAME-DOC spans can corroborate but never carry the verdict
  alone.

## Instructions

Always reason step-by-step before assigning a label. Complete the reasoning field fully before choosing verdict_type."""


def _format_evidence_block(i: int, span: dict) -> str:
    """Render one evidence span with optional source-attribution tags.

    The evidence dict may carry:
      - span_text (required)
      - relevance_score (required)
      - source_filename (optional): the filename of the document the span
        came from. Rendered as `source: <filename>` when provided.
      - is_same_doc_as_claim (optional, bool): True when the span came from
        the same document the claim appears in. Rendered as an explicit
        `SAME DOCUMENT AS CLAIM` vs `EXTERNAL SOURCE` tag so the classifier
        can reason about self-verify mode deterministically instead of
        having to infer from context.

    Backward-compatible: callers that don't populate source metadata get the
    same compact format as before.
    """
    parts = [f"Evidence {i + 1}"]
    src = span.get("source_filename")
    same_doc = span.get("is_same_doc_as_claim")
    if src:
        parts.append(f"source: {src}")
    if same_doc is True:
        parts.append("SAME DOCUMENT AS CLAIM")
    elif same_doc is False:
        parts.append("EXTERNAL SOURCE")
    parts.append(f"relevance: {span['relevance_score']:.2f}")
    header = "[" + " | ".join(parts) + "]"
    return f"{header}\n{span['span_text']}"


async def classify_claim(
    claim_text: str,
    evidence_spans: list[dict],
) -> VerdictClassificationResponse:
    """Classify a claim against retrieved evidence spans using structured LLM output.

    Runs the synchronous OpenAI client in a thread pool (via asyncio.to_thread)
    to prevent httpx from blocking the uvicorn event loop on macOS.

    Args:
        claim_text: The factual claim to be verified.
        evidence_spans: List of dicts with keys span_text, relevance_score, rank.

    Returns:
        VerdictClassificationResponse with reasoning, verdict_type, and confidence_score.
        Returns needs_review with confidence_score=0.0 if the model issues a refusal.
    """
    if evidence_spans:
        evidence_blocks = "\n\n".join(
            _format_evidence_block(i, span) for i, span in enumerate(evidence_spans)
        )
    else:
        evidence_blocks = "No evidence spans available."

    user_message = (
        f"CLAIM:\n{claim_text}\n\n"
        f"EVIDENCE SPANS:\n{evidence_blocks}"
    )

    # Run synchronous OpenAI client in a thread to avoid blocking the event loop.
    # AsyncOpenAI + httpx on macOS blocks the asyncio selector for the full
    # request duration; the sync client in a thread is safe and non-blocking.
    def _sync_request(msg: str = user_message) -> object:
        chain = settings.model_fallback_chain or [settings.classification_model]
        timeout = (
            settings.llm_fallback_timeout_seconds
            if len(chain) > 1
            else settings.llm_request_timeout_seconds
        )
        return sync_call_with_fallback(
            model_chain=chain,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ],
            response_format=VerdictClassificationResponse,
            timeout=timeout,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
            gemini_api_key=settings.gemini_api_key,
            groq_api_key=settings.groq_api_key,
        )

    message = await asyncio.to_thread(_sync_request)

    if message.refusal:
        return VerdictClassificationResponse(
            reasoning="Model refused to classify this claim.",
            verdict_type="needs_review",
            confidence_score=0.0,
        )

    if message.parsed is None:
        logger.error(
            "LLM returned null parsed response for classification — "
            "structured output may have failed. Returning needs_review fallback."
        )
        return VerdictClassificationResponse(
            reasoning="Classification failed: structured output parsing returned null.",
            verdict_type="needs_review",
            confidence_score=0.0,
        )

    return message.parsed


def apply_confidence_threshold(
    result: VerdictClassificationResponse,
    threshold: float,
) -> VerdictClassificationResponse:
    """Override verdict_type to needs_review when confidence falls below threshold.

    Args:
        result: The VerdictClassificationResponse from classify_claim().
        threshold: Minimum confidence required to retain the verdict label.

    Returns:
        The original result unchanged if confidence >= threshold or verdict is already needs_review.
        A new VerdictClassificationResponse with verdict_type="needs_review" and the original
        confidence_score preserved if confidence < threshold.
    """
    if result.verdict_type == "needs_review":
        return result

    if result.confidence_score < threshold:
        return VerdictClassificationResponse(
            reasoning=(
                f"Low confidence ({result.confidence_score:.2f} < threshold {threshold:.2f}) "
                f"— routed to needs_review. Original reasoning: {result.reasoning}"
            ),
            verdict_type="needs_review",
            confidence_score=result.confidence_score,
        )

    return result


class LLMClassifier:
    """ClassifierBackend Protocol implementation delegating to classify_claim().

    Byte-identical behavior to the pre-Plan-02-01 `classify_claim` path — this
    class exists solely so the Protocol dispatch in `backend.get_backend()` has
    something concrete to return on the `llm_primary` branch.
    """

    async def classify(
        self,
        claim_text: str,
        evidence_spans: list[dict],
    ) -> VerdictClassificationResponse:
        return await classify_claim(claim_text, evidence_spans)
