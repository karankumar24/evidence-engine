"""Verdict classification logic: classify_claim() and apply_confidence_threshold().

Uses the synchronous OpenAI beta structured output endpoint (beta.chat.completions.parse)
inside asyncio.to_thread so that httpx network I/O does not block the uvicorn event loop.
"""

import asyncio
import logging

from evidenceengine.classification.schemas import VerdictClassificationResponse
from evidenceengine.core.config import settings

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

## Instructions

Always reason step-by-step before assigning a label. Complete the reasoning field fully before choosing verdict_type."""


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
            f"[Evidence {i + 1} (relevance: {span['relevance_score']:.2f})]\n{span['span_text']}"
            for i, span in enumerate(evidence_spans)
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
        from evidenceengine.llm.fallback import sync_call_with_fallback  # noqa: PLC0415
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
