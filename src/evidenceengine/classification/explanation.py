"""Post-verdict explanation generator (Phase 02 Plan 03).

Always runs AFTER the verdict is decided. Failure logs a warning and
returns ``None`` — NEVER blocks the verdict. Uses a DIFFERENT system prompt
from the LLM verdict classifier so failures don't cascade across the two
LLM call paths (tiebreaker vs explanation).

Contract (CLF-06):
    - ``generate_explanation(claim, spans, verdict) -> str | None``
    - On any exception (chain exhaustion, timeout, structured-output
      parse failure) returns ``None`` and logs a warning at WARNING.
    - The caller (pipeline.py) merges the returned string into
      ``VerdictClassificationResponse.explanation`` and concatenates
      into the persisted ``Verdict.reasoning`` for audit.

Why this file is separate from ``llm_classifier.py`` (Pitfall 4):
    - Tiebreaker and explanation are two SEPARATE LLM calls with two
      SEPARATE prompts and two SEPARATE structured-output schemas. DRY-ing
      them would create a single point of failure; if the explanation
      schema broke, tiebreaker verdicts would regress. They live apart.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from evidenceengine.classification.schemas import VerdictClassificationResponse
from evidenceengine.core.config import settings
from evidenceengine.llm.fallback import sync_call_with_fallback

logger = logging.getLogger(__name__)


# Distinct from ``llm_classifier.SYSTEM_PROMPT`` — this prompt EXPLAINS the
# already-decided verdict. It must not re-classify.
EXPLANATION_SYSTEM_PROMPT = (
    "You explain verdicts. Given a claim, evidence spans, a verdict label, "
    "and a confidence score, produce 2-3 sentences explaining why the "
    "verdict was assigned. Do NOT change the verdict. Do NOT speculate. "
    "Cite specific evidence snippets when possible."
)


class ExplanationResponse(BaseModel):
    """Structured-output schema for the explanation call.

    Length bounds keep reviewers from being flooded by chatty models and
    ensure there's at least a meaningful sentence (prevents "ok" / "yes"
    parseable-but-useless outputs from the primary structured-output chain).
    """

    explanation: str = Field(min_length=10, max_length=800)


def _sync_explain(
    claim_text: str,
    evidence_spans: list[dict],
    verdict: VerdictClassificationResponse,
):
    """Sync side of the explanation call — invoked via ``asyncio.to_thread``.

    Keeps the HTTP round-trip off the event loop. Factored out so tests can
    spy on the ``sync_call_with_fallback`` binding in ``explanation`` module
    scope without indirecting through private closures.
    """
    user = (
        f"CLAIM:\n{claim_text}\n\nEVIDENCE:\n"
        + "\n".join(f"- {s['span_text'][:500]}" for s in evidence_spans)
        + f"\n\nVERDICT: {verdict.verdict_type} "
        f"(confidence {verdict.confidence_score:.2f})"
    )
    return sync_call_with_fallback(
        model_chain=settings.model_fallback_chain,
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        response_format=ExplanationResponse,
        timeout=settings.llm_fallback_timeout_seconds,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url or None,
        gemini_api_key=settings.gemini_api_key,
        groq_api_key=settings.groq_api_key,
    )


async def generate_explanation(
    claim_text: str,
    evidence_spans: list[dict],
    verdict: VerdictClassificationResponse,
) -> str | None:
    """Generate a post-verdict explanation. Fail-open by contract (CLF-06).

    Returns the explanation string, or ``None`` when generation failed for
    any reason. Never raises — pipeline code can safely chain the result
    without try/except.
    """
    try:
        message = await asyncio.to_thread(
            _sync_explain, claim_text, evidence_spans, verdict,
        )
    except Exception as exc:  # noqa: BLE001 — FAIL-OPEN by contract
        logger.warning(
            "Explanation generation failed (non-blocking): %s", exc,
        )
        return None

    # sync_call_with_fallback returns the OpenAI choices[0].message directly
    # (see llm/fallback.py — ``return message`` at the end of the per-model
    # try block). We mirror the same access pattern ``classify_claim`` uses.
    parsed = getattr(message, "parsed", None)
    if parsed is None:
        return None
    return parsed.explanation
