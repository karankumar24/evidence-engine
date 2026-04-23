"""LLM claim extractor using structured output.

Uses the synchronous OpenAI client inside asyncio.to_thread so that
httpx network I/O (which blocks the event loop on macOS via the async
client) runs in a worker thread, keeping uvicorn fully responsive.
"""

import asyncio
import logging

from evidenceengine.core.config import settings
from evidenceengine.extraction.schemas import ClaimExtractionResponse

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise factual-claim extractor. Your task is to identify every verifiable factual claim in the text.

A factual claim is a COMPLETE SENTENCE that asserts something a reader could check — numbers, dates, percentages, causal statements, attributions, outcomes, comparisons, superlatives, or any specific empirical assertion.

Rules:
1. Extract only well-formed sentences: must start with a capital letter, end with a period/question/exclamation, and contain a subject and a verb.
2. SKIP table cells, row headers, column headers, bullet fragments, and bare numeric line items. Examples to SKIP: "Products $69,958", "Services", "Total net sales (1)", "Three Months Ended". These are data, not claims.
3. SKIP pure navigation or document structure ("This report summarises...", "See page 12", "Figure 3").
4. Copy claim sentences VERBATIM — word-for-word, no paraphrasing, no summarizing.
5. Extract a sentence whether or not it has a citation marker. If markers exist, record them with style:
   - "numeric": [1], [2,3], [1-3]
   - "author_year": Smith 2023, (Jones et al., 2021)
   - "footnote": ¹, ², superscript numbers
   If a claim has no citation marker, return an empty citation_markers list for it.
6. Aim for QUALITY over quantity. A 4-page financial statement should produce ~5–20 claims, not 100+. If you find yourself extracting every table row, you are over-extracting: stop, reconsider, and keep only narrative sentences that frame the numbers."""

_MAX_BLOCKS_PER_CALL = 50


async def extract_claims_from_blocks(
    citation_blocks: list[dict],
) -> ClaimExtractionResponse:
    """Extract cited claims from pre-filtered text blocks using LLM structured output.

    Runs the synchronous OpenAI client in a thread pool (via asyncio.to_thread)
    to prevent httpx from blocking the uvicorn event loop on macOS.

    Args:
        citation_blocks: Blocks already filtered to those containing citation markers.
                         Each dict has at minimum a "text" key.

    Returns:
        ClaimExtractionResponse with only sentences carrying citation markers.
        Returns empty claims list for empty input or on LLM refusal.
    """
    if not citation_blocks:
        return ClaimExtractionResponse(claims=[])

    all_claims: list = []
    for chunk_start in range(0, len(citation_blocks), _MAX_BLOCKS_PER_CALL):
        chunk = citation_blocks[chunk_start : chunk_start + _MAX_BLOCKS_PER_CALL]
        text_content = "\n\n".join(block.get("text", "") for block in chunk)

        # Run synchronous OpenAI client in a thread to avoid blocking the event loop.
        # AsyncOpenAI + httpx on macOS blocks the asyncio selector for the full
        # request duration; the sync client in a thread is safe and non-blocking.
        def _sync_request(content: str = text_content) -> object:
            from evidenceengine.llm.fallback import sync_call_with_fallback  # noqa: PLC0415
            chain = settings.model_fallback_chain or [settings.extraction_model]
            timeout = (
                settings.llm_fallback_timeout_seconds
                if len(chain) > 1
                else settings.llm_request_timeout_seconds
            )
            return sync_call_with_fallback(
                model_chain=chain,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format=ClaimExtractionResponse,
                timeout=timeout,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
                gemini_api_key=settings.gemini_api_key,
                groq_api_key=settings.groq_api_key,
            )

        try:
            message = await asyncio.to_thread(_sync_request)
        except Exception as exc:
            # Per-batch resilience: one chain-exhausted / timeout failure
            # must not discard claims from earlier successful batches.
            logger.warning(
                "Extraction batch starting at block %d failed (%s: %s) — "
                "skipping this batch, continuing with next.",
                chunk_start, type(exc).__name__, exc,
            )
            continue

        if message.refusal:
            logger.warning(
                "LLM refused to extract claims: %s. Returning empty for this chunk.",
                message.refusal,
            )
            continue

        if message.parsed is None:
            logger.error(
                "LLM returned null parsed response for extraction chunk starting at "
                "block %d — structured output may have failed. Skipping chunk.",
                chunk_start,
            )
            continue

        all_claims.extend(message.parsed.claims)

    return ClaimExtractionResponse(claims=all_claims)
