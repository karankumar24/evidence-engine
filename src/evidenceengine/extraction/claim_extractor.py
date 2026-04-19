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

_SYSTEM_PROMPT = """You are a precise scientific claim extractor. Your task is to identify sentences that make factual claims supported by citations.

Rules:
1. Extract ONLY sentences that contain citation markers (e.g., [1], Smith 2023, ¹)
2. Copy claim sentences VERBATIM — word-for-word, no paraphrasing, no summarizing
3. Do NOT include uncited background sentences
4. For each claim, identify ALL citation markers within it and their style:
   - "numeric": [1], [2,3], [1-3]
   - "author_year": Smith 2023, (Jones et al., 2021)
   - "footnote": ¹, ², superscript numbers
5. A single sentence may have multiple citation markers — include all of them
6. If no citation markers are found in the text, return an empty claims list"""

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
        def _sync_request(content: str = text_content) -> ClaimExtractionResponse | None:
            from openai import OpenAI as _SyncOpenAI  # noqa: PLC0415
            with _SyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or None,
                timeout=settings.llm_request_timeout_seconds,
                max_retries=settings.llm_max_retries,
            ) as client:
                completion = client.beta.chat.completions.parse(
                    model=settings.extraction_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    response_format=ClaimExtractionResponse,
                )
            return completion.choices[0].message

        message = await asyncio.to_thread(_sync_request)

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
