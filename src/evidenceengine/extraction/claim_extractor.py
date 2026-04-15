"""Async LLM claim extractor using AsyncOpenAI structured output.

Calls GPT-4o-mini (or configured model) with response_format=ClaimExtractionResponse
to extract only sentences that carry citation markers from report text blocks.
"""

import logging

from openai import AsyncOpenAI

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

    Args:
        citation_blocks: Blocks already filtered to those containing citation markers.
                         Each dict has at minimum a "text" key.

    Returns:
        ClaimExtractionResponse with only sentences carrying citation markers.
        Returns empty claims list for empty input or on LLM refusal.
    """
    if not citation_blocks:
        return ClaimExtractionResponse(claims=[])

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Chunk blocks to avoid context limits
    all_claims: list = []
    for chunk_start in range(0, len(citation_blocks), _MAX_BLOCKS_PER_CALL):
        chunk = citation_blocks[chunk_start : chunk_start + _MAX_BLOCKS_PER_CALL]
        text_content = "\n\n".join(block.get("text", "") for block in chunk)

        completion = await client.chat.completions.parse(
            model=settings.extraction_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text_content},
            ],
            response_format=ClaimExtractionResponse,
        )

        message = completion.choices[0].message
        if message.refusal:
            logger.warning(
                "LLM refused to extract claims: %s. Returning empty for this chunk.",
                message.refusal,
            )
            continue

        if message.parsed:
            all_claims.extend(message.parsed.claims)

    return ClaimExtractionResponse(claims=all_claims)
