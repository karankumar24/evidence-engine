"""Position recovery: maps LLM-returned claim text back to character offsets in raw_text.

Pure function — no database access, no I/O. Takes plain strings and dicts.
"""


def recover_position(
    claim_text: str,
    raw_text: str,
    parsed_blocks: list[dict],
) -> dict:
    """Locate a claim string in raw_text and return its character offsets and metadata.

    Tries exact match first; falls back to normalized whitespace comparison.

    Args:
        claim_text: The verbatim claim text returned by the LLM extractor.
        raw_text: The full raw text of the source document.
        parsed_blocks: Parsed content blocks with position metadata.
                       Each block has: text, block_type, position (page, paragraph,
                       char_start, char_end, section_header).

    Returns:
        Dict with:
            char_start (int): Start character offset in raw_text (0 if not found)
            char_end (int): End character offset in raw_text (0 if not found)
            page_number (int | None): Page number from matching block, or None
            paragraph_index (int | None): Paragraph index from matching block, or None
            section_header (str | None): Section header from matching block, or None
            position_exact (bool): True if exact text match succeeded
    """
    # --- Exact match ---
    idx = raw_text.find(claim_text)
    position_exact = idx != -1

    # --- Normalized fallback ---
    # Track norm_claim so char_end can use its length when it's the matched form.
    norm_claim: str | None = None
    if idx == -1:
        norm_claim = " ".join(claim_text.split())
        norm_raw = " ".join(raw_text.split())
        norm_idx = norm_raw.find(norm_claim)
        if norm_idx != -1:
            # Prefer finding the normalized form directly in raw_text so idx is a
            # true raw offset. norm_idx is an offset into the *normalized* string —
            # only use it as a last-resort approximation.
            raw_idx = raw_text.find(norm_claim)
            idx = raw_idx if raw_idx != -1 else norm_idx

    char_start = max(idx, 0)
    if idx == -1:
        char_end = 0
    elif not position_exact and norm_claim is not None:
        # Matched via normalized form — span length must use norm_claim, not
        # claim_text, because that's what was actually located in raw_text.
        char_end = char_start + len(norm_claim)
    else:
        char_end = char_start + len(claim_text)

    # --- Find matching block by char_start offset ---
    # Only match when we actually located the claim; when idx == -1 the
    # char_start=0 sentinel would spuriously match the first block starting at 0.
    matched_block: dict | None = None
    if idx != -1:
        for block in parsed_blocks:
            pos = block.get("position", {})
            block_start = pos.get("char_start", -1)
            block_end = pos.get("char_end", -1)
            if block_start <= char_start < block_end:
                matched_block = block
                break

    page_number: int | None = None
    paragraph_index: int | None = None
    section_header: str | None = None

    if matched_block is not None:
        pos = matched_block.get("position", {})
        page_number = pos.get("page")
        paragraph_index = pos.get("paragraph")
        section_header = pos.get("section_header")

    return {
        "char_start": char_start,
        "char_end": char_end,
        "page_number": page_number,
        "paragraph_index": paragraph_index,
        "section_header": section_header,
        "position_exact": position_exact,
    }
