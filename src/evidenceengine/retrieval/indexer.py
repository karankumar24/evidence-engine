"""Span extraction from parsed_content blocks at sentence + paragraph granularity."""

import nltk


def ensure_punkt() -> None:
    """Download punkt_tab tokenizer data if not already present (idempotent).

    punkt_tab is the current resource name in NLTK 3.9+; punkt is deprecated.
    Must be called before any nltk.sent_tokenize() invocation.
    """
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


ensure_punkt()  # Idempotent — safe at module import time


def extract_spans(parsed_content: dict) -> list[dict]:
    """Extract sentence-level and paragraph-level spans from a parsed_content block dict.

    Each block in parsed_content["blocks"] produces:
    - 1 paragraph-level span (the full block text)
    - N sentence-level spans (from NLTK sent_tokenize, each >= 20 chars)

    Sentence-level spans inherit positional metadata (page, paragraph, char_start,
    char_end, section_header) from their parent block. char_start/char_end for sentences
    are approximate (paragraph-level positions) — exact offsets would require char search
    within each block, adding complexity with no benefit for Phase 3 retrieval.

    Returns list of span dicts, each with:
    {text, page, paragraph, char_start, char_end, section_header, span_type}
    """
    spans = []
    for block in parsed_content.get("blocks", []):
        text = block.get("text", "").strip()
        if not text:
            continue
        pos = block.get("position", {})
        page = pos.get("page")
        paragraph = pos.get("paragraph")
        char_start = pos.get("char_start", 0)
        char_end = pos.get("char_end", 0)
        section_header = pos.get("section_header")

        # Paragraph-level span (full block text)
        spans.append({
            "text": text,
            "page": page,
            "paragraph": paragraph,
            "char_start": char_start,
            "char_end": char_end,
            "section_header": section_header,
            "span_type": "paragraph",
        })

        # Sentence-level spans
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            if len(sent.strip()) < 20:
                continue  # skip very short fragments (e.g. "Fig. 3.")
            spans.append({
                "text": sent,
                "page": page,
                "paragraph": paragraph,
                "char_start": char_start,   # approximate
                "char_end": char_end,         # approximate
                "section_header": section_header,
                "span_type": "sentence",
            })
    return spans
