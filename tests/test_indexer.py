"""Unit tests for span extraction (indexer.py) — RED phase."""

import pytest


def make_block(text, page=1, paragraph=2, char_start=100, char_end=200, section_header="Results"):
    return {
        "text": text,
        "block_type": "paragraph",
        "position": {
            "page": page,
            "paragraph": paragraph,
            "char_start": char_start,
            "char_end": char_end,
            "section_header": section_header,
        },
    }


def make_parsed_content(*blocks):
    return {"blocks": list(blocks)}


# --- Tests ---

def test_paragraph_span_created():
    """Single block → one paragraph span with same text."""
    from evidenceengine.retrieval.indexer import extract_spans

    content = make_parsed_content(make_block("Carbon emissions increased by 12% in this period."))
    spans = extract_spans(content)
    para_spans = [s for s in spans if s["span_type"] == "paragraph"]
    assert len(para_spans) == 1
    assert para_spans[0]["text"] == "Carbon emissions increased by 12% in this period."


def test_sentence_spans_created():
    """Block with 2 sentences → 2 sentence-level spans."""
    from evidenceengine.retrieval.indexer import extract_spans

    content = make_parsed_content(
        make_block(
            "Carbon emissions increased by 12% in this period. Sea ice extent declined rapidly."
        )
    )
    spans = extract_spans(content)
    sent_spans = [s for s in spans if s["span_type"] == "sentence"]
    assert len(sent_spans) == 2


def test_position_inherited():
    """Paragraph span inherits all positional fields from block.position."""
    from evidenceengine.retrieval.indexer import extract_spans

    content = make_parsed_content(
        make_block("Some text here for testing position.", page=5, paragraph=3, char_start=400, char_end=500, section_header="Discussion")
    )
    spans = extract_spans(content)
    para_span = next(s for s in spans if s["span_type"] == "paragraph")
    assert para_span["page"] == 5
    assert para_span["paragraph"] == 3
    assert para_span["char_start"] == 400
    assert para_span["char_end"] == 500
    assert para_span["section_header"] == "Discussion"


def test_sentence_span_inherits_page():
    """Sentence spans inherit page and paragraph from parent block."""
    from evidenceengine.retrieval.indexer import extract_spans

    content = make_parsed_content(
        make_block(
            "First sentence is long enough. Second sentence is also long enough.",
            page=7,
            paragraph=4,
        )
    )
    spans = extract_spans(content)
    sent_spans = [s for s in spans if s["span_type"] == "sentence"]
    for s in sent_spans:
        assert s["page"] == 7
        assert s["paragraph"] == 4


def test_short_sentence_skipped():
    """Sentence shorter than 20 chars should not appear as a sentence span."""
    from evidenceengine.retrieval.indexer import extract_spans

    content = make_parsed_content(
        make_block("Fig. 3. This is a much longer sentence that meets the minimum length threshold.")
    )
    spans = extract_spans(content)
    sent_spans = [s for s in spans if s["span_type"] == "sentence"]
    # "Fig." alone or short fragments should be filtered
    for s in sent_spans:
        assert len(s["text"].strip()) >= 20


def test_empty_text_block_skipped():
    """Block with empty text produces no spans at all."""
    from evidenceengine.retrieval.indexer import extract_spans

    content = make_parsed_content(make_block(""))
    spans = extract_spans(content)
    assert spans == []


def test_abbreviation_not_split():
    """'Dr. Smith reported findings.' is a single sentence (NLTK handles abbreviations)."""
    from evidenceengine.retrieval.indexer import extract_spans

    content = make_parsed_content(
        make_block("Dr. Smith reported the findings in the study.")
    )
    spans = extract_spans(content)
    sent_spans = [s for s in spans if s["span_type"] == "sentence"]
    assert len(sent_spans) == 1


def test_multiple_blocks():
    """Two blocks → paragraph spans for each + sentence spans for each."""
    from evidenceengine.retrieval.indexer import extract_spans

    content = make_parsed_content(
        make_block("First block text with a sentence. Another sentence here in block one."),
        make_block("Second block text content here. And another sentence in block two."),
    )
    spans = extract_spans(content)
    para_spans = [s for s in spans if s["span_type"] == "paragraph"]
    assert len(para_spans) == 2


def test_ensure_punkt_idempotent():
    """Calling ensure_punkt() twice does not raise any exception."""
    from evidenceengine.retrieval.indexer import ensure_punkt

    ensure_punkt()
    ensure_punkt()  # second call — should be a no-op
