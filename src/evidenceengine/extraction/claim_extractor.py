"""Local claim extractor using NLTK sentence splitting.

Replaces the LLM-based extractor to eliminate API dependency from the
hot path. No network calls, no rate limits, no token costs.

Heuristic filters keep only sentence-length factual-looking text,
discarding headers, captions, and table fragments.
"""

import logging
import re

from evidenceengine.extraction.schemas import (
    ClaimExtractionResponse,
    ExtractedClaim,
    ExtractedCitationMarker,
)

logger = logging.getLogger(__name__)

# Prefixes that reliably indicate non-claim content
_SKIP_PREFIXES = (
    "fig", "figure", "table", "appendix", "note ", "notes ",
    "see ", "cf.", "e.g.", "i.e.", "et al.", "ibid",
    "acknowledgement", "acknowledgment", "reference", "bibliography",
    "keywords", "keyword:", "key words",
    "doi:", "doi ", "pmid", "pmcid", "epub ahead",
)

# Patterns that suggest a sentence is a structural artifact, not a claim
_STRUCTURAL_RE = re.compile(
    r"^\s*(\d+[\.\)]\s|[-•–]\s|[A-Z]{2,}\s*:|\([a-z]\))",
    re.IGNORECASE,
)

# Acknowledgement content anywhere in the sentence (not just prefixes)
_ACK_RE = re.compile(
    r"\b(thank|grateful|generously|supported by grant|funded by|"
    r"we gratefully|with the support of)\b",
    re.IGNORECASE,
)

# Self-referential meta-sentences: structural commentary, not verifiable claims
_META_RE = re.compile(
    r"^(in this (paper|work|study|article|section|chapter)|"
    r"this (paper|work|study|section|report|edition) (presents|proposes|describes|introduces|discusses|summari[sz]es|outlines|examines|provides)|"
    r"the (latest|current|new|present) edition of|"
    r"the following (table|figure|section|appendix|chart|graph|diagram)\b|"
    r"(section|chapter|figure|table|appendix)\s+\d)",
    re.IGNORECASE,
)

# URLs: sentences containing links are reference/nav content, not factual claims
_URL_RE = re.compile(r"https?://")

# Author/affiliation lines (CDC, WHO, GPT-4 staff credits):
# require BOTH multiple semicolons AND an academic degree abbreviation.
# Catches: "Smith J. PhD¹; Jones A. MSc11; Lee B. Ph D³"
# Note: no trailing \b — trailing digits (MSc11, Ph D11) are common in PDFs
# Doesn't catch: "Dr. Smith showed that..." (no semicolons)
_AFFILIATION_RE = re.compile(
    r"\b(Ph\.?\s*D|M\.?\s*Sc|M\.?\s*D\b|MPH|MBChB|MBBS|FRCS|FRCP|B\.?\s*Sc)\d*",
    re.IGNORECASE,
)

# TOC fill-character artifacts from PyMuPDF table-of-contents blocks:
# catches "SPM.4.2.1\t......17", Unicode replacement runs, and inline TOC entries
# like "Health-related SDGs 23 2.1 Infectious..." (page number + section X.Y inline).
_TOC_RE = re.compile(r"\t\.{5,}|\t{3,}|\ufffd{4,}|\b\d{1,3}\s+\d{1,2}\.\d{1,2}\s+[A-Z]")

# Publisher imprint lines: "Cambridge University Press, Cambridge, United Kingdom"
# These have no verb so already fail has_verb, but only when terminal-punctuated.
# Explicit guard handles the rare case they slip through as non-punct lines.
_PUBLISHER_RE = re.compile(
    r"^(Cambridge|Oxford|Springer|Elsevier|Wiley|MIT|Academic|CRC|Routledge|"
    r"Taylor\s*&\s*Francis|Sage|IEEE|ACM|Nature|Palgrave|Penguin|HarperCollins|"
    r"Random\s+House)\s+(University\s+Press|Press|Publishing|Publications?)\b",
    re.IGNORECASE,
)

# Student/course header: surname followed immediately by a digit, then more names
# "Kumar 1 Karan Kumar Dr. Tariq BIO 101" — the " \d+ " break after a word is the signal
_STUDENT_HEADER_RE = re.compile(r"^[A-Z][a-z]+ \d+ [A-Z]")

# Author/org list with country or role in parentheses:
# "Paola Arias (Colombia), Mercedes Bustamante (Brazil), Ismail Elgizouli (Sudan)"
# Fires when 2+ parenthesized items appear in a sentence with 3+ commas.
# Catches IPCC, WHO, UN reports that list authors with country affiliations.
_COUNTRY_PARENS_RE = re.compile(r'\([A-Z][a-zA-Z ]{2,25}\)')

# Copyright, legal, and editorial boilerplate common in published PDFs
# and international organization reports (WHO, UN, World Bank, IMF, etc.)
_LEGAL_RE = re.compile(
    r"\b(all rights reserved|reproduction prohibited|without authorization|"
    r"right of publication|rights of translation|editorial correspondence|"
    r"requests to publish|reproduce or translate|©\s*\d{4}|copyright \d{4}|"
    r"isbn[\s\-:]\d|issn[\s\-:]\d|doi[\s:]*10\.|printed in|"
    # CIP / library cataloguing
    r"cataloguing.in.publication|cataloging.in.publication|cip data|"
    # WHO/legal warranty and liability boilerplate
    r"errors and omissions excepted|without warranty of any|"
    r"liable for damages|all reasonable precautions.{1,40}verify|"
    r"distributed without warranty|being distributed without|"
    # International org disclaimers (WHO, UN, World Bank, IMF)
    r"designations employed|approximate border lines|dotted and dashed lines on maps|"
    r"not responsible for the content or accuracy|"
    r"binding and authentic edition|"
    r"mention of specific companies|mention of specific manufacturers|"
    r"reuse material from this work|"
    r"infringement of any third.party|"
    r"logo is not permitted|"
    r"translation of this work[^.!?]{0,60}add the following|"
    r"sales.{1,20}rights.{1,20}licensing|"
    r"presentation of the material in this publication)\b",
    re.IGNORECASE,
)

# Printed webpage navigation — covers common print-to-PDF chrome patterns
# Includes bullet/icon characters (•, ○, ▸) mid-sentence (nav lists)
_NAV_RE = re.compile(
    r"\b(skip to (main )?content|toggle (navigation|menu|button)|"
    r"sign in|log (in|out)|search results|cookie (policy|settings)|"
    r"privacy policy|terms of (use|service)|all rights reserved|"
    r"back to home|stay connected|breadcrumb|navbar|sidebar|"
    r"read more|click here|subscribe now)\b"
    r"|[•○◦▸►▶]{2,}",  # multiple bullet chars = nav list
    re.IGNORECASE,
)

# Known word concatenations from PyMuPDF line-break extraction in ML/scientific papers.
# PyMuPDF drops the space when two consecutive lines share a word boundary without a
# hyphen (e.g. "been\nused" → "beenused"). Lookup applied before sent_tokenize().
_KNOWN_JOINS: dict[str, str] = {
    "asthe": "as the",
    "thesame": "the same",
    "beenused": "been used",
    "isthe": "is the",
    "inthe": "in the",
    "ofthe": "of the",
    "tothe": "to the",
    "forthe": "for the",
    "andthe": "and the",
    "withthe": "with the",
    "fromthe": "from the",
    "bythe": "by the",
    "onthe": "on the",
    "atthe": "at the",
    "isbased": "is based",
    # Capital-prefix joins: "The\nfeature" → "Thefeature" (PyMuPDF line break
    # where the second line starts with lowercase drops the leading space)
    "Thefeature": "The feature",
    "Themodel": "The model",
    # ML paper compound joins seen in BERT/Attention papers (no hyphen, just
    # two words fused by a PyMuPDF line-break that dropped the space)
    "empiricallypowerful": "empirically powerful",
    "layerto": "layer to",
    "widerange": "wide range",
    "andlanguage": "and language",
    "initializemodels": "initialize models",
    "isfeature": "is feature",
    "outputlayer": "output layer",
    "taskspecific": "task specific",
    "downstreamtasks": "downstream tasks",
    "languagemodel": "language model",
    "trainingdata": "training data",
    "machinelearning": "machine learning",
    "deeplearning": "deep learning",
    "neuralnetwork": "neural network",
    # Hyphenation artifacts from PDF line-break extraction
    "re-sult": "result",
    "re-strictions": "restrictions",
    "re-lationships": "relationships",
    "re-presentation": "representation",
    "re-quired": "required",
    "representa-tion": "representation",
    "informa-tion": "information",
    "pre-sented": "presented",
    "pre-diction": "prediction",
    "evalu-ation": "evaluation",
    "classi-fication": "classification",
    "incor-porate": "incorporate",
    "incor-porating": "incorporating",
    "lan-guage": "language",
    "lan-guages": "languages",
    "ma-jor": "major",
    "at-tend": "attend",
    "at-tending": "attending",
    "unidi-rectionality": "unidirectionality",
    "unidirec-tional": "unidirectional",
    "archi-tecture": "architecture",
    "outper-forming": "outperforming",
    "outper-form": "outperform",
    "param-eters": "parameters",
    "param-eter": "parameter",
    # Word joins (no space between words due to PDF line-break)
    "suchas": "such as",
    "suchan": "such an",
    "asnatural": "as natural",
    "aslanguage": "as language",
    "aswell": "as well",
    "asfollows": "as follows",
    "embeddingsare": "embeddings are",
    "embeddingswith": "embeddings with",
    "tocoarser": "to coarser",
    "tofine": "to fine",
    "areunidirectional": "are unidirectional",
    "approachesis": "approaches is",
    "approachis": "approach is",
    "advantageis": "advantage is",
    "modelis": "model is",
    "forexample": "for example",
    "forinstance": "for instance",
    "achievesstate": "achieves state",
    "suiteof": "suite of",
    "Dolanand": "Dolan and",
}


def _fix_word_boundaries(text: str) -> str:
    """Restore spaces lost to PDF line-break extraction artifacts.

    Three passes:
    1. Regex: split at lowercase→Uppercase boundary ("Thefeature" → "The feature").
       Does NOT split all-caps acronyms (BERT, NLP) — regex requires a lowercase
       char before the uppercase char.
    2. Hyphen removal: delete mid-word hyphens before common word suffixes that
       appear only from line-break hyphenation, never in real compound words
       (e.g. "repre-sentation" → "representation", "classi-fied" → "classified").
       Leaves legitimate hyphens ("state-of-the-art", "fine-tuned") untouched.
    3. Lookup: case-insensitive replace of known common joins from ML papers
       ("beenused" → "been used", "Inthe" → "In the"). Preserves leading
       capitalisation so sentence-start joins stay capitalised.
    """
    # Pass 0: sentence boundary — period/comma touching a letter with no space.
    # "powerful.It" → "powerful. It", "result,The" → "result, The"
    text = re.sub(r'([a-z])([.,])([A-Z])', r'\1\2 \3', text)
    # Also fix comma/semicolon immediately followed by a lowercase letter (no space).
    # "tasks,and" → "tasks, and". Guard: don't match digit before comma (1,000 stays).
    text = re.sub(r'([a-z])([,;])([a-z])', r'\1\2 \3', text)
    # Pass 1: camelCase boundary split
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Pass 1.5: proper-noun line-break hyphenation ("Rad-ford" → "Radford").
    # Guard excludes function words ("State-of" stays "State-of") and long parts
    # (> 5 chars after hyphen are likely real compound word halves, not name fragments).
    _keep_after = {'of', 'in', 'to', 'by', 'at', 'an', 'the', 'a', 'is', 'or', 'as', 'on'}
    def _dehyphenate_proper(m: re.Match) -> str:
        return m.group(0) if m.group(2).lower() in _keep_after else m.group(1) + m.group(2)
    text = re.sub(r'([A-Z][a-z]{2,})-([a-z]{2,5})\b', _dehyphenate_proper, text)
    # Pass 2: remove hyphens before suffixes that are never real compound parts
    # Suffixes: -tion, -sion, -ation, -ization, -ment, -ness, -ful, -tion, -ance
    text = re.sub(
        r'([a-z]{3,})-(sentation|tion|sion|ation|ization|ment|ness|ful|ance|ence|'
        r'ture|ive|ary|ory|able|ible|ly|ing|ings|ers|ists|ology|ologies|'
        r'strictions|striction|porate|porates|guage|guages|jority|tend|tending)\b',
        r'\1\2', text, flags=re.IGNORECASE,
    )
    # Pass 3: case-insensitive lookup replacements with word-boundary guards.
    # \b anchors prevent matching within correct words: "Caswell" contains "aswell"
    # but the word boundary between "C" and "a" (both \w) does not exist, so
    # \baswell\b does NOT match inside "Caswell". Artifact "aswell" surrounded by
    # spaces/punctuation does have word boundaries and matches correctly.
    for bad, good in _KNOWN_JOINS.items():
        def _replace(m: re.Match, _good: str = good) -> str:
            return _good[0].upper() + _good[1:] if m.group(0)[0].isupper() else _good
        text = re.sub(r'\b' + re.escape(bad) + r'\b', _replace, text, flags=re.IGNORECASE)
    return text


def _merge_short_blocks(blocks: list[dict]) -> list[dict]:
    """Merge adjacent line-level blocks into paragraph-level blocks.

    PDFs exported from Word or with line-based layout produce one block per line
    (2-15 words, no terminal punctuation). sent_tokenize() returns each fragment
    as-is, and _is_likely_claim() rejects it because it lacks terminal punctuation.

    Merges blocks until the buffer ends in terminal punctuation or exceeds 50 words.
    Paragraph-level PDFs (BERT, Attention) flush immediately since each block already
    ends in terminal punctuation — no merging occurs, no regression.
    """
    merged: list[dict] = []
    buffer_text = ""
    buffer_block: dict | None = None
    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue
        buffer_text = (buffer_text + " " + text).strip() if buffer_text else text
        buffer_block = block
        if text[-1] in ".!?" or len(buffer_text.split()) >= 50:
            merged.append({**buffer_block, "text": buffer_text})
            buffer_text = ""
            buffer_block = None
    if buffer_text and buffer_block is not None:
        merged.append({**buffer_block, "text": buffer_text})
    return merged


# Regex patterns for inline citation marker detection.
# Numeric: [1], [1,2], [1, 2-4]
_NUMERIC_CITATION_RE = re.compile(r'\[\d+(?:[,\s\-]+\d+)*\]')
# Author-year parenthesized: (Smith, 2023), (Jones et al., 2020), (A et al., 2020; B, 2021)
_AUTHOR_YEAR_PAREN_RE = re.compile(
    r'\([A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?(?:,\s*\d{4}[a-z]?)?'
    r'(?:[;,]\s*[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?(?:,\s*\d{4}[a-z]?)?)*'
    r',?\s+\d{4}[a-z]?\)'
)
# Author-year bare: Smith (2023), Jones et al. (2020)
_AUTHOR_YEAR_BARE_RE = re.compile(
    r'[A-Z][a-zA-Z\-]+(?:\s+et\s+al\.?)?\s+\(\d{4}[a-z]?\)'
)


def _detect_citation_markers(text: str) -> list[ExtractedCitationMarker]:
    """Detect numeric and author-year citation markers in a sentence.

    The NLTK extractor can't ask an LLM to identify citations, so we use
    regex. Misses footnote superscripts (no reliable plain-text pattern)
    but catches the two dominant styles in academic PDFs.
    """
    markers: list[ExtractedCitationMarker] = []
    seen: set[str] = set()

    for m in _NUMERIC_CITATION_RE.finditer(text):
        raw = m.group()
        if raw not in seen:
            seen.add(raw)
            markers.append(ExtractedCitationMarker(raw_marker=raw, citation_style="numeric"))

    for pattern in (_AUTHOR_YEAR_PAREN_RE, _AUTHOR_YEAR_BARE_RE):
        for m in pattern.finditer(text):
            raw = m.group()
            if raw not in seen:
                seen.add(raw)
                markers.append(ExtractedCitationMarker(raw_marker=raw, citation_style="author_year"))

    return markers


_MIN_CLAIM_WORDS = 3    # lowered from 5 — catches short metric claims
_MAX_CLAIM_WORDS = 150  # raised from 120 — captures long technical sentences
_MIN_NOPUNCT_WORDS = 15  # raised from 8 — prevents 50-word flush from emitting truncated mid-sentences

_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|shows?|demonstrates?|"
    r"achieves?|reduces?|increases?|decreases?|improves?|"
    r"suggests?|indicates?|contains?|provides?|results?|found|"
    r"occurs?|appears?|causes?|leads?|prevents?|enables?|allows?|"
    r"involves?|affects?|produces?|reveals?|confirms?|supports?|"
    r"includes?|defines?|relates?|depends?|varies?|follows?|"
    r"associates?|correlates?|mediates?|transmits?|inhibits?|"
    r"promotes?|triggers?|requires?|generates?|determines?|"
    r"\w+ed|\w+ing)\b"
)
_METRIC_RE = re.compile(r":\s*[\d\.\-\+]")  # "Accuracy: 95%." style


def _is_likely_claim(sentence: str, _counts: dict | None = None) -> bool:
    """Return True if the sentence looks like a verifiable factual claim.

    _counts: optional mutable dict tracking per-reason rejection counts
             for diagnostic generation when 0 claims are extracted.
    """
    def reject(reason: str) -> bool:
        if _counts is not None:
            _counts[reason] = _counts.get(reason, 0) + 1
        return False

    s = sentence.strip()
    if not s:
        return reject("empty")
    words = s.split()
    if len(words) < _MIN_CLAIM_WORDS or len(words) > _MAX_CLAIM_WORDS:
        return reject("length")
    if not s[0].isupper():
        return reject("lowercase_start")
    lower = s.lower()
    if any(lower.startswith(pfx) for pfx in _SKIP_PREFIXES):
        return reject("skip_prefix")
    if _STRUCTURAL_RE.match(s):
        return reject("structural")
    if _ACK_RE.search(s):
        return reject("acknowledgement")
    if _META_RE.match(s):
        return reject("meta")

    # All-caps heading/TOC: if ≥60% of alphabetic words are fully uppercase,
    # this is a heading or table-of-contents entry, not a factual claim.
    # Threshold of 60% admits legitimate claims with 1-2 acronyms (BERT, WHO, NLP)
    # while catching "MAJOR RECOMMENDATIONS FOR CHINA FOR COUNTRIES..." (100% caps).
    alpha_words = [w for w in words if w.isalpha() and len(w) >= 3]
    if alpha_words and sum(1 for w in alpha_words if w.isupper()) / len(alpha_words) >= 0.60:
        return reject("all_caps_heading")

    # URL-containing sentences are reference/nav content, not factual claims
    if _URL_RE.search(s):
        return reject("url")
    # Author/affiliation lines: semicolons + degree abbreviations
    if s.count(";") >= 2 and _AFFILIATION_RE.search(s):
        return reject("affiliation")
    # TOC fill-character artifacts or tab-heavy lines
    if _TOC_RE.search(s):
        return reject("toc_artifact")
    # Printed webpage navigation chrome
    if _NAV_RE.search(s):
        return reject("webpage_nav")
    # Publisher imprint lines and student course headers
    if _PUBLISHER_RE.match(s):
        return reject("publisher_imprint")
    if _STUDENT_HEADER_RE.match(s):
        return reject("student_header")
    # Author/org lists with country or role in parens (IPCC, WHO, UN style)
    if s.count(",") >= 3 and len(_COUNTRY_PARENS_RE.findall(s)) >= 2:
        return reject("author_org_list")
    # Copyright, legal, and editorial boilerplate
    if _LEGAL_RE.search(s):
        return reject("legal_boilerplate")

    has_punct = s[-1] in ".!?"
    has_verb = bool(_VERB_RE.search(lower))

    if not has_punct:
        # Slide/bullet-point style: accept if long enough AND has a verb.
        # Handles PowerPoint PDFs, Beamer slides, docs without terminal periods.
        if len(words) >= _MIN_NOPUNCT_WORDS and has_verb:
            return True
        return reject("no_terminal_punct")

    if not has_verb:
        # Metric/data-sheet style: accept "Accuracy: 95%." patterns.
        if _METRIC_RE.search(s):
            return True
        return reject("no_verb")

    return True


_ZERO_CLAIM_MESSAGES: dict[str, str] = {
    "no_terminal_punct": (
        "Most sentences lack terminal punctuation. "
        "This PDF may be a slide deck, use heavy bullet-point formatting, "
        "or was exported from a format that strips sentence endings."
    ),
    "no_verb": (
        "Most sentences lack a verb. "
        "Common in data-heavy documents, spreadsheets exported to PDF, "
        "or files dominated by numeric tables and captions."
    ),
    "length": (
        "Sentences are either too short (under 3 words) or very long (over 150 words). "
        "The document may consist mostly of headings, footnotes, or run-on text blocks."
    ),
    "structural": (
        "Content appears to be mostly structural: numbered lists, bullet markers, "
        "or ALL-CAPS headings that don't read as factual claims."
    ),
    "skip_prefix": (
        "Content is dominated by figures, tables, references, or acknowledgements — "
        "all filtered out as non-claim content by design."
    ),
    "lowercase_start": (
        "Many text fragments start with lowercase letters, which usually indicates "
        "parsing artifacts or incomplete sentences from PDF extraction."
    ),
    "meta": (
        "Content is mostly self-referential commentary (\"In this paper...\") "
        "rather than verifiable factual claims."
    ),
    "acknowledgement": "Content is mostly acknowledgements or funding statements.",
    "url": (
        "Most text blocks contain URLs. "
        "This may be a printed webpage rather than a research document. "
        "Upload the original PDF source, not a browser print-to-PDF."
    ),
    "affiliation": (
        "Content is dominated by author/affiliation lists. "
        "The document may be a cover page or directory rather than a research document."
    ),
    "toc_artifact": (
        "Content contains table-of-contents formatting artifacts. "
        "The PDF may have extraction issues or be a TOC-only document."
    ),
    "webpage_nav": (
        "Content contains webpage navigation elements (menus, login buttons, etc.). "
        "This appears to be a printed webpage rather than a research document. "
        "Download the original PDF from the publisher, not a browser print-to-PDF."
    ),
}


def _make_zero_claims_diagnostic(counts: dict[str, int], total_candidates: int) -> str:
    """Build a user-readable explanation for why 0 claims were extracted."""
    if total_candidates == 0:
        return (
            "The document produced no text blocks. "
            "It may be a scanned image-only PDF with no embedded text, "
            "or the file may be empty or corrupt. "
            "Try re-exporting as a PDF with selectable text."
        )
    top = max(counts, key=counts.__getitem__) if counts else "unknown"
    return _ZERO_CLAIM_MESSAGES.get(
        top,
        f"No verifiable claims were found across {total_candidates} candidate sentences.",
    )


async def extract_claims_from_blocks(
    citation_blocks: list[dict],
) -> ClaimExtractionResponse:
    """Extract factual claims from text blocks using NLTK sentence splitting.

    No LLM calls. No API dependency. Runs locally in milliseconds.

    Args:
        citation_blocks: Text blocks from the parsed document.
                         Each dict has at minimum a "text" key.

    Returns:
        ClaimExtractionResponse with filtered factual sentences as claims.
    """
    if not citation_blocks:
        return ClaimExtractionResponse(
            claims=[],
            diagnostic=_make_zero_claims_diagnostic({}, 0),
        )

    # Merge line-level blocks (Word-exported PDFs) into paragraph-level blocks so
    # sent_tokenize receives complete sentences with terminal punctuation.
    citation_blocks = _merge_short_blocks(citation_blocks)

    # Webpage print detection: if >30% of blocks contain nav/UI patterns,
    # this is likely a print-to-PDF of a webpage, not a document.
    nav_count = sum(1 for b in citation_blocks if _NAV_RE.search(b.get("text", "")))
    if citation_blocks and nav_count / len(citation_blocks) > 0.30:
        diag = _ZERO_CLAIM_MESSAGES["webpage_nav"]
        logger.warning("PDF appears to be a printed webpage (%d/%d nav blocks) — %s",
                       nav_count, len(citation_blocks), diag)
        return ClaimExtractionResponse(claims=[], diagnostic=diag)

    try:
        from nltk.tokenize import sent_tokenize
    except ImportError:
        logger.error("nltk not available — returning empty claims")
        return ClaimExtractionResponse(claims=[])

    all_claims: list[ExtractedClaim] = []
    seen: set[str] = set()
    rejection_counts: dict[str, int] = {}
    total_candidates = 0
    prev_sentence = ""  # tracks the sentence immediately before the current one, across blocks

    for block in citation_blocks:
        text = _fix_word_boundaries(block.get("text", "").strip())
        if not text:
            continue
        try:
            sentences = sent_tokenize(text)
        except Exception as exc:
            logger.warning("sent_tokenize failed for block: %s", exc)
            sentences = [s.strip() for s in text.split(".") if s.strip()]

        for sentence in sentences:
            sentence = sentence.strip()
            total_candidates += 1
            if not _is_likely_claim(sentence, rejection_counts):
                prev_sentence = sentence
                continue
            # Deduplicate
            key = sentence.lower()
            if key in seen:
                prev_sentence = sentence
                continue
            seen.add(key)
            markers = _detect_citation_markers(sentence)
            # If no in-sentence citations, inherit from the preceding sentence.
            # Scientific writing often places a citation in sentence N and makes
            # a verifiable claim in sentence N+1: "...shown by Vaswani et al.
            # (2017). Such restrictions are sub-optimal..." — the claim sentence
            # has no marker, but the citation context clearly applies.
            if not markers and prev_sentence:
                markers = _detect_citation_markers(prev_sentence)
            all_claims.append(ExtractedClaim(
                claim_text=sentence,
                citation_markers=markers,
            ))
            prev_sentence = sentence

    logger.info(
        "Local extractor produced %d claims from %d candidates across %d blocks",
        len(all_claims), total_candidates, len(citation_blocks),
    )

    if not all_claims:
        diagnostic = _make_zero_claims_diagnostic(rejection_counts, total_candidates)
        logger.warning("0 claims extracted — %s", diagnostic)
        return ClaimExtractionResponse(claims=[], diagnostic=diagnostic)

    return ClaimExtractionResponse(claims=all_claims)
