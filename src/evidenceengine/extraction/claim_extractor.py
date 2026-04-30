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
    "for a review", "for an overview", "for more detail", "for details,",
    # Journal header lines (publication metadata, not claims).
    # "journal of " covers JMLR, NEJM, etc. "proceedings of " excluded — legal/policy
    # docs legitimately write "proceedings of this investigation revealed…"
    "journal of ",
    # Figure/chart description starters — universal across all document types
    "the dotted ", "the dashed ", "the solid line", "the shaded area",
    "bars represent", "bars show", "bars indicate", "the bars ",
    "the horizontal lines", "the vertical lines",
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

# Funding / conflict / data-availability / authorship-disclosure boilerplate.
# Universal across post-2020 transparency-required journals (PMC, Frontiers,
# PLOS, BMC, NEJM, Cell). These sentences are NOT verifiable research claims
# even though NLI may say "supported" because they're literally true (e.g.
# "the authors received funding" → matches actual funding disclosure spans).
# Surfaced in 2026-04-27 bio pressure test (semaglutide review false-positive
# supported on funding-declaration sentence).
# Bibliography / reference-list titles — NOT research findings.
# Clinical-trial bibliography entries follow universal "(ACRONYM): a [design] trial"
# shape across all biomedical journals. Body findings phrase differently
# ("patients in SUSTAIN 11 showed...") so this pattern has very low FP risk.
# Surfaced 2026-04-27 bio pressure test: 4 of 5 false-positive SUPPORTED on
# bibliography titles after disclosure filter freed up extraction slots.
# Title-shape: any sentence ending with "trial." / "study." / "analysis." /
# "report." preceded by ": a" or ": the" — the universal bibliography-entry
# pattern across clinical journals. Catches both:
#   "(SUSTAIN 11): a randomized, open-label, phase 3b trial."
#   "(SUSTAIN 2): a 56-week, double-blind, phase 3a, randomised trial."
#   ": the PIONEER 2 trial."
# Body findings phrase as "patients in SUSTAIN 11 trial showed..." — no
# ":\s*(?:a|the)\s+" gate before "trial.", so they don't match.
_BIBLIOGRAPHY_RE = re.compile(
    # Original: ":/(): a/the ... trial/study/analysis/report/review."
    r"(?:\):\s*a|\):\s*the|:\s*a|:\s*the)\s+"
    r"[^.]{0,200}?"
    r"\b(?:trial|study|analysis|report|review)\.?\s*$"
    # Also catch ref-list shape: ends in "(ACRONYM). JournalAbbrev." pattern.
    # Surfaced 2026-04-28: "(SUSTAIN 10). Diabetes Metab." leaked as supported.
    # Journal-abbrev tail = 1-3 Title-Case tokens, optional period, end-of-sentence.
    r"|\([A-Z][A-Z0-9 \-]{1,30}\)\.\s+"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\.?\s*$"
    # v3 (2026-04-30): URL access-date boilerplate from ref lists.
    # "Accessed January 24, 2024." / "Accessed July 10, 2020."
    # Surfaced as 4 contradicted FPs on PCV20 paper — pure trash, NLI ran on
    # date strings against retrieved spans and produced spurious contradictions.
    r"|^accessed\s+\w+\s+\d+,?\s+\d{4}\.?\s*$",
    re.IGNORECASE,
)

# Meta-claims about study objectives ("objective was demonstration of X",
# "objectives included demonstrating Y") trivially self-verify because the
# paper explicitly states what they intended — not what they found.
# Surfaced 2026-04-30 visual QA on PCV20 paper: 2/7 supported FPs were
# objective-meta sentences. Distinct from research findings, which describe
# observed results, not study design.
_META_OBJECTIVE_RE = re.compile(
    r"\b(?:primary|secondary|key|main)?\s*objectives?\s+"
    r"(?:was|were|include[ds]?)\s+"
    r"(?:to\s+)?"
    r"(?:demonstrat|prov|show|establish|test|assess|determin|evaluat|examin)",
    re.IGNORECASE,
)

# Glossary / abbreviation lists: "GMC indicates X; GMFR, Y; Ig G, Z" — universal
# table-footnote shape across journals. Surfaced 2026-04-30 visual QA.
_GLOSSARY_RE = re.compile(
    r"\b(indicates|stands\s+for|refers\s+to|denotes|represents)\b"
    r".*?[;,].*?[A-Z][a-z]*[,;]",
    re.IGNORECASE,
)

# Open-access / Creative Commons license boilerplate. Universal across
# open-access bio journals (PMC, PLOS, BMC, etc.). Not a research finding.
_LICENSE_RE = re.compile(
    r"\b(open[\- ]access\s+article|creative\s+commons|"
    r"CC[\s\-]BY[\s\-]?(?:NC|ND|SA)?|"
    r"distributed\s+under\s+the\s+terms\s+of)\b",
    re.IGNORECASE,
)

# Journal page-of-total header bleeding into sentence body. PyMuPDF can glue
# a running-head ("Cells 2024, 13, 8004 of 27") onto the start of the next
# paragraph's first sentence. Pattern: TitleCase + year + numbers + "X of Y".
# Surfaced 2026-04-30 visual QA on CRISPR paper (run 3d8dae7c).
_JOURNAL_PAGE_HEADER_RE = re.compile(
    r"^[A-Z][a-z]{2,15}\s+\d{4},?\s+\d+,?\s+\d+\s+of\s+\d+\s+",
    re.IGNORECASE,
)

# Methods-procedural sentences: statistical/methodology boilerplate that NLI
# trivially "supports" via self-verify but carries no research finding.
# Surfaced 2026-04-28 visual QA: 4/8 cardio supported = methods boilerplate
# ("p<0.05", "ROBINS-I assessed", "Cochrane guidelines", "Open Meta-Analyst").
# v2 (2026-04-29): allow optional adverb between aux + verb to catch
# "were strictly reserved", "was carefully assessed". Add direct "assessed using"
# pattern. Tighten escape to require finding verb (not just clinical noun).
_METHODS_PROCEDURAL_RE = re.compile(
    r"\b(?:was|were|are|is)\s+(?:\w+\s+)?"
    r"(?:assessed|used|considered|performed|"
    r"conducted|calculated|applied|reserved|extracted|computed|"
    r"recorded|reviewed|scored|evaluated|measured)\b"
    r"|\bassessed\s+using\b"
    r"|\bp\s+(?:value|values?)\b\s+(?:below|above|of|<|>)",
    re.IGNORECASE,
)
_CLINICAL_NOUN_RE = re.compile(
    r"\b(?:patient|patients|trial|trials|cohort|subject|subjects|"
    r"participant|participants|dose|dosing|treatment|treatments|"
    r"outcome|outcomes|risk|risks|mortality|incidence|rate|rates|"
    r"effect|effects|reduction|increase|decrease|response|responses|"
    r"adverse|efficacy|safety|hazard|odds|relative)\b",
    re.IGNORECASE,
)
# Finding-style verbs that signal a research result rather than methodology.
# Required (in addition to clinical noun) to escape the methods-procedural reject.
_FINDING_RESULT_VERB_RE = re.compile(
    r"\b(?:reduced|increased|decreased|showed|demonstrated|found|"
    r"observed|reported|achieved|improved|worsened|associated|"
    r"correlated|attributed|prevented|eliminated|delivered|"
    r"experienced|received|developed|exhibited|presented)\b",
    re.IGNORECASE,
)

_DISCLOSURE_RE = re.compile(
    # author(s) declare — tolerant of PyMuPDF "decl are" line-break artifact
    r"\b(author(?:\(s\)|s)?\s+decl\s*(?:are|ares|aration)|"
    r"received\s+(?:financial\s+)?support\s+(?:from|for)|"
    r"competing\s+interests?|conflict\s+of\s+interest|"
    r"funding\s+(?:statement|source|disclosure)|"
    r"data\s+availability\s+statement|"
    r"ethics\s+(?:approval|statement|committee)|"
    r"informed\s+consent\s+was\s+obtained|"
    r"institutional\s+review\s+board|\bIRB\s+approval)\b",
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

# TOC fill-character artifacts from PyMuPDF table-of-contents blocks.
# Catches tab+dots, multiple tabs, Unicode replacement runs, inline numeric TOC,
# AND plain-period fill runs ("Conclusion ......... 29") where the dots are not
# tab-preceded \u2014 common in BIS/ECB/World Bank quarterly reports.
_TOC_RE = re.compile(r"\t\.{5,}|\t{3,}|\ufffd{4,}|\b\d{1,3}\s+\d{1,2}\.\d{1,2}\s+[A-Z]|\.{5,}")

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

# Universal legal/deontic/IP filter.
# 4 categories that appear in legal text across ANY organization or document type:
# deontic modals (shall not / may not), IP terms (copyright/trademark/patent),
# liability language (liable/disclaimer/warranty), reproduction control.
# No org-specific phrases — this works for WHO, World Bank, journals, patents, standards.
_LEGAL_UNIVERSAL_RE = re.compile(
    r"\b(all\s+rights?\s+reserved|copyright\s+\d{4}|©\s*\d{4}|"
    r"ISBN[\s\-:]\d|ISSN[\s\-:]\d|doi[\s:]*10\.|"                  # publication IDs
    r"reproduction\s+(?:prohibited|without)|"                         # reproduction restriction
    r"shall\s+not\s+be\s+(?:liable|reproduced|used)|"                # deontic modal
    r"may\s+not\s+be\s+(?:reproduced|copied|used)|"                  # deontic modal
    r"without\s+(?:written\s+)?permission|"                           # permission requirement
    r"for\s+(?:commercial\s+)?reproduction|"                          # reproduction clause
    r"\bliabilit(?:y|ies)\b|\bliable\b|\bindemnif\w+\b|"            # liability terms
    r"\bwithout\s+warrant(?:y|ies)\b|\bno\s+warrant(?:y|ies)\b|"   # warranty disclaimer
    r"\bdisclaims?\b|\bdisclaimer\b|"                                 # disclaimer
    r"\btrademark\b|\bpatented?\b|"                                   # IP terms
    r"\bprohibited\b|\bforbidden\b|\bunauthori[sz]ed\b|"             # access control
    r"printed\s+in\s+[A-Z][a-z]|"                                    # "Printed in France"
    r"catalogu?ing[- ]in[- ]publication|cip\s+data)",                  # library CIP
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

# Known word concatenations from PyMuPDF line-break extraction in scientific papers.
# PyMuPDF drops the space when two consecutive lines share a word boundary without a
# hyphen (e.g. "been\nused" → "beenused"). Lookup applied before sent_tokenize().
# Limited to function-word and generic-suffix joins seen across all scientific domains
# (biomedical, clinical, life sciences). The universal regex Pass 1.6 in
# _fix_word_boundaries covers most cases; this lookup is residual coverage.
_KNOWN_JOINS: dict[str, str] = {
    # Function-word + article joins (universal across all domains)
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
    # Generic hyphenation artifacts from PDF line-break extraction
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
    "ma-jor": "major",
    "param-eters": "parameters",
    "param-eter": "parameter",
    # Generic word joins seen across scientific PDFs
    "suchas": "such as",
    "suchan": "such an",
    "aswell": "as well",
    "asfollows": "as follows",
    "approachesis": "approaches is",
    "approachis": "approach is",
    "advantageis": "advantage is",
    "modelis": "model is",
    "forexample": "for example",
    "forinstance": "for instance",
    "suiteof": "suite of",
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
    # Pass 1.6: split content-word + function-word joins where space was lost
    # ("intendedfor" → "intended for", "Transformerwas" → "Transformer was",
    # "referthe" → "refer the"). Requires content word ≥4 chars to avoid
    # damaging real words. Function words restricted to ≥3 chars: 2-char words
    # ("as", "or", "is", "to", "of") cause greedy false matches inside real
    # words ("Transformerwas" → "Transformerw as" via greedy "as" capture).
    text = re.sub(
        r'([a-z]{4,})(for|the|and|but|with|from|that|this|these|those|when|then|than|'
        r'over|under|after|while|before|because|though|was|were|are|'
        r'has|have|had|would|could|should|may|might|will|shall|can|did|does)\b',
        r'\1 \2', text,
    )
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
# Numeric (Vancouver bracketed): [1], [1,2], [1, 2-4]
_NUMERIC_CITATION_RE = re.compile(r'\[\d+(?:[,\s\-]+\d+)*\]')
# Numeric (paren-bracketed): (12), (12, 13), (12-15)
# Bio journals (Frontiers, PLOS, BMC, Cell, Heliyon) commonly use this.
# 1-3 digit cap excludes year-like 4-digit numbers to avoid conflict with author-year style.
_NUMERIC_PAREN_CITATION_RE = re.compile(r'\(\d{1,3}(?:[,\s–\-]+\d{1,3})*\)')
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

    # Author-year scanned BEFORE numeric-paren so (Smith, 2023) doesn't get
    # mis-classified by the looser numeric-paren pattern (4-digit year filter
    # in numeric-paren means it shouldn't match author-year, but ordering is
    # belt-and-suspenders).
    for pattern in (_AUTHOR_YEAR_PAREN_RE, _AUTHOR_YEAR_BARE_RE):
        for m in pattern.finditer(text):
            raw = m.group()
            if raw not in seen:
                seen.add(raw)
                markers.append(ExtractedCitationMarker(raw_marker=raw, citation_style="author_year"))

    # Numeric paren style — bio journals (Frontiers, PLOS, BMC, Cell). Run
    # last so existing author-year and bracketed-numeric matches take priority.
    for m in _NUMERIC_PAREN_CITATION_RE.finditer(text):
        raw = m.group()
        if raw not in seen:
            seen.add(raw)
            # Reuse "numeric" style — same downstream resolution path
            # (lookup reference list by number, fuzzy match to source docs).
            markers.append(ExtractedCitationMarker(raw_marker=raw, citation_style="numeric"))

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

# Precision gate: a sentence must contain at least one factual signal to pass.
# Filters vague definitional claims ("Protein folding is complex") that are not
# evidence-checkable. Any 2+ digit number, causal/comparative language, or a
# statistical qualifier qualifies. Built from universal linguistic properties —
# not document-type-specific.
_FACTUAL_SIGNAL_RE = re.compile(
    # Decimal number — almost always scientific measurement (1.1, 17.9, 0.05)
    r"\b\d+\.\d+\b"
    # Quantified number with unit or scientific domain noun
    r"|\b\d+\.?\d*\s*(%|percent|‰|fold|times|"
    r"million|billion|trillion|thousand|"
    r"mg|kg|µg|ng|ml|mm|cm|km|nm|µm|Hz|kHz|MHz|°C|°F|"
    r"degree[s]?\s+(?:celsius|fahrenheit|kelvin|centigrade)|"
    r"bp|kb|Mb|yr|year|day|week|month|"
    r"patient|case|sample|participant|subject|trial|individual|cohort|"
    r"species|gene|protein|compound|drug|dose)\b"
    # Causal language
    r"|\b(due\s+to|caused?\s+by|leads?\s+to|results?\s+in|"
    r"associated\s+with|attributed\s+to|because\s+of|owing\s+to|"
    r"consequently|hence|therefore|driven\s+by|mediated\s+by|modulated\s+by)\b"
    # Comparative language
    r"|\b(more\s+than|less\s+than|higher\s+than|lower\s+than|"
    r"greater\s+than|fewer\s+than|at\s+least|at\s+most|"
    r"compared\s+(?:to|with)|relative\s+to|versus|vs\.|"
    r"exceed[s]?|outperform[s]?|superior\s+to|inferior\s+to)\b"
    # Statistical qualifier
    r"|\b(significantly|substantially|markedly|statistically|"
    r"considerably|dramatically|notably)\b"
    # Any bare 2+ digit number (catches years, counts, measurements)
    r"|\b\d{2,}\b",
    re.IGNORECASE,
)

# Finding verbs — fallback signal for long declarative sentences that lack a
# hard number but clearly report a scientific result.
_FINDING_VERB_RE = re.compile(
    r"\b(shows?|demonstrates?|indicates?|suggests?|reveals?|confirms?|"
    r"finds?|found|reports?|establishes?|proves?|validates?|"
    r"supports?|contradicts?|refutes?|challenges?|"
    r"observes?|observed|measures?|measured|analyz(?:es?|ed))\b",
    re.IGNORECASE,
)


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
    if _DISCLOSURE_RE.search(s):
        return reject("disclosure_boilerplate")
    if _BIBLIOGRAPHY_RE.search(s):
        return reject("bibliography_entry")
    if _METHODS_PROCEDURAL_RE.search(s):
        # Escape only when sentence has BOTH a clinical noun AND a finding verb —
        # i.e. it describes an observed result, not a methodology step.
        has_clinical = bool(_CLINICAL_NOUN_RE.search(s))
        has_finding = bool(_FINDING_RESULT_VERB_RE.search(s))
        if not (has_clinical and has_finding):
            return reject("methods_procedural")
    if _META_RE.match(s):
        return reject("meta")
    if _META_OBJECTIVE_RE.search(s):
        return reject("meta_objective")
    if _GLOSSARY_RE.search(s):
        return reject("glossary")
    if _LICENSE_RE.search(s):
        return reject("license_boilerplate")
    if _JOURNAL_PAGE_HEADER_RE.match(s):
        return reject("journal_page_header_bleed")

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
    # Universal legal/IP/deontic filter (not document-specific)
    if _LEGAL_UNIVERSAL_RE.search(s):
        return reject("legal_boilerplate")

    # POS-based universal structural filters.
    # Run after all cheap string filters to minimise tagging overhead.
    try:
        from nltk import pos_tag, word_tokenize  # cached after first load
        pos_tags = pos_tag(word_tokenize(s))
    except (ImportError, LookupError):
        pos_tags = []

    if pos_tags:
        # VBG opener = preamble clause ("Noting that...", "Having regard to...")
        # Appear in legal/policy preambles across ALL document types.
        first_content = next(
            (t for _, t in pos_tags if t not in ('DT', 'IN', 'CC', 'TO', ',')), None
        )
        if first_content == 'VBG':
            return reject("vbg_preamble")

        # High proper-noun ratio = author/institution/country list
        # e.g. "Dr John Smith Director-General World Health Organization" (~80% NNP)
        nnp_count = sum(1 for _, t in pos_tags if t == 'NNP')
        if nnp_count / len(pos_tags) > 0.45:
            return reject("high_nnp_ratio")

        # Union with regex fallback: POS tagger can misparse domain-specific nouns
        # as verbs or vice versa (e.g. "accounts" → NNS instead of VBZ).
        # If either method finds a finite verb, accept it.
        has_verb_pos = any(t in {'VBZ', 'VBD', 'VBP', 'VBN', 'VB', 'MD'} for _, t in pos_tags)
        has_verb = has_verb_pos or bool(_VERB_RE.search(lower))
    else:
        # POS tagger unavailable — fall back to regex
        has_verb = bool(_VERB_RE.search(lower))

    has_punct = s[-1] in ".!?"

    if not has_punct:
        # Slide/bullet-point style: accept if long enough AND has a verb.
        if len(words) >= _MIN_NOPUNCT_WORDS and has_verb:
            pass  # continue to factual signal check
        else:
            return reject("no_terminal_punct")

    if not has_verb:
        # Metric/data-sheet style: accept "Accuracy: 95%." patterns.
        if _METRIC_RE.search(s):
            return True
        return reject("no_verb")

    # Factual signal gate — precision-first.
    # A verifiable claim must assert something measurable or relational.
    # Vague sentences ("Protein folding is complex") fail here by design —
    # they cannot be verified against specific evidence anyway.
    has_factual_signal = bool(_FACTUAL_SIGNAL_RE.search(s))
    # Fallback: a long sentence with a scientific finding verb can pass even without
    # a hard number (e.g. "Studies have consistently found that influenza spreads
    # through respiratory droplets during close-contact interactions with infected...").
    is_long_declarative = len(words) >= 15 and bool(_FINDING_VERB_RE.search(lower))
    if not has_factual_signal and not is_long_declarative:
        return reject("no_factual_signal")

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
    "disclosure_boilerplate": (
        "Content is mostly funding / conflict-of-interest / data-availability "
        "disclosure boilerplate. The document may be a cover page or back-matter "
        "section rather than a research paper."
    ),
    "bibliography_entry": (
        "Content is mostly bibliography / reference-list entries (paper titles "
        "like \"(SUSTAIN 11): a randomized, phase 3b trial\"). The document may be "
        "a reference list section rather than research-finding content."
    ),
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
    "vbg_preamble": (
        "Content is dominated by preamble clauses (\"Noting that...\", \"Having regard to...\"). "
        "This is common in legal/policy documents. The document may be a treaty, "
        "resolution, or regulatory filing rather than a research paper."
    ),
    "high_nnp_ratio": (
        "Content is dominated by proper nouns — likely author lists, "
        "institutional headers, or country/organisation catalogues."
    ),
    "no_factual_signal": (
        "Sentences lack verifiable factual content: no numbers, measurements, "
        "causal language, or comparative assertions. "
        "The document may be introductory, definitional, or purely narrative. "
        "Try uploading the methods or results section directly."
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


def _extract_claims_sync(
    citation_blocks: list[dict],
    sent_tokenize_fn,  # passed in so the caller handles the import
) -> ClaimExtractionResponse:
    """Synchronous extraction loop — runs in a thread pool via asyncio.to_thread.

    Separated from the async wrapper so POS tagging (CPU-intensive, synchronous)
    never blocks the event loop. Health checks remain responsive even for large
    documents.
    """
    all_claims: list[ExtractedClaim] = []
    seen: set[str] = set()
    rejection_counts: dict[str, int] = {}
    total_candidates = 0
    prev_sentence = ""

    for block in citation_blocks:
        text = _fix_word_boundaries(block.get("text", "").strip())
        if not text:
            continue
        try:
            sentences = sent_tokenize_fn(text)
        except Exception as exc:
            logger.warning("sent_tokenize failed for block: %s", exc)
            sentences = [s.strip() for s in text.split(".") if s.strip()]

        for sentence in sentences:
            sentence = sentence.strip()
            total_candidates += 1
            if not _is_likely_claim(sentence, rejection_counts):
                prev_sentence = sentence
                continue
            # Normalised dedup key: strip whitespace AND hyphens to fold
            # PyMuPDF line-break artifacts ("lim-ited"/"limitedby"/"limited by")
            # into the same bucket. Display still uses the original sentence.
            # Lesson 73 (2026-04-30): hyphen-line-break gap surfaced 4 dups on
            # CRISPR paper because byte-equal dedup keyed on raw lower-case.
            key = re.sub(r"[-\s]", "", sentence.lower())
            if key in seen:
                prev_sentence = sentence
                continue
            seen.add(key)
            markers = _detect_citation_markers(sentence)
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


async def extract_claims_from_blocks(
    citation_blocks: list[dict],
) -> ClaimExtractionResponse:
    """Extract factual claims from text blocks using NLTK sentence splitting.

    No LLM calls. No API dependency. Runs locally in milliseconds.

    The CPU-intensive work (sentence tokenisation + per-sentence heuristics
    including NLTK POS tagging) runs in a thread pool via asyncio.to_thread
    so the event loop stays responsive for health checks and other requests.
    """
    import asyncio

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

    # Offload CPU-intensive synchronous work to a thread pool.
    # This prevents NLTK POS tagging (called per-sentence) from blocking the
    # event loop on large documents.
    return await asyncio.to_thread(_extract_claims_sync, citation_blocks, sent_tokenize)
