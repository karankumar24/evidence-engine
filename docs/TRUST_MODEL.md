# EvidenceEngine Trust Model

## Overview

The trust model defines how EvidenceEngine assigns verdicts to claims. The
design philosophy is conservative: a claim is classified as `supported` only
when the retrieved evidence directly and unambiguously confirms the specific
value or fact stated. Any ambiguity, partial coverage, or low model confidence
routes to `insufficient_support` or `needs_review` rather than silently
asserting support. This conservatism is intentional — a false `supported`
verdict erodes reviewer trust in the entire system.

The trust model is enforced at three points in the pipeline: (1) the
zero-evidence shortcut, which assigns `insufficient_support` before any LLM
call when no evidence spans exist; (2) the system prompt disambiguation
instructions, which explicitly teach the LLM to reject partial evidence as
support; and (3) the confidence routing mechanism, which overrides the LLM's
verdict to `needs_review` whenever the model's own confidence falls below a
configurable threshold. Together these mechanisms ensure that the most common
failure modes in automated fact-checking — hallucinated support, topical
conflation, and low-confidence acceptance — are caught before a verdict is
written.

---

## The Four Verdicts

### Verdict Taxonomy

---

**`supported`**

*Meaning:* The retrieved evidence directly and unambiguously confirms the claim.
The specific value, fact, or statement in the claim appears in the source
document — not merely a related topic or surrounding context.

*Example:*
- Claim: "CO2 reached 421 ppm at Mauna Loa in May 2023."
- Evidence: "Monthly mean CO2 was 421.47 ppm at Mauna Loa Observatory in May 2023."
- Verdict: `supported` — the evidence quotes the specific measurement.

*Decision rule:* Strong lexical and semantic alignment with the claim's specific
assertion; model `confidence_score ≥ threshold` (default 0.7).

---

**`contradicted`**

*Meaning:* The retrieved evidence directly and explicitly contradicts the claim.
A different value or an opposing fact appears in the source document on the same
specific topic.

*Example:*
- Claim: "Global mean temperature rose 1.5 °C above pre-industrial levels by 2022."
- Evidence: "The 2022 global mean surface temperature was approximately 1.15 °C
  above the 1850–1900 baseline."
- Verdict: `contradicted` — the evidence states a materially different value for
  the same measurement.

*Decision rule:* Evidence states a directly contrary value or fact on the same
specific topic; model `confidence_score ≥ threshold`.

---

**`insufficient_support`**

*Meaning:* The evidence is topically related but does not directly confirm or
contradict the claim. This covers: no evidence retrieved, evidence from a
wrong source, evidence too vague to confirm or deny, and evidence that addresses
the general topic but not the specific value or assertion.

*Example:*
- Claim: "Arctic sea ice extent reached a record low in September 2022."
- Evidence: "Arctic sea ice has declined at approximately 13% per decade since
  1979 relative to the 1981–2010 average."
- Verdict: `insufficient_support` — the evidence confirms a long-term trend but
  does not confirm or deny the specific September 2022 record claim.

*Decision rule:* See **The Weak-Evidence Rule** below.

---

**`needs_review`**

*Meaning:* The model's confidence in its verdict is below the configured
threshold, OR the claim's citation reference could not be resolved to a source
document. The claim is routed to a human reviewer rather than accepting an
uncertain or unverifiable verdict.

*Example:*
- LLM outputs `verdict_type='supported'`, `confidence_score=0.61`.
- Threshold is 0.70.
- `apply_confidence_threshold` overrides `verdict_type` to `needs_review`.

*Decision rule:* See **Confidence Routing** below.

---

## The Weak-Evidence Rule

This is the most critical trust model decision. It deliberately breaks from
the naive "found some evidence → call it supported" default.

### Why partial evidence yields `insufficient_support`, not `supported`

**1. Topical relevance is not evidentiary support.**

BM25 retrieval returns documents about the same topic — not documents that
confirm the specific claim. A paper about Arctic ice decline is topically
relevant to any claim about ice extent, but topical relevance does not confirm
a specific measured value or event. The retrieval stage finds the best
*candidate* evidence; the classification stage determines whether that candidate
is actually *confirming*.

**2. The asymmetric cost of false support.**

Incorrectly labelling a claim as `supported` when evidence only partially
addresses it is a trust failure: a reviewer acting on that verdict may accept a
claim that has not been verified. Labelling it `insufficient_support` sends it
to the reviewer — the conservative default. The cost of routing a correctly
supported claim to review is extra reviewer time. The cost of marking an
unsupported claim as supported is a false assurance.

**3. Evidence specificity matters.**

"Sea ice has declined since 1979" does not confirm "sea ice extent was 4.67
million km² in September 2022." The claim requires a specific measurement; the
evidence provides only a trend. These are categorically different statements,
even though both are about the same phenomenon.

### How the weak-evidence rule is enforced

Three mechanisms work together:

**Mechanism 1 — Zero-evidence shortcut**

If no `EvidenceSpan` rows are retrieved for a claim (the BM25 + cross-encoder
pipeline returned nothing above the relevance threshold), the pipeline
immediately assigns `insufficient_support` with `model_name='none'` and skips
the LLM call entirely. The LLM never sees a "no evidence" case and cannot
hallucinate support from an empty context window.

```
0 evidence spans → insufficient_support  (no LLM call)
```

**Mechanism 2 — System prompt disambiguation**

The LLM classification prompt includes an explicit instruction:

> "If evidence only partially addresses the claim, or is topically related but
> does not directly confirm the specific value or fact claimed, return
> `insufficient_support`. Do NOT return `supported` unless the evidence
> directly and unambiguously confirms the claim."

This instruction is present on every classification call. The LLM is not left
to infer the distinction between "relevant" and "confirming" — the distinction
is stated explicitly in the prompt.

**Mechanism 3 — Field ordering in `VerdictClassificationResponse`**

The Pydantic response schema used with `beta.chat.completions.parse` orders its
fields as:

```
reasoning → verdict_type → confidence_score
```

This ordering forces the model to produce chain-of-thought reasoning *before*
it assigns a label and before it assigns a confidence score. Because language
models generate tokens left-to-right, the reasoning field must be populated
first — reducing shortcut label assignment where the model emits a label before
thinking through the evidence.

---

## Confidence Routing

### How low-confidence verdicts become `needs_review`

The classification pipeline applies confidence routing as a post-processing step
after every LLM call:

1. The LLM returns a `VerdictClassificationResponse` containing:
   - `reasoning` (free text)
   - `verdict_type` (`supported` | `contradicted` | `insufficient_support`)
   - `confidence_score` (float in `[0.0, 1.0]`)

2. `apply_confidence_threshold(verdict_type, confidence_score, threshold)` is
   called. This is a pure, synchronous function — it does not call the LLM and
   has no side effects.

3. If `confidence_score < threshold`:
   - `verdict_type` is overridden to `needs_review`
   - The original `reasoning` text and original `confidence_score` are
     **preserved** in the `Verdict` row so the reviewer can see what the model
     was thinking and why it was uncertain

4. The `Verdict` row is written with the final (possibly overridden)
   `verdict_type`.

### What the threshold controls

| Setting | Behavior |
|---------|----------|
| Default threshold | 0.7 (70% model confidence) |
| Configuration | `VERDICT_NEEDS_REVIEW_THRESHOLD` environment variable |
| Below threshold | Verdict (any label) → `needs_review` |
| Above threshold | Model's verdict label accepted as-is |

The threshold applies to *any* verdict label — not just `supported`. A claim
the model classifies as `contradicted` with `confidence_score=0.55` is also
overridden to `needs_review`, because the system is not confident enough in the
contradiction to present it without human review.

`needs_review` does **not** mean "wrong." It means "uncertain enough to warrant
human eyes." Reviewers can see the original `reasoning` and `confidence_score`
and decide whether the verdict stands or should be overridden.

### Unresolvable citation anchors → `needs_review`

Separately from the confidence threshold: if a claim's citation reference `[N]`
cannot be resolved to any uploaded `SourceDocument`, the pipeline assigns
`needs_review` with `model_name='none'` — no LLM call is made.

The rationale: without knowing which source to evaluate against, any verdict
would be speculation. `insufficient_support` would be misleading (it implies
evidence was sought but found wanting); `needs_review` correctly signals that a
human must locate the source before any verdict is possible.

```
unresolvable CitationAnchor → needs_review  (no LLM call)
```

---

## Reviewer Override

When a human reviewer examines a verdict in the dashboard, they can:

- **Accept** — confirm the verdict as correct
- **Override** — replace the verdict with a different label
- **Flag** — mark the claim for escalation or further investigation

Each action writes a `ReviewDecision` row that records the reviewer's judgment,
timestamp, and (for overrides) the new verdict label. Crucially, the original
`Verdict` row is **never modified**. The provenance chain is preserved: the
original LLM verdict, the original confidence score, and the reviewer's
decision are all stored as separate records. An audit trail of every human
intervention is maintained without overwriting the system's output.

This design means the dashboard can show both "what the system concluded" and
"what the reviewer decided" side by side, and historical runs are fully
reproducible.

---

## Trust Hierarchy Summary

| Situation | Verdict | Rationale |
|-----------|---------|-----------|
| Strong, specific evidence directly confirms the claim | `supported` | Direct confirmation; confidence ≥ threshold |
| Evidence states a different value or contrary fact | `contradicted` | Direct contradiction; confidence ≥ threshold |
| No evidence spans retrieved for the claim | `insufficient_support` | Cannot claim support without evidence (zero-evidence shortcut) |
| Evidence is topically related but does not confirm the specific claim | `insufficient_support` | Partial evidence is not support (prompt disambiguation) |
| Model produces any verdict with confidence < threshold | `needs_review` | Uncertain — route to human (confidence routing) |
| Citation `[N]` cannot be resolved to a source document | `needs_review` | No source to evaluate against (unresolvable anchor shortcut) |

### Quick self-test

*"A claim has two evidence spans that are about the right topic but don't quote
the specific number — what verdict does it get?"*

Answer: `insufficient_support`. The spans are topically relevant (BM25 and the
cross-encoder found them), but the LLM classification prompt requires direct
confirmation of the specific value. Topical relevance without specific
confirmation → `insufficient_support`, not `supported`.

---

## Related Documents

- See [docs/ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline stages and data
  model — including how `EvidenceSpan` rows are created, how `CitationAnchor`
  rows are resolved, and how `VerdictEvidence` links verdicts to their evidence.
- See [eval/README.md](../eval/README.md) for how verdict quality is measured —
  including the false-positive and false-negative rate metrics that quantify
  trust model performance on benchmark cases.
