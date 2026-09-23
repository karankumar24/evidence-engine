# Trust Model

This document explains the rules that govern when EvidenceEngine assigns a verdict and how confident it is allowed to be. The short version: **the system is biased toward saying "needs review" rather than risking a false `supported`.**

---

## The four verdicts

| Verdict | Meaning |
|---|---|
| `supported` | The cited evidence directly entails the claim. NLI entailment probability cleared the 0.92 threshold. |
| `contradicted` | The cited evidence directly contradicts the claim. NLI contradiction probability cleared the 0.75 threshold. |
| `insufficient_support` | Evidence was retrieved but does not entail or contradict the claim with enough confidence. The claim is plausible but not justified by the cited material. |
| `needs_review` | The system declined to commit. This is the conservative default whenever something is uncertain: confidence below threshold, no resolvable citation anchor, retrieval failure, or an edge case the heuristics do not handle. |

A reviewer is expected to look at every verdict, but the four bins exist so you can triage: contradicted first (potentially wrong claims), then needs_review (system was uncertain), then insufficient_support (nothing clearly for or against), then supported (system was confident, spot-check the loud ones).

---

## NLI thresholds

The verdict comes from `cross-encoder/nli-deberta-v3-small` fine-tuned on SciFact. The model produces `(p_entail, p_contradict, p_neutral)` for each claim-vs-evidence pair. Per-span scores are aggregated via `nli_probs_to_verdict()` in `src/evidenceengine/classification/nli_classifier.py`.

The thresholds (locked by `tests/test_config.py` invariants, defined in `src/evidenceengine/core/config.py`):

```
nli_entailment_supported_threshold        = 0.92
nli_contradiction_contradicted_threshold  = 0.75
nli_tiebreaker_threshold                  = 0.65
nli_min_confidence_for_verdict            = 0.50
```

Band invariant: `0 < nli_min_confidence_for_verdict < nli_tiebreaker_threshold < nli_entailment_supported_threshold < 1` and `0 < nli_tiebreaker_threshold < nli_contradiction_contradicted_threshold < 1`.

### Why 0.92 entailment

DeBERTa-v3 is overconfident on this calibration band by approximately 8 percentage points. A 0.85 entailment score on this model corresponds in practice to around 75% true entailment in the SciFact dev split. Setting the supported threshold at 0.92 closes that gap. The cost is a higher rate of `needs_review` assignment on borderline claims; the benefit is a lower false-support rate.

### Why 0.75 contradiction (was 0.85)

Lowered after a threshold sweep (`eval/run_nli_threshold_sweep.py`) on the gold set in `eval/benchmark/fixtures/gold/`, which is climate and energy claims rather than biomedical. Lowering the threshold improved 3-class accuracy and contradiction recall without raising the false-support rate. Specific deltas are not quoted here because that set is not biomedical (see `CHALLENGES.md`). The asymmetry between the entailment threshold (0.92) and contradiction threshold (0.75) is intentional: contradiction is harder to fake on this dataset, so the system can be less stingy without trust regression.

### When `needs_review` is used

`needs_review` comes from rules, not from the model's score: a claim whose citations match none of your uploads, or a model that fails to load. There is a floor, `nli_min_confidence_for_verdict = 0.50`, below which a verdict would become `needs_review`, but in practice the model's top probability is always above it.

---

## The self-verify cap

When a user uploads a report without any cited sources, the retriever pulls evidence from other paragraphs of the *same document*. This is useful (papers often contradict themselves or make unsupported assertions internally) but it is not a substitute for verifying citations against the actual cited papers.

The self-verify trust guard, in `src/evidenceengine/classification/pipeline.py`:

```python
self_verify_mode = bool(claim.evidence_spans) and all(
    span.source_document_id == claim.source_document_id
    for span in claim.evidence_spans
)

cap = settings.self_verify_supported_cap   # 0.80
if (
    self_verify_mode
    and classification.verdict_type == "supported"
    and classification.confidence_score > cap
):
    classification = VerdictClassificationResponse(
        verdict_type="supported",
        confidence_score=cap,
        reasoning=f"[Self-verify cap {cap:.2f}] {classification.reasoning}",
    )
```

Three things to notice:

1. **The cap applies only to `supported` verdicts.** `contradicted` and `insufficient_support` are not capped. The reasoning: an internal contradiction inside a paper is a meaningful finding even when the source is the same document, so we should not gaslight the user out of it. A `supported` verdict, by contrast, becomes circular if the system is using the paper's own claims to justify the paper's own claims.

2. **Detection is provenance-based, not mode-flag-based.** The system checks whether every retrieved evidence span came from the same `source_document_id` as the claim. This means a multi-document upload where the retriever happened to find no relevant cited evidence and fell back to report-internal spans would also trigger the cap. This is intended: if the only evidence is from the same paper, treat it as self-verify regardless of how we got there.

3. **The 0.80 cap is a hard ceiling on the displayed confidence, not a re-classification.** A claim with raw entailment 0.95 in self-verify mode displays as 80%, with the reasoning string prefixed `[Self-verify cap 0.80]`. The reviewer can see the cap was applied.

The UI in `src/evidenceengine/templates/upload.html` discloses this: "In self-verify mode, confidence on Supported verdicts is capped at 80%."

---

## Why weak evidence forces `insufficient_support`

If the retriever returns evidence spans but none of them clear the entailment or contradiction thresholds, the verdict is `insufficient_support`. Not `needs_review`, not a low-confidence `supported`. Specifically:

- `supported` requires `p_entail >= 0.92` on the best aggregated score.
- `contradicted` requires `p_contradict >= 0.75`.
- Anything else with non-empty evidence becomes `insufficient_support`.
- Empty evidence (zero spans retrieved) becomes `insufficient_support` directly, bypassing the model.

This is the central conservative rule. The system is allowed to say "I retrieved passages but they do not actually justify this claim." It is not allowed to round up a 0.65 entailment to a `supported` verdict.

The cost of this rule is that `insufficient_support` is sometimes assigned to claims that a careful human reader would call supported, because the retriever picked weak passages or the NLI model was conservative. The benefit is that a `supported` verdict on the dashboard is a high-precision signal: the model was confident AND the evidence came from a real cited source AND the confidence cleared a 0.92 threshold.

---

## Citation anchor handling

Each claim sentence is scanned for citation markers (Vancouver `[1]`, APA `(Smith et al., 2023)`). For each marker, the resolver attempts to match it to an uploaded `SourceDocument`:

- **Resolved:** Retrieval is scoped to that specific cited paper. This is Mode B1.
- **Unresolved (anchor exists, no matching upload):** The system writes a `CitationAnchor` row with `status='unresolvable_anchor'`, marks the claim `needs_review`, and skips broad retrieval to avoid pulling spurious evidence from unrelated cited papers. This is Mode B2.
- **No anchor:** Falls through to broad multi-document retrieval across all uploaded cited papers (Mode B3) or to self-verify (Mode A) if no cited papers were uploaded.

Unresolved anchors are never silently dropped. The `unresolvable_anchor` status surfaces on the dashboard so a reviewer can see "this claim cited reference [4], but you didn't upload that paper."

---

## What is NOT in the trust model

Removed from earlier revisions:

- **`apply_confidence_threshold` / LLM-primary path.** Deleted with the Phase A refactor. The trust model is now NLI-only.
- **Retry-on-low-confidence.** No automatic re-classification with different parameters. A low-confidence verdict stays low-confidence and routes to `needs_review`.
- **Cross-claim consistency.** Each claim is classified independently. The system does not detect "claim A and claim B contradict each other within the same paper." That is left to the reviewer.

---

## Summary

A `supported` verdict on the EvidenceEngine dashboard means:

1. The retriever found at least one evidence span.
2. At least one of those spans came from a real cited source (or, in self-verify mode, the same document with the cap applied).
3. The NLI model assigned `entailment` probability of at least 0.92 to the best aggregated claim-vs-evidence pair.
4. If the only evidence was from the same document as the claim, the displayed confidence is capped at 0.80.

A reviewer who clicks the verdict can see all four pieces: the confidence score, the verbatim quote from the source, the source document name, and (if applicable) the `[Self-verify cap]` prefix on the reasoning. The provenance is the contract. The number is meaningless without the quote.
