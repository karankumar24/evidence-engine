# SciFact Evaluation Harness

Standalone evaluation pipeline for EvidenceEngine v1.2.9 on the
[SciFact dev-300 benchmark](https://huggingface.co/datasets/allenai/scifact).

## Quick Start

### Smoke test (10 claims, ~45-60 seconds on M1 MPS)
```bash
python -m eval.scifact run --split dev --max-cases 10
```

### Full evaluation (300 claims, ~25-75 minutes on M1 MPS)
```bash
python -m eval.scifact run --split dev
```

### Resume after interruption
```bash
python -m eval.scifact run --split dev --resume
```

### Compute metrics from existing JSONL (no re-run)
```bash
python -m eval.scifact metrics eval/results/scifact-dev300.jsonl
```

## Output Files

| File | Description |
|------|-------------|
| `eval/results/scifact-dev300.jsonl` | Per-claim results (one JSON line per claim) |
| `eval/results/scifact-dev300-reliability.png` | Calibration diagram (10-bin reliability curve) |
| `eval/results/scifact-dev300-reliability.csv` | Raw bin data for the diagram |

## Metrics Reported

- **macro_F1**: Unweighted mean of F1(SUPPORT) and F1(CONTRADICT).
  NEI claims (no gold evidence) are excluded per Wadden et al. 2020 convention.
- **support_F1 / contradict_F1**: Per-class precision, recall, F1.
- **ECE** (Expected Calibration Error): Weighted mean |accuracy - confidence| across 10 bins.
  Lower = better calibrated. 0.0 = perfect calibration.
- **retrieval_recall**: Fraction of SUPPORT/CONTRADICT claims where the gold
  abstract's sentences appeared in the top-5 retrieved spans.

## Reproducibility Protocol

### Model Weights (pinned)

| Model | HuggingFace ID | Commit SHA |
|-------|---------------|------------|
| NLI classifier (primary) | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | *(see `settings.nli_model` in `core/config.py`)* |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (pinned in `settings.reranker_model_revision`) |

### Dataset

- **Source**: `allenai/scifact` on HuggingFace Datasets
- **Split**: `validation` (= `claims_dev.jsonl`, 300 unique claims)
- **Cached to**: `eval/data/scifact/` (gitignored)
- **Label strings**: `"SUPPORT"`, `"CONTRADICT"`, `""` (NEI = empty evidence dict)

### Commit Hash

Record the git commit hash when running for publication:
```bash
git rev-parse HEAD
```

### Hardware

Reference run: MacBook Air M1 2020, 8 GB unified memory, macOS 14.
MPS acceleration enabled for both DeBERTa and bge-reranker-v2-m3.
Expected peak resident memory: ~3.5 GB.
Expected per-claim latency: 5-15 seconds.

### Re-run Instructions

1. Ensure model weights are cached (first run downloads ~2 GB):
   ```bash
   python -c "from evidenceengine.retrieval.reranker import get_reranker; get_reranker()"
   python -c "from evidenceengine.classification.nli_classifier import NLIClassifier; NLIClassifier()"
   ```
2. Run evaluation:
   ```bash
   python -m eval.scifact run --split dev --concurrency 1
   ```
3. Results in `eval/results/scifact-dev300.jsonl`.

## Architecture

```
SciFact claim
    |
    +- load_scifact() --> cited_doc_ids -> corpus sentences
    |
    +- BM25 (ephemeral, scoped to cited abstracts)
    |      bm25s.BM25 + query_index() -> top-10 candidates
    |
    +- bge-reranker-v2-m3 cross-encoder
    |      rerank() -> top-5 evidence spans
    |
    +- DeBERTa-v3-large NLI (nli_primary backend)
           get_backend().classify() -> verdict + confidence
```

The retrieval is scoped to `cited_doc_ids` — abstracts the claim was written about.
This mirrors the production pipeline (users upload source documents).

## Pitfalls Documented in Research

- HF `split="validation"` = dev-300 (NOT `split="dev"` — raises ValueError)
- Gold labels are `"SUPPORT"` / `"CONTRADICT"` (NOT "SUPPORTS" / "REFUTES")
- HF builder may return >300 rows; always deduplicate on `id` field
- NEI claims = empty evidence dict; excluded from macro-F1 (not counted as wrong)
- Retrieval scoped to cited_doc_ids, not all 5,183 corpus abstracts
