# EvidenceEngine Eval Harness

A standalone CLI tool that measures the quality of EvidenceEngine's verdict classification against a hand-verified gold benchmark. No running web server, no database, and no external data downloads are required — all benchmark fixtures ship with the repository.

## Overview

The eval harness runs every case in the benchmark suite through `classify_claim()`, compares the predicted verdict against a hand-verified gold label, and produces four metrics that quantify classifier quality:

| Metric | Measures | Direction |
|--------|----------|-----------|
| `false_support_rate` | How often the model incorrectly says "supported" when truth is contradicted or insufficient | Lower is better |
| `contradiction_recall` | Fraction of genuinely contradicted claims the model correctly flags | Higher is better |
| `insufficient_support_precision` | Of everything the model calls "insufficient_support", what fraction is correct | Higher is better |
| `evidence_retrieval_recall` | Fraction of gold cases where at least one evidence span was retrieved | Higher is better |

The CI gate (`ci_gate.py`) enforces a hard threshold on `false_support_rate` — false positives on "supported" are the highest-risk failure mode in a fact-checking context.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager (`pip install uv` or `brew install uv`)
- `OPENAI_API_KEY` environment variable set
- No database or running server required

### Install

```bash
git clone <repo-url>
cd vibebuild
uv sync
export OPENAI_API_KEY=sk-...
```

### Run the full evaluation

```bash
python -m eval.cli run
```

This runs all 220 gold cases (plus 80 synthetic) through `classify_claim()` concurrently, then prints a report. Expect it to take 1–3 minutes depending on concurrency and model.

### Run with human-readable text output

```bash
python -m eval.cli run --output-format text
```

### Run with a specific model and higher concurrency

```bash
python -m eval.cli run --model gpt-4o --concurrency 10
```

### Check only the CI gate metric

```bash
python eval/ci_gate.py
```

Exits 0 if `false_support_rate` is within threshold, 1 if it exceeds the threshold, 2 if evaluation failed.

---

## CLI Reference

### `python -m eval.cli run`

Runs the full benchmark evaluation suite and prints the report.

```
usage: python -m eval.cli run [--output-format {json,text}] [--model MODEL] [--concurrency N]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-format` | `text` | `json` for machine-readable output, `text` for human-readable table |
| `--model` | `gpt-4o-mini` (runner default) | OpenAI model name for verdict classification |
| `--concurrency` | `5` | Number of concurrent `classify_claim` API calls |

**Example: text output**

```
============================================================
EvidenceEngine Eval Report
============================================================
Cases evaluated:  gold=220  synthetic=80

false_support_rate          : 0.0273  (lower is better)
contradiction_recall        : 0.8909  (higher is better)
insufficient_support_prec   : 0.8421  (higher is better)
evidence_retrieval_recall   : 1.0000  (higher is better)

Denominators:
  n_gold                    : 220
  n_gold_contradicted       : 55
  n_gold_insufficient       : 55
  n_predicted_insufficient  : 57
  n_gold_with_evidence      : 220
============================================================
```

**Example: JSON output** (`--output-format json`)

```json
{
  "metrics": {
    "false_support_rate": 0.0273,
    "contradiction_recall": 0.8909,
    "insufficient_support_precision": 0.8421,
    "evidence_retrieval_recall": 1.0,
    "n_gold": 220,
    "n_gold_contradicted": 55,
    "n_gold_insufficient": 55,
    "n_predicted_insufficient": 57,
    "n_gold_with_evidence": 220
  },
  "summary": {
    "n_gold": 220,
    "n_synthetic": 80
  }
}
```

---

### `python eval/ci_gate.py`

Threshold-enforced quality gate. Runs the evaluation and exits non-zero if the false support rate exceeds the configured threshold.

```
usage: python eval/ci_gate.py
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key for OpenAI classify_claim calls |
| `EVAL_FALSE_SUPPORT_THRESHOLD` | `0.10` | Gate fails if `false_support_rate` exceeds this value |
| `EVAL_CONCURRENCY` | `5` | Concurrent API calls during evaluation |

**Exit codes:**

| Exit Code | Meaning |
|-----------|---------|
| `0` | Gate passed — `false_support_rate <= threshold` |
| `1` | Gate failed — `false_support_rate > threshold` |
| `2` | Error loading benchmark or running evaluation |

**Example output (passing):**

```
Running EvidenceEngine eval gate (threshold: false_support_rate <= 0.1)
Evaluated 220 gold cases
  false_support_rate         = 0.0273  (threshold: 0.1)
  contradiction_recall       = 0.8909
  insufficient_support_prec  = 0.8421
  evidence_retrieval_recall  = 1.0000

GATE PASSED: false_support_rate 0.0273 <= 0.1
```

---

### `python eval/regression.py add`

Captures a production miss as a regression test case in `eval/benchmark/fixtures/regression/`. Does not modify any application code.

```
usage: python eval/regression.py add --case-file PATH
```

| Flag | Required | Description |
|------|----------|-------------|
| `--case-file` | Yes | Path to a JSON file containing a `BenchmarkCase` object |

**Case file format:**

```json
{
  "id": "regression-001",
  "verdict_class": "contradicted",
  "claim_text": "Sea levels rose 3.6mm/year from 2006-2018",
  "evidence_spans": [
    {
      "span_text": "The rate of global mean sea level rise from 2006 to 2018 was 3.9mm per year.",
      "relevance_score": 0.9,
      "rank": 1
    }
  ],
  "gold_verdict": "contradicted",
  "source": "gold",
  "notes": "Model incorrectly classified as supported; sea level value mismatch (3.6 vs 3.9 mm/year)"
}
```

**Example workflow for a production miss:**

```bash
# 1. Create the case file
cat > /tmp/regression-sea-level.json << 'EOF'
{
  "id": "regression-sea-level-001",
  "verdict_class": "contradicted",
  "claim_text": "Sea levels rose 3.6mm/year from 2006-2018",
  "evidence_spans": [
    {
      "span_text": "The rate of global mean sea level rise from 2006 to 2018 was 3.9mm per year.",
      "relevance_score": 0.9,
      "rank": 1
    }
  ],
  "gold_verdict": "contradicted",
  "source": "gold",
  "notes": "Model incorrectly classified as supported; sea level value mismatch"
}
EOF

# 2. Add the regression case
python eval/regression.py add --case-file /tmp/regression-sea-level.json

# 3. Commit to repository
git add eval/benchmark/fixtures/regression/
git commit -m "test(regression): add case regression-sea-level-001"
```

Regression cases are automatically picked up by the eval runner on the next run.

---

## Metrics Reference

All four metrics operate exclusively on **gold-sourced** cases (`"source": "gold"`). Synthetic cases are excluded from metric computation. Zero denominators return `1.0` (vacuous truth — not treated as a failure).

### `false_support_rate`

**Formula:**

```
false_positives_supported / (n_gold_contradicted + n_gold_insufficient)
```

Where `false_positives_supported` = cases where `gold_verdict in {contradicted, insufficient_support}` AND `predicted_verdict == "supported"`.

**Interpretation:** The fraction of non-supported gold cases that the model incorrectly labeled "supported". This is the highest-risk failure mode — a user would see a false endorsement of a claim the evidence contradicts. The CI gate enforces an upper bound on this metric.

**Target:** < 0.05 in production. CI default threshold: 0.10.

---

### `contradiction_recall`

**Formula:**

```
n_correctly_contradicted / n_gold_contradicted
```

Where `n_correctly_contradicted` = cases where `gold_verdict == "contradicted"` AND `predicted_verdict == "contradicted"`.

**Interpretation:** Of all genuinely contradicted claims in the gold set, what fraction did the model correctly flag as contradicted? Low recall means the model is letting contradictions slip through as "needs_review" or "insufficient_support".

**Target:** > 0.85.

---

### `insufficient_support_precision`

**Formula:**

```
n_true_insufficient / n_predicted_insufficient
```

Where `n_true_insufficient` = cases where `gold_verdict == "insufficient_support"` AND `predicted_verdict == "insufficient_support"`.

**Interpretation:** Of all claims the model labeled "insufficient_support", what fraction was actually insufficient support in the gold set? Low precision means the model is over-labeling claims as ambiguous when they are actually supported or contradicted.

**Target:** > 0.80.

---

### `evidence_retrieval_recall`

**Formula:**

```
n_gold_cases_with_evidence / n_gold
```

Where `n_gold_cases_with_evidence` = gold cases where `evidence_span_count > 0`.

**Interpretation:** Fraction of gold cases where at least one evidence span was retrieved and passed to the classifier. Because fixture cases include pre-populated evidence spans, this metric will always be 1.0 in the eval harness. It becomes meaningful in integration tests against a live retrieval pipeline.

**Target:** 1.0 (all gold cases have evidence in fixtures).

---

## Benchmark Data

All benchmark data ships with the repository. No external downloads are required.

### Location

```
eval/benchmark/fixtures/
├── gold/                          # Hand-verified ground truth (55 per class = 220 total)
│   ├── supported.json             # 55 cases where claim is fully supported by evidence
│   ├── contradicted.json          # 55 cases where evidence contradicts the claim
│   ├── insufficient_support.json  # 55 cases where evidence is present but insufficient
│   └── needs_review.json          # 55 cases requiring human judgment (ambiguous)
├── synthetic/                     # LLM-generated expansion cases (20 per class = 80 total)
│   ├── supported.json
│   ├── contradicted.json
│   ├── insufficient_support.json
│   └── needs_review.json
└── regression/                    # Production miss cases (added via regression.py)
    └── README.md                  # Instructions for adding regression cases
```

### Case Format

Each file contains a JSON array of `BenchmarkCase` objects. A complete example:

```json
{
  "id": "gold-sup-001",
  "verdict_class": "supported",
  "claim_text": "Global mean surface temperature increased by 1.1°C above pre-industrial levels by 2022.",
  "evidence_spans": [
    {
      "span_text": "The global mean surface temperature in 2022 was approximately 1.15°C above the 1850–1900 pre-industrial baseline.",
      "relevance_score": 0.92,
      "rank": 1
    }
  ],
  "gold_verdict": "supported",
  "source": "gold",
  "notes": "Direct numeric confirmation"
}
```

**Field descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique case identifier (e.g., `gold-sup-001`, `regression-001`) |
| `verdict_class` | enum | The intended verdict class for this case |
| `claim_text` | string | The claim to be evaluated |
| `evidence_spans` | array | Pre-retrieved evidence spans passed to the classifier |
| `evidence_spans[].span_text` | string | Text of the evidence passage |
| `evidence_spans[].relevance_score` | float [0,1] | Retrieval relevance score |
| `evidence_spans[].rank` | int | Retrieval rank (1 = most relevant) |
| `gold_verdict` | enum | Hand-verified correct verdict |
| `source` | enum | `"gold"` or `"synthetic"` — controls which metrics the case contributes to |
| `notes` | string? | Optional rationale or annotation |

### Gold vs Synthetic

**Gold cases** (`"source": "gold"`) are the 220 hand-verified cases that drive all four metrics. They are the authoritative benchmark. The CI gate uses only gold metrics.

**Synthetic cases** (`"source": "synthetic"`) are LLM-generated cases added to increase coverage across edge cases and topic diversity. They do NOT contribute to `false_support_rate`, `contradiction_recall`, or `insufficient_support_precision`. They are included in `evidence_retrieval_recall` via the full result set — though in practice their evidence spans are always populated.

**Regression cases** are individual JSON files in `regression/`. The loader picks them up automatically. Each regression case should have `"source": "gold"` so it contributes to gold metrics — the regression directory is for capturing production misses that should become permanent test cases.

### Adding Regression Cases

See `python eval/regression.py add --help` and the [CLI Reference](#python-evalregressionpy-add) above.

---

## CI Integration

### GitHub Actions

The workflow at `.github/workflows/eval-gate.yml` runs on every push and pull request to `main`:

```yaml
- name: Run eval quality gate
  run: uv run python eval/ci_gate.py
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    EVAL_FALSE_SUPPORT_THRESHOLD: "0.10"
    EVAL_CONCURRENCY: "3"
```

The workflow fails automatically if `ci_gate.py` exits with code 1 (false support rate exceeded). This blocks merges that degrade classifier quality below the threshold.

The full workflow also runs database migrations before the gate — this is required because `classify_claim()` logs results to the database in integration mode. In pure eval mode (using the fixture data), the database steps are not strictly needed but are included for workflow consistency.

### Configuring the Threshold

Set `EVAL_FALSE_SUPPORT_THRESHOLD` to a float between 0 and 1:

```bash
# Strict: fail if more than 5% of non-supported cases are mislabeled "supported"
export EVAL_FALSE_SUPPORT_THRESHOLD=0.05
python eval/ci_gate.py

# Lenient: allow up to 15% (not recommended for production)
export EVAL_FALSE_SUPPORT_THRESHOLD=0.15
python eval/ci_gate.py
```

The default threshold (when the variable is not set) is `0.10` — defined as `DEFAULT_THRESHOLD` in `eval/ci_gate.py`.

### Secrets Required

Add `OPENAI_API_KEY` to your GitHub repository secrets at:
`Settings > Secrets and variables > Actions > New repository secret`

---

## Architecture

```
eval/
├── __init__.py              # Package root
├── __main__.py              # Enables: python -m eval (delegates to cli.main)
├── cli.py                   # argparse CLI entry point — build_parser(), main()
├── runner.py                # Async evaluation loop — run_evaluation(model, concurrency)
├── ci_gate.py               # Threshold-enforced exit code gate — reads EVAL_* env vars
├── regression.py            # Add regression cases from production misses
├── benchmark/
│   ├── __init__.py
│   ├── schema.py            # BenchmarkCase, BenchmarkSuite, EvidenceSpanFixture (Pydantic)
│   ├── loader.py            # load_benchmark() -> BenchmarkSuite (gold + synthetic + regression)
│   └── fixtures/            # Benchmark data files (JSON arrays of BenchmarkCase)
│       ├── gold/            # 220 hand-verified cases (55 per verdict class)
│       ├── synthetic/       # 80 LLM-generated expansion cases
│       └── regression/      # Individual production miss files
└── metrics/
    ├── __init__.py
    ├── types.py             # BenchmarkResult, EvalReport (Pydantic dataclasses)
    └── compute.py           # compute_metrics(results) -> EvalReport (pure function, no I/O)
```

### Data flow

```
load_benchmark()          BenchmarkSuite (gold + synthetic)
     |
     v
run_evaluation()          calls classify_claim() for each case concurrently
     |
     v
compute_metrics()         pure function: list[BenchmarkResult] -> EvalReport
     |
     v
cli.py / ci_gate.py       formats output or enforces threshold
```

### Key design decisions

- **No server required:** `classify_claim()` is called directly as a library function — the FastAPI app does not need to be running.
- **Gold/synthetic separation enforced at metric layer:** `compute_metrics()` filters to `source == "gold"` before computing any rate. Synthetic cases cannot inflate or deflate metrics.
- **Pure metric functions:** `compute.py` has no I/O, no side effects, and is fully unit-tested. The runner handles all async coordination and result collection.
- **Regression cases are first-class:** Files in `regression/` are loaded alongside gold fixtures and can have `source: "gold"` — they permanently expand the gold benchmark.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'eval'`**

Run from the project root (`vibebuild/`), not from inside the `eval/` directory:

```bash
cd vibebuild
python -m eval.cli run     # correct
# NOT: cd eval && python -m cli run
```

**`AuthenticationError: No API key provided`**

Set the environment variable before running:

```bash
export OPENAI_API_KEY=sk-...
python -m eval.cli run
```

**`RateLimitError` during evaluation**

Reduce concurrency:

```bash
python -m eval.cli run --concurrency 2
```

**Evaluation runs but `false_support_rate` is 1.0**

This usually means the model returned an unexpected response format. Check that your `OPENAI_API_KEY` is valid and has access to the model specified with `--model`.
