# Evaluation

Scripts that measure the verdict step. No results are committed: `eval/results/` is gitignored.

## Gold benchmark

`benchmark/fixtures/gold/` holds 220 labeled cases, 55 per verdict. They are climate and energy claims, not biomedical, so treat the numbers as a smoke test rather than a measure of biomedical accuracy.

```bash
uv run python -m eval.run_nli_benchmark          # accuracy, false-support rate, contradiction recall
uv run python -m eval.run_nli_threshold_sweep    # try entailment and contradiction thresholds
```

Both run the NLI model directly, with no database, server or LLM.

## SciFact

`scifact/` runs retrieval and the verdict on the SciFact dev set (300 biomedical claims) and reports macro F1, per-class F1, calibration (ECE) and retrieval recall. The data is fetched on first use into `eval/data/scifact/`, which is not tracked in git. SciFact is CC BY-NC 2.0.

```bash
uv run python -m eval.scifact run --split dev --max-cases 10   # smoke test
uv run python -m eval.scifact run --split dev                   # all 300 claims
uv run python -m eval.scifact metrics eval/results/scifact-dev300.jsonl
```

`scifact/finetune.py` trains the SciFact copy of the NLI model. Point `NLI_MODEL_PATH` at its output (`./checkpoints/scifact-nli` by default) to use it.

More detail in [scifact/README.md](scifact/README.md).
