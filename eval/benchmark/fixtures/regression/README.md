# Regression Cases

This directory holds individual BenchmarkCase JSON files capturing production misses.

Each file is a single JSON object matching the BenchmarkCase schema.
Files are auto-discovered by loader.load_benchmark() and added to the gold set.

## Adding a new regression case

1. Create a JSON file following the BenchmarkCase schema
2. Set "source": "gold" and "id": "regression-NNN"
3. Run: `python eval/regression.py add --case-file path/to/your-case.json`
4. Commit the resulting file in this directory
5. No application code changes needed

## Naming convention

regression-{NNN}.json  (e.g. regression-001.json, regression-042.json)

## Case schema reference

```json
{
  "id": "regression-001",
  "verdict_class": "contradicted",
  "claim_text": "The claim text that was misclassified in production.",
  "evidence_spans": [
    {
      "span_text": "The evidence span that was retrieved.",
      "relevance_score": 0.85,
      "rank": 1
    }
  ],
  "gold_verdict": "contradicted",
  "source": "gold",
  "notes": "Production miss: model predicted supported. Discovered YYYY-MM-DD."
}
```
