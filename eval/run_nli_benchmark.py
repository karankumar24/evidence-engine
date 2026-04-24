"""Standalone NLI benchmark — runs cross-encoder/nli-deberta-v3-small on gold fixtures.

No LLM, no database, no HTTP. Loads the gold fixture JSONs and classifies
each case directly through nli_probs_for_pair + nli_probs_to_verdict.

Usage (inside the Docker container or any env with torch + transformers):
    python -m eval.run_nli_benchmark
"""
import json
import sys
from collections import Counter
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "benchmark" / "fixtures" / "gold"


def load_cases() -> list[dict]:
    cases = []
    for f in sorted(FIXTURES_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        cases.extend(data if isinstance(data, list) else [data])
    return cases


def run() -> None:
    from evidenceengine.classification.nli_classifier import (
        aggregate_nli,
        nli_probs_for_pair,
        nli_probs_to_verdict,
    )

    cases = load_cases()
    print(f"Loaded {len(cases)} gold cases\n")

    results: list[dict] = []
    for i, case in enumerate(cases, 1):
        claim = case["claim_text"]
        spans = case.get("evidence_spans") or []
        gold = case["gold_verdict"]

        if not spans:
            predicted, conf = "insufficient_support", 0.0
        else:
            per_span = [
                nli_probs_for_pair(s["span_text"], claim) for s in spans
            ]
            valid = [p for p in per_span if p is not None]
            if not valid:
                predicted, conf = "needs_review", 0.0
            else:
                agg = aggregate_nli(valid)
                predicted, conf = nli_probs_to_verdict(*agg)

        results.append({
            "id": case.get("id", f"case-{i}"),
            "gold": gold,
            "predicted": predicted,
            "correct": gold == predicted,
            "conf": round(conf, 3),
        })

        if i % 20 == 0:
            correct = sum(r["correct"] for r in results)
            print(f"  [{i}/{len(cases)}] running accuracy: {correct/i:.1%}")

    # Metrics
    correct_total = sum(r["correct"] for r in results)
    total = len(results)
    overall_acc = correct_total / total

    by_class: dict[str, dict] = {}
    for r in results:
        g = r["gold"]
        if g not in by_class:
            by_class[g] = {"tp": 0, "total": 0, "predicted_as": Counter()}
        by_class[g]["total"] += 1
        by_class[g]["predicted_as"][r["predicted"]] += 1
        if r["correct"]:
            by_class[g]["tp"] += 1

    # false_support_rate: predicted supported when gold != supported
    false_supports = sum(
        1 for r in results if r["predicted"] == "supported" and r["gold"] != "supported"
    )
    non_supported_total = sum(1 for r in results if r["gold"] != "supported")
    fsr = false_supports / non_supported_total if non_supported_total else 0.0

    # contradiction_recall
    contra_cases = [r for r in results if r["gold"] == "contradicted"]
    contra_recall = (
        sum(1 for r in contra_cases if r["predicted"] == "contradicted") / len(contra_cases)
        if contra_cases else 0.0
    )

    print(f"\n{'='*50}")
    print(f"RESULTS: {total} cases")
    print(f"{'='*50}")
    print(f"Overall accuracy:       {overall_acc:.1%}  ({correct_total}/{total})")
    print(f"False support rate:     {fsr:.1%}  (lower is better)")
    print(f"Contradiction recall:   {contra_recall:.1%}  (higher is better)")
    print(f"\nPer-class breakdown:")
    for cls, stats in sorted(by_class.items()):
        recall = stats["tp"] / stats["total"]
        top_preds = ", ".join(f"{k}:{v}" for k, v in stats["predicted_as"].most_common(3))
        print(f"  {cls:<22} recall={recall:.1%}  ({stats['tp']}/{stats['total']})  predicted_as=[{top_preds}]")
    print(f"{'='*50}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    run()
