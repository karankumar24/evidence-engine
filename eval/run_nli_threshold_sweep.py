"""NLI threshold sweep — find best (entail_thresh, contra_thresh) for the
internal gold benchmark.

Why:
    Filter raise from 0.86/0.80 → 0.92/0.85 on Apr 26 was suspected to cause
    accuracy regression (64.5% → 61.8%). Memory record #6664 traced the
    regression to that change. Need empirical data, not guess.

Strategy:
    Per-span NLI inference is expensive (~5-10min for 220 cases). Threshold
    aggregation is microseconds. So compute per-span probs ONCE, then sweep
    the threshold combos cheaply.

Usage:
    python -m eval.run_nli_threshold_sweep
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


def verdict_for_thresholds(
    p_entail: float,
    p_neutral: float,
    p_contra: float,
    *,
    min_conf: float,
    entail_thresh: float,
    contra_thresh: float,
) -> tuple[str, float]:
    """Pure verdict logic. Mirrors nli_probs_to_verdict but takes thresholds
    as args instead of reading from settings — so we can sweep without
    monkeypatching."""
    max_p = max(p_entail, p_neutral, p_contra)
    if max_p < min_conf:
        return "needs_review", max_p
    if p_entail >= entail_thresh:
        return "supported", p_entail
    if p_contra >= contra_thresh:
        return "contradicted", p_contra
    return "insufficient_support", max_p


def evaluate(per_case_aggs: list[dict], entail: float, contra: float, min_conf: float) -> dict:
    """Compute metrics for one threshold combo. per_case_aggs is the
    pre-computed per-case (gold, p_entail, p_neutral, p_contra) tuples."""
    results = []
    for case in per_case_aggs:
        if case["empty_spans"]:
            predicted, conf = "insufficient_support", 0.0
        elif case["all_invalid"]:
            predicted, conf = "needs_review", 0.0
        else:
            predicted, conf = verdict_for_thresholds(
                case["p_entail"], case["p_neutral"], case["p_contra"],
                min_conf=min_conf, entail_thresh=entail, contra_thresh=contra,
            )
        results.append({
            "gold": case["gold"], "predicted": predicted, "correct": case["gold"] == predicted,
        })
    total = len(results)
    correct = sum(r["correct"] for r in results)
    overall_acc = correct / total

    # 3-class accuracy excludes needs_review
    three_class = [r for r in results if r["gold"] != "needs_review"]
    three_class_acc = (
        sum(r["correct"] for r in three_class) / len(three_class)
        if three_class else 0.0
    )

    false_supports = sum(
        1 for r in results if r["predicted"] == "supported" and r["gold"] != "supported"
    )
    non_sup_total = sum(1 for r in results if r["gold"] != "supported")
    fsr = false_supports / non_sup_total if non_sup_total else 0.0

    contra_cases = [r for r in results if r["gold"] == "contradicted"]
    contra_recall = (
        sum(1 for r in contra_cases if r["predicted"] == "contradicted") / len(contra_cases)
        if contra_cases else 0.0
    )

    return {
        "overall_acc": overall_acc,
        "three_class_acc": three_class_acc,
        "false_support_rate": fsr,
        "contradiction_recall": contra_recall,
        "n_total": total,
        "n_correct": correct,
    }


def run() -> None:
    from evidenceengine.classification.nli_classifier import (
        aggregate_nli,
        nli_probs_for_pair,
    )
    from evidenceengine.core.config import settings

    cases = load_cases()
    print(f"Loaded {len(cases)} gold cases")
    print(f"Current settings: entail={settings.nli_entailment_supported_threshold} "
          f"contra={settings.nli_contradiction_contradicted_threshold} "
          f"min_conf={settings.nli_min_confidence_for_verdict}\n")

    # Stage 1: compute per-case aggregated (max) probs ONCE.
    print("Stage 1/2: computing per-span NLI (this is the slow part)...")
    per_case_aggs: list[dict] = []
    for i, case in enumerate(cases, 1):
        claim = case["claim_text"]
        spans = case.get("evidence_spans") or []
        gold = case["gold_verdict"]

        if not spans:
            per_case_aggs.append({
                "gold": gold, "empty_spans": True, "all_invalid": False,
                "p_entail": 0.0, "p_neutral": 0.0, "p_contra": 0.0,
            })
        else:
            per_span = [nli_probs_for_pair(s["span_text"], claim) for s in spans]
            valid = [p for p in per_span if p is not None]
            if not valid:
                per_case_aggs.append({
                    "gold": gold, "empty_spans": False, "all_invalid": True,
                    "p_entail": 0.0, "p_neutral": 0.0, "p_contra": 0.0,
                })
            else:
                pe, pn, pc = aggregate_nli(valid)
                per_case_aggs.append({
                    "gold": gold, "empty_spans": False, "all_invalid": False,
                    "p_entail": pe, "p_neutral": pn, "p_contra": pc,
                })

        if i % 20 == 0:
            print(f"  [{i}/{len(cases)}]")

    # Stage 2: sweep threshold combos.
    print("\nStage 2/2: sweeping thresholds...")
    entail_grid = [0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]
    contra_grid = [0.75, 0.78, 0.80, 0.82, 0.85, 0.88]
    min_conf = settings.nli_min_confidence_for_verdict

    rows = []
    for et in entail_grid:
        for ct in contra_grid:
            m = evaluate(per_case_aggs, et, ct, min_conf)
            rows.append({"entail": et, "contra": ct, **m})

    # Print table sorted by 3-class accuracy desc, tiebreak by lower FSR.
    rows.sort(key=lambda r: (-r["three_class_acc"], r["false_support_rate"]))

    print(f"\n{'='*80}")
    print(f"SWEEP RESULTS — {len(rows)} combos, sorted by 3-class accuracy desc")
    print(f"{'='*80}")
    print(f"{'entail':>7} {'contra':>7} {'overall':>9} {'3-class':>9} {'FSR':>8} {'cont_R':>8}")
    for r in rows[:15]:
        print(f"{r['entail']:>7.2f} {r['contra']:>7.2f} "
              f"{r['overall_acc']:>9.1%} {r['three_class_acc']:>9.1%} "
              f"{r['false_support_rate']:>8.1%} {r['contradiction_recall']:>8.1%}")

    print(f"\n{'='*80}")
    best = rows[0]
    current_pair = (
        settings.nli_entailment_supported_threshold,
        settings.nli_contradiction_contradicted_threshold,
    )
    cur_row = next(
        (r for r in rows if r["entail"] == current_pair[0] and r["contra"] == current_pair[1]),
        None,
    )
    print(f"BEST:    entail={best['entail']:.2f} contra={best['contra']:.2f} "
          f"3class={best['three_class_acc']:.1%} FSR={best['false_support_rate']:.1%}")
    if cur_row:
        print(f"CURRENT: entail={cur_row['entail']:.2f} contra={cur_row['contra']:.2f} "
              f"3class={cur_row['three_class_acc']:.1%} FSR={cur_row['false_support_rate']:.1%}")
        delta_3c = best["three_class_acc"] - cur_row["three_class_acc"]
        delta_fsr = best["false_support_rate"] - cur_row["false_support_rate"]
        print(f"DELTA:   3class={delta_3c:+.1%} FSR={delta_fsr:+.1%}")
    print(f"{'='*80}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    run()
