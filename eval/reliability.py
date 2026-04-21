"""Reliability-diagram utility: reads an eval JSONL and reports calibration.

A reliability diagram bins verdicts by predicted confidence and measures
how the model's self-reported confidence compares with actual accuracy.
A well-calibrated classifier has bin_accuracy ≈ bin_avg_confidence for
each bin. Deviation is quantified with Expected Calibration Error (ECE)
— weighted mean gap across bins.

Usage:
    python -m eval.reliability [<path-to-jsonl>]
Default path: eval/results/latest.jsonl

Zero LLM cost, runs fully locally on an existing JSONL file.

Needs the JSONL rows to carry a `confidence` field. The current runner's
PersistedCaseResult doesn't yet — if the confidence isn't present we fall
back to reporting the per-verdict accuracy table (still useful, just
without the calibration curve).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _bin_edges(n_bins: int) -> list[tuple[float, float]]:
    step = 1.0 / n_bins
    return [(i * step, (i + 1) * step) for i in range(n_bins)]


def reliability_diagram(
    rows: list[dict],
    *,
    n_bins: int = 10,
    gold_field: str = "gold_verdict",
    pred_field: str = "predicted_verdict",
    conf_field: str = "confidence",
) -> dict:
    """Compute per-bin accuracy vs confidence and Expected Calibration Error.

    Only rows with status=='classified' (if present) are included — others
    have no LLM opinion worth calibrating.
    """
    classified = [
        r for r in rows
        if (r.get("status") in (None, "classified"))
    ]
    if not classified:
        return {"bins": [], "ece": None, "n": 0, "note": "no classified rows"}

    # If confidence is missing from the JSONL, we can still report accuracy.
    have_confidence = all(conf_field in r for r in classified)

    bins_report = []
    ece = 0.0
    total = len(classified)

    for low, high in _bin_edges(n_bins):
        if have_confidence:
            in_bin = [
                r for r in classified
                if (low <= r[conf_field] < high or (high == 1.0 and r[conf_field] == 1.0))
            ]
        else:
            in_bin = []  # unreachable unless we get confidence later
        if not in_bin:
            bins_report.append({
                "range": f"[{low:.2f}, {high:.2f})",
                "n": 0,
                "accuracy": None,
                "avg_confidence": None,
                "gap": None,
            })
            continue
        n = len(in_bin)
        correct = sum(1 for r in in_bin if r[gold_field] == r[pred_field])
        accuracy = correct / n
        avg_conf = sum(r[conf_field] for r in in_bin) / n
        gap = abs(accuracy - avg_conf)
        ece += (n / total) * gap
        bins_report.append({
            "range": f"[{low:.2f}, {high:.2f})",
            "n": n,
            "accuracy": round(accuracy, 3),
            "avg_confidence": round(avg_conf, 3),
            "gap": round(gap, 3),
        })

    return {
        "n": total,
        "bins": bins_report,
        "ece": round(ece, 4) if have_confidence else None,
        "have_confidence": have_confidence,
    }


def verdict_confusion_matrix(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Per-verdict confusion: gold verdict -> predicted verdict count dict."""
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        if r.get("status") not in (None, "classified"):
            continue
        gold = r.get("gold_verdict", "?")
        pred = r.get("predicted_verdict", "?")
        out.setdefault(gold, {})
        out[gold][pred] = out[gold].get(pred, 0) + 1
    return out


def format_report(rows: list[dict]) -> str:
    """Produce a human-readable ASCII reliability + confusion report."""
    r = reliability_diagram(rows)
    confusion = verdict_confusion_matrix(rows)

    lines = []
    lines.append("=" * 64)
    lines.append("Reliability Report")
    lines.append("=" * 64)
    lines.append(f"Classified rows: {r['n']}")
    lines.append("")

    if r.get("have_confidence") and r.get("bins"):
        lines.append("Calibration bins (lower gap = better calibration):")
        lines.append(f"  {'range':<14} {'n':>5}  {'acc':>6}  {'avg_conf':>8}  {'gap':>6}")
        for b in r["bins"]:
            if b["n"] == 0:
                lines.append(f"  {b['range']:<14} {0:>5}  {'-':>6}  {'-':>8}  {'-':>6}")
            else:
                lines.append(
                    f"  {b['range']:<14} {b['n']:>5}  "
                    f"{b['accuracy']:.3f}  {b['avg_confidence']:>8.3f}  {b['gap']:.3f}"
                )
        lines.append("")
        lines.append(f"Expected Calibration Error (ECE): {r['ece']:.4f}  (0 = perfect)")
    else:
        lines.append("Reliability-diagram bins unavailable: no 'confidence' field in")
        lines.append("the JSONL rows. (Runner currently records predicted_verdict but not")
        lines.append("confidence_score per case. Add `confidence` to PersistedCaseResult")
        lines.append("to unlock this view — see eval/runner.py.)")
    lines.append("")

    # Confusion table
    all_verdicts = ["supported", "contradicted", "insufficient_support", "needs_review"]
    lines.append("Confusion (rows=gold, cols=predicted):")
    header = " " * 22 + " ".join(f"{v[:6]:>8}" for v in all_verdicts)
    lines.append(header)
    for gold in all_verdicts:
        row = confusion.get(gold, {})
        total = sum(row.values())
        if total == 0:
            continue
        parts = [f"{row.get(v, 0):>8}" for v in all_verdicts]
        lines.append(f"  {gold:<20} " + " ".join(parts) + f"   (n={total})")

    # Overall accuracy
    all_rows = [
        r for r in rows if r.get("status") in (None, "classified")
    ]
    correct = sum(1 for r in all_rows if r.get("gold_verdict") == r.get("predicted_verdict"))
    if all_rows:
        lines.append("")
        lines.append(f"Overall accuracy: {correct}/{len(all_rows)} = {correct/len(all_rows)*100:.1f}%")
    lines.append("=" * 64)
    return "\n".join(lines)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/results/latest.jsonl")
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 1
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(format_report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
