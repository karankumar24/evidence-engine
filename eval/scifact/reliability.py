"""Reliability diagram (calibration) for SciFact evaluation results.

Bins predictions by confidence (10 equal-width bins), computes Expected
Calibration Error (ECE), and writes a PNG reliability diagram + CSV summary.

Reuses the bin-edge math from eval/reliability.py (already correct + tested).
Adds matplotlib rendering and CSV export that the original module omits.

No imports from evidenceengine.* — DB-free, importable without SQLAlchemy.
"""
from __future__ import annotations

import csv
from pathlib import Path

from eval.scifact.metrics import EE_TO_SCIFACT_LABEL


def _bin_edges(n_bins: int = 10) -> list[tuple[float, float]]:
    """Identical to eval/reliability._bin_edges — copied to keep this module self-contained."""
    step = 1.0 / n_bins
    return [(i * step, (i + 1) * step) for i in range(n_bins)]


def reliability_diagram_scifact(
    results: list[dict],
    output_prefix: str,
    *,
    n_bins: int = 10,
) -> dict:
    """Compute ECE and write reliability diagram PNG + CSV.

    Args:
        results: list of result dicts with keys:
            - gold_label: str
            - predicted_label: str (EE verdict)
            - confidence: float (0.0–1.0)
            - status: str ("classified" | "error")
        output_prefix: path prefix for output files, e.g. "eval/results/scifact-dev300"
            PNG → {output_prefix}-reliability.png
            CSV → {output_prefix}-reliability.csv
        n_bins: number of equal-width confidence bins (default 10)

    Returns:
        {"ece": float, "bins": list[dict], "n_classified": int,
         "output_png": str, "output_csv": str}
    """
    classified = [r for r in results if r.get("status") == "classified"]
    n_classified = len(classified)

    edges = _bin_edges(n_bins)
    bins: list[dict] = []

    for low, high in edges:
        # Last bin is inclusive on the right edge to capture confidence==1.0
        is_last = high == 1.0
        if is_last:
            in_bin = [
                r for r in classified
                if low <= float(r.get("confidence", 0.0)) <= 1.0
            ]
        else:
            in_bin = [
                r for r in classified
                if low <= float(r.get("confidence", 0.0)) < high
            ]

        if not in_bin:
            bins.append({
                "bin_low": low,
                "bin_high": high,
                "n": 0,
                "accuracy": None,
                "avg_confidence": None,
            })
            continue

        # Accuracy: fraction where predicted SciFact label matches gold SciFact label
        correct = sum(
            1 for r in in_bin
            if r.get("predicted_label") is not None
            and r.get("gold_label") is not None
            and EE_TO_SCIFACT_LABEL.get(r["predicted_label"], "") == r["gold_label"]
        )
        accuracy = correct / len(in_bin)
        avg_conf = sum(float(r.get("confidence", 0.0)) for r in in_bin) / len(in_bin)

        bins.append({
            "bin_low": low,
            "bin_high": high,
            "n": len(in_bin),
            "accuracy": accuracy,
            "avg_confidence": avg_conf,
        })

    # ECE = weighted mean |accuracy - avg_confidence| over non-empty bins
    ece = 0.0
    if n_classified > 0:
        for b in bins:
            if b["n"] > 0 and b["accuracy"] is not None:
                ece += (b["n"] / n_classified) * abs(b["accuracy"] - b["avg_confidence"])

    output_png = f"{output_prefix}-reliability.png"
    output_csv = f"{output_prefix}-reliability.csv"

    # Write CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin_low", "bin_high", "n", "accuracy", "avg_confidence"],
        )
        writer.writeheader()
        writer.writerows(bins)

    # Write PNG
    _plot_reliability(bins, ece, output_png)

    return {
        "ece": ece,
        "bins": bins,
        "n_classified": n_classified,
        "output_png": output_png,
        "output_csv": output_csv,
    }


def _plot_reliability(bins_data: list[dict], ece: float, output_path: str) -> None:
    """Render and save the reliability diagram PNG."""
    import matplotlib  # noqa: PLC0415 — lazy; not available in all test envs
    matplotlib.use("Agg")  # Non-interactive backend — safe for headless/CI
    import matplotlib.pyplot as plt  # noqa: PLC0415

    centers = [(b["bin_low"] + b["bin_high"]) / 2 for b in bins_data]
    accuracies = [b["accuracy"] if b["accuracy"] is not None else 0.0 for b in bins_data]
    counts = [b["n"] for b in bins_data]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(centers, accuracies, width=0.08, alpha=0.7, label="Accuracy", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"SciFact Reliability Diagram (ECE={ece:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    # Annotate bars with counts
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                str(count),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
