"""Tests for eval/scifact/reliability.py — 10-bin ECE + PNG + CSV reliability diagram."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eval.scifact.reliability import reliability_diagram_scifact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(confidence: float, correct: bool, status: str = "classified") -> dict:
    """Produce a result row where gold_label and predicted_label are consistent with `correct`."""
    if correct:
        return {
            "gold_label": "SUPPORT",
            "predicted_label": "supported",
            "confidence": confidence,
            "status": status,
        }
    else:
        return {
            "gold_label": "SUPPORT",
            "predicted_label": "contradicted",
            "confidence": confidence,
            "status": status,
        }


# ---------------------------------------------------------------------------
# Test 1: perfectly calibrated inputs (confidence == accuracy per bin) → ECE ≈ 0.0
# Put 10 rows, each in a different bin, each correct (accuracy=1.0 per bin)
# BUT confidence matches 1.0 too per row — a perfectly calibrated bin has
# accuracy = avg_confidence.  Use confidence=1.0 for all → each bin accuracy=1.0,
# avg_conf=1.0 → ECE=0.  But that clusters in bin 9.  Instead:
# One row per bin at the bin midpoint, all correct → accuracy=1.0, avg_conf=midpoint.
# That means ECE ≠ 0 unless accuracy == avg_conf.
# Simplest perfect calibration: all in bin [0.9,1.0], confidence=1.0, all correct.
# accuracy=1.0, avg_conf=1.0 → |1.0-1.0|=0 → ECE=0.
# ---------------------------------------------------------------------------

def test_perfectly_calibrated_ece_is_zero(tmp_path):
    # All rows in one bin, accuracy == avg_confidence == 1.0
    results = [
        _make_row(confidence=1.0, correct=True)
        for _ in range(5)
    ]
    out = reliability_diagram_scifact(results, str(tmp_path / "scifact-test"))
    assert out["ece"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 2: bins computed correctly — 10 bins, each covering 0.1 width
# ---------------------------------------------------------------------------

def test_10_bins_width(tmp_path):
    results = [_make_row(0.5, True)]
    out = reliability_diagram_scifact(results, str(tmp_path / "scifact-bins"))
    bins = out["bins"]
    assert len(bins) == 10
    # Each bin covers exactly 0.1 width
    for b in bins:
        assert b["bin_high"] - b["bin_low"] == pytest.approx(0.1, abs=1e-9)
    # First bin: [0.0, 0.1)
    assert bins[0]["bin_low"] == pytest.approx(0.0)
    assert bins[0]["bin_high"] == pytest.approx(0.1)
    # Last bin: [0.9, 1.0]
    assert bins[-1]["bin_low"] == pytest.approx(0.9)
    assert bins[-1]["bin_high"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 3: empty bin has accuracy=None, avg_confidence=None (not 0.0)
# ---------------------------------------------------------------------------

def test_empty_bin_returns_none(tmp_path):
    # Only put rows in the [0.5, 0.6) bin
    results = [_make_row(0.55, True) for _ in range(3)]
    out = reliability_diagram_scifact(results, str(tmp_path / "scifact-empty"))
    bins = out["bins"]
    # Find a bin that has no rows (e.g., [0.0, 0.1))
    empty_bins = [b for b in bins if b["n"] == 0]
    assert len(empty_bins) > 0
    for b in empty_bins:
        assert b["accuracy"] is None, f"Expected None accuracy for empty bin, got {b['accuracy']}"
        assert b["avg_confidence"] is None, f"Expected None avg_confidence for empty bin, got {b['avg_confidence']}"


# ---------------------------------------------------------------------------
# Test 4: PNG and CSV files written to tmp_path; both exist after call
# ---------------------------------------------------------------------------

def test_png_and_csv_written(tmp_path):
    results = [_make_row(0.8, True) for _ in range(3)]
    prefix = str(tmp_path / "scifact-out")
    out = reliability_diagram_scifact(results, prefix)

    assert out["output_png"] == f"{prefix}-reliability.png"
    assert out["output_csv"] == f"{prefix}-reliability.csv"
    assert Path(out["output_png"]).exists(), "PNG file not written"
    assert Path(out["output_csv"]).exists(), "CSV file not written"

    # CSV must have correct columns
    with open(out["output_csv"]) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
    assert columns is not None
    assert set(columns) == {"bin_low", "bin_high", "n", "accuracy", "avg_confidence"}


# ---------------------------------------------------------------------------
# Test 5: status != "classified" rows excluded from bin computation
# ---------------------------------------------------------------------------

def test_non_classified_rows_excluded(tmp_path):
    results = [
        _make_row(0.9, True,  status="classified"),
        _make_row(0.9, False, status="error"),       # must be excluded
        _make_row(0.9, False, status="error"),       # must be excluded
    ]
    out = reliability_diagram_scifact(results, str(tmp_path / "scifact-excl"))
    assert out["n_classified"] == 1
    # Only 1 classified row — accuracy in its bin should be 1.0
    top_bin = next(b for b in out["bins"] if b["n"] > 0)
    assert top_bin["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 6: ECE computed correctly for a known fixture
# Two bins: bin [0.8,0.9) with 2 correct rows at conf=0.85 (acc=1.0, avg_conf=0.85)
#           bin [0.3,0.4) with 2 wrong  rows at conf=0.35 (acc=0.0, avg_conf=0.35)
# n_classified=4
# ECE = (2/4)*|1.0-0.85| + (2/4)*|0.0-0.35| = 0.5*0.15 + 0.5*0.35 = 0.075+0.175 = 0.25
# ---------------------------------------------------------------------------

def test_ece_computation_known_fixture(tmp_path):
    results = [
        _make_row(0.85, correct=True),
        _make_row(0.85, correct=True),
        _make_row(0.35, correct=False),
        _make_row(0.35, correct=False),
    ]
    out = reliability_diagram_scifact(results, str(tmp_path / "scifact-ece"))
    assert out["ece"] == pytest.approx(0.25, abs=1e-6)
