"""Tests for eval/scifact/metrics.py — SciFact macro-F1 per Wadden 2020 convention."""
from __future__ import annotations

import pytest

from eval.scifact.metrics import compute_macro_f1, EE_TO_SCIFACT_LABEL


# ---------------------------------------------------------------------------
# Test 1: perfect predictions → macro_f1 = 1.0
# ---------------------------------------------------------------------------

def test_perfect_predictions_yield_macro_f1_1():
    results = [
        {"gold_label": "SUPPORT",    "predicted_label": "supported"},
        {"gold_label": "SUPPORT",    "predicted_label": "supported"},
        {"gold_label": "CONTRADICT", "predicted_label": "contradicted"},
        {"gold_label": "CONTRADICT", "predicted_label": "contradicted"},
    ]
    out = compute_macro_f1(results)
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["support"]["f1"] == pytest.approx(1.0)
    assert out["contradict"]["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 2: all wrong (SUPPORT predicted as CONTRADICT) → macro_f1 = 0.0
# ---------------------------------------------------------------------------

def test_all_wrong_yields_macro_f1_0():
    results = [
        {"gold_label": "SUPPORT",    "predicted_label": "contradicted"},
        {"gold_label": "SUPPORT",    "predicted_label": "contradicted"},
        {"gold_label": "CONTRADICT", "predicted_label": "supported"},
        {"gold_label": "CONTRADICT", "predicted_label": "supported"},
    ]
    out = compute_macro_f1(results)
    assert out["macro_f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: mixed predictions → computed macro_f1 matches sklearn value
# sklearn calculation:
#   gold: SUP SUP SUP CON CON
#   pred: SUP SUP CON CON SUP
#   SUPPORT:    TP=2, FP=1, FN=1  → P=0.667, R=0.667, F1=0.667
#   CONTRADICT: TP=1, FP=1, FN=1  → P=0.500, R=0.500, F1=0.500
#   macro_f1 = (0.667 + 0.500) / 2 = 0.5833
# ---------------------------------------------------------------------------

def test_mixed_predictions_macro_f1():
    results = [
        {"gold_label": "SUPPORT",    "predicted_label": "supported"},      # TP SUP
        {"gold_label": "SUPPORT",    "predicted_label": "supported"},      # TP SUP
        {"gold_label": "SUPPORT",    "predicted_label": "contradicted"},   # FN SUP / FP CON
        {"gold_label": "CONTRADICT", "predicted_label": "contradicted"},   # TP CON
        {"gold_label": "CONTRADICT", "predicted_label": "supported"},      # FN CON / FP SUP
    ]
    out = compute_macro_f1(results)
    assert out["macro_f1"] == pytest.approx(0.5833, abs=1e-3)
    assert out["n_scored"] == 5
    assert out["n_nei"] == 0
    assert out["n_total"] == 5


# ---------------------------------------------------------------------------
# Test 4: NEI claims excluded — n_nei reported, n_scored = n_total - n_nei
# ---------------------------------------------------------------------------

def test_nei_excluded_from_f1():
    results = [
        {"gold_label": "SUPPORT",    "predicted_label": "supported"},
        {"gold_label": "CONTRADICT", "predicted_label": "contradicted"},
        {"gold_label": "",           "predicted_label": "insufficient_support"},  # NEI
        {"gold_label": "",           "predicted_label": "needs_review"},           # NEI
    ]
    out = compute_macro_f1(results)
    assert out["n_nei"] == 2
    assert out["n_total"] == 4
    assert out["n_scored"] == 2
    # Perfect predictions over the 2 scored claims
    assert out["macro_f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 5: empty results list → macro_f1 = 0.0, n_total = 0 (no crash)
# ---------------------------------------------------------------------------

def test_empty_results_no_crash():
    out = compute_macro_f1([])
    assert out["macro_f1"] == pytest.approx(0.0)
    assert out["n_total"] == 0
    assert out["n_scored"] == 0
    assert out["n_nei"] == 0


# ---------------------------------------------------------------------------
# Sanity: EE_TO_SCIFACT_LABEL covers all 4 EE verdicts
# ---------------------------------------------------------------------------

def test_ee_to_scifact_label_complete():
    expected_keys = {"supported", "contradicted", "insufficient_support", "needs_review"}
    assert set(EE_TO_SCIFACT_LABEL.keys()) == expected_keys
    assert EE_TO_SCIFACT_LABEL["supported"] == "SUPPORT"
    assert EE_TO_SCIFACT_LABEL["contradicted"] == "CONTRADICT"
    assert EE_TO_SCIFACT_LABEL["insufficient_support"] == ""
    assert EE_TO_SCIFACT_LABEL["needs_review"] == ""
