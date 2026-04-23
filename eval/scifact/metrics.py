"""SciFact abstract-level macro-F1 computation.

Convention (Wadden et al. 2020, "Fact or Fiction"):
  - Macro-F1 is computed over SUPPORT + CONTRADICT classes only.
  - NEI (No Evidence Info) claims — gold_label=="" — are excluded from the
    F1 denominator but counted separately for coverage reporting.
  - Macro-F1 = unweighted mean of F1(SUPPORT) and F1(CONTRADICT).

No imports from evidenceengine.* — this module is DB-free and importable
from any context without triggering the SQLAlchemy hang (STATE.md Pitfall 3).
"""
from __future__ import annotations

from sklearn.metrics import precision_recall_fscore_support

# EvidenceEngine 4-way verdict → SciFact 3-way label
EE_TO_SCIFACT_LABEL: dict[str, str] = {
    "supported":            "SUPPORT",
    "contradicted":         "CONTRADICT",
    "insufficient_support": "",   # maps to NEI
    "needs_review":         "",   # maps to NEI
}

_SCIFACT_LABELS = ["SUPPORT", "CONTRADICT"]


def compute_macro_f1(results: list[dict]) -> dict:
    """Compute SciFact abstract-level macro-F1.

    Args:
        results: list of dicts with at minimum:
            - gold_label: str ("SUPPORT", "CONTRADICT", or "" for NEI)
            - predicted_label: str (EE verdict string)

    Returns:
        {
            "macro_f1": float,
            "support":   {"precision": float, "recall": float, "f1": float, "n": int},
            "contradict": {"precision": float, "recall": float, "f1": float, "n": int},
            "n_nei":    int,   # NEI gold claims (excluded from F1)
            "n_scored": int,   # SUPPORT + CONTRADICT gold claims
            "n_total":  int,   # all claims
        }
    """
    n_total = len(results)
    n_nei = sum(1 for r in results if r.get("gold_label", "") == "")

    # Filter to SUPPORT + CONTRADICT gold claims only
    scored = [r for r in results if r.get("gold_label", "") in _SCIFACT_LABELS]
    n_scored = len(scored)

    if n_scored == 0:
        return {
            "macro_f1": 0.0,
            "support":   {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0},
            "contradict": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n": 0},
            "n_nei": n_nei,
            "n_scored": 0,
            "n_total": n_total,
        }

    gold = [r["gold_label"] for r in scored]
    pred = [EE_TO_SCIFACT_LABEL.get(r.get("predicted_label", ""), "") for r in scored]

    p, r, f1, support_counts = precision_recall_fscore_support(
        gold,
        pred,
        labels=_SCIFACT_LABELS,
        average=None,
        zero_division=0,
    )

    macro_f1 = float(f1.mean())

    return {
        "macro_f1": macro_f1,
        "support": {
            "precision": float(p[0]),
            "recall":    float(r[0]),
            "f1":        float(f1[0]),
            "n":         int(support_counts[0]),
        },
        "contradict": {
            "precision": float(p[1]),
            "recall":    float(r[1]),
            "f1":        float(f1[1]),
            "n":         int(support_counts[1]),
        },
        "n_nei":    n_nei,
        "n_scored": n_scored,
        "n_total":  n_total,
    }
