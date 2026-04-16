"""Pure-function metric computation for the EvidenceEngine benchmark.

All metrics operate only on gold-sourced results (EVAL-07 enforcement).
Zero denominators return 1.0 (vacuous truth — not a failure).
"""

from eval.metrics.types import BenchmarkResult, EvalReport


def _safe_divide(numerator: int, denominator: int) -> float:
    """Divide, returning 1.0 on zero denominator (vacuous truth)."""
    if denominator == 0:
        return 1.0
    return numerator / denominator


def compute_metrics(results: list[BenchmarkResult]) -> EvalReport:
    """Compute all four eval metrics from a list of BenchmarkResult.

    Only gold-sourced results contribute to metrics (EVAL-07).

    Args:
        results: Mixed list of gold and synthetic BenchmarkResult objects.

    Returns:
        EvalReport with all four metrics computed from gold results only.

    Metric definitions:
        false_support_rate =
            count(gold_verdict ∈ {contradicted, insufficient_support} AND predicted == "supported")
            / count(gold_verdict ∈ {contradicted, insufficient_support})

        contradiction_recall =
            count(gold_verdict == "contradicted" AND predicted == "contradicted")
            / count(gold_verdict == "contradicted")

        insufficient_support_precision =
            count(gold_verdict == "insufficient_support" AND predicted == "insufficient_support")
            / count(predicted == "insufficient_support")

        evidence_retrieval_recall =
            count(evidence_span_count > 0)
            / count(all gold results)
    """
    # Filter to gold only (EVAL-07)
    gold = [r for r in results if r.source == "gold"]

    # -- false_support_rate --
    false_support_denominator = [
        r for r in gold
        if r.gold_verdict in ("contradicted", "insufficient_support")
    ]
    false_support_numerator = [
        r for r in false_support_denominator
        if r.predicted_verdict == "supported"
    ]
    false_support_rate = _safe_divide(len(false_support_numerator), len(false_support_denominator))

    # -- contradiction_recall --
    gold_contradicted = [r for r in gold if r.gold_verdict == "contradicted"]
    correctly_contradicted = [r for r in gold_contradicted if r.predicted_verdict == "contradicted"]
    contradiction_recall = _safe_divide(len(correctly_contradicted), len(gold_contradicted))

    # -- insufficient_support_precision --
    gold_insufficient = [r for r in gold if r.gold_verdict == "insufficient_support"]
    predicted_insufficient = [r for r in gold if r.predicted_verdict == "insufficient_support"]
    true_insufficient = [
        r for r in predicted_insufficient if r.gold_verdict == "insufficient_support"
    ]
    insufficient_support_precision = _safe_divide(len(true_insufficient), len(predicted_insufficient))

    # -- evidence_retrieval_recall --
    gold_with_evidence = [r for r in gold if r.evidence_span_count > 0]
    evidence_retrieval_recall = _safe_divide(len(gold_with_evidence), len(gold))

    return EvalReport(
        false_support_rate=false_support_rate,
        contradiction_recall=contradiction_recall,
        insufficient_support_precision=insufficient_support_precision,
        evidence_retrieval_recall=evidence_retrieval_recall,
        n_gold=len(gold),
        n_gold_contradicted=len(gold_contradicted),
        n_gold_insufficient=len(gold_insufficient),
        n_predicted_insufficient=len(predicted_insufficient),
        n_gold_with_evidence=len(gold_with_evidence),
    )
