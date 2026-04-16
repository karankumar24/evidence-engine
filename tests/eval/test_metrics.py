"""TDD tests for eval.metrics.compute — seven deterministic test cases."""

import pytest
from eval.metrics.types import BenchmarkResult, EvalReport
from eval.metrics.compute import compute_metrics


def _r(case_id: str, gold: str, pred: str, span_count: int = 1, source: str = "gold") -> BenchmarkResult:
    """Helper to build a BenchmarkResult quickly."""
    return BenchmarkResult(
        case_id=case_id,
        gold_verdict=gold,
        predicted_verdict=pred,
        evidence_span_count=span_count,
        source=source,
    )


class TestComputeMetrics:
    """Seven TDD test cases for compute_metrics()."""

    def test_perfect_predictions(self):
        """Test 1: All four gold cases predict correctly, evidence present."""
        results = [
            _r("1", "supported", "supported", span_count=2),
            _r("2", "contradicted", "contradicted", span_count=2),
            _r("3", "insufficient_support", "insufficient_support", span_count=2),
            _r("4", "needs_review", "needs_review", span_count=2),
        ]
        report = compute_metrics(results)
        assert report.false_support_rate == 0.0
        assert report.contradiction_recall == 1.0
        assert report.insufficient_support_precision == 1.0
        assert report.evidence_retrieval_recall == 1.0

    def test_false_support_rate_maximum(self):
        """Test 2: All non-supported gold predicted 'supported' → rate = 1.0."""
        results = [
            _r("1", "contradicted", "supported"),
            _r("2", "contradicted", "supported"),
            _r("3", "insufficient_support", "supported"),
        ]
        report = compute_metrics(results)
        assert report.false_support_rate == pytest.approx(1.0)

    def test_contradiction_recall_partial(self):
        """Test 3: 1 of 3 contradicted cases correctly predicted → recall = 1/3."""
        results = [
            _r("1", "contradicted", "contradicted"),
            _r("2", "contradicted", "insufficient_support"),
            _r("3", "contradicted", "insufficient_support"),
        ]
        report = compute_metrics(results)
        assert report.contradiction_recall == pytest.approx(1 / 3)

    def test_insufficient_support_precision(self):
        """Test 4: 3 of 4 predicted insufficient are truly insufficient → precision = 0.75."""
        results = [
            _r("1", "insufficient_support", "insufficient_support"),
            _r("2", "insufficient_support", "insufficient_support"),
            _r("3", "insufficient_support", "insufficient_support"),
            _r("4", "contradicted", "insufficient_support"),
        ]
        report = compute_metrics(results)
        assert report.insufficient_support_precision == pytest.approx(0.75)

    def test_evidence_retrieval_recall(self):
        """Test 5: 3 of 4 gold cases have evidence → recall = 0.75."""
        results = [
            _r("1", "supported", "supported", span_count=2),
            _r("2", "contradicted", "contradicted", span_count=1),
            _r("3", "insufficient_support", "insufficient_support", span_count=3),
            _r("4", "needs_review", "needs_review", span_count=0),
        ]
        report = compute_metrics(results)
        assert report.evidence_retrieval_recall == pytest.approx(0.75)

    def test_synthetic_results_excluded(self):
        """Test 6: Synthetic results are excluded from all metric computations."""
        gold_results = [
            _r("g1", "contradicted", "supported", source="gold"),   # false support
            _r("g2", "insufficient_support", "supported", source="gold"),  # false support
        ]
        # Synthetic: perfect — would lower false_support_rate if included
        synthetic_results = [
            _r("s1", "contradicted", "contradicted", source="synthetic"),
            _r("s2", "insufficient_support", "insufficient_support", source="synthetic"),
        ]
        report = compute_metrics(gold_results + synthetic_results)
        # false_support_rate based on gold only: 2/2 = 1.0
        assert report.false_support_rate == pytest.approx(1.0)
        # n_gold should reflect only gold cases
        assert report.n_gold == 2

    def test_zero_denominator_vacuous_truth(self):
        """Test 7: Zero denominator returns 1.0 (vacuous truth, not ZeroDivisionError)."""
        # No contradicted or insufficient_support gold cases → contradiction_recall denom = 0
        results = [
            _r("1", "supported", "supported", span_count=1),
        ]
        report = compute_metrics(results)
        assert report.contradiction_recall == 1.0  # vacuous
        # false_support_rate denom = 0 (no contradicted or insufficient gold) → 1.0
        assert report.false_support_rate == 1.0
        # insufficient_support_precision denom = 0 (no predicted insufficient) → 1.0
        assert report.insufficient_support_precision == 1.0
