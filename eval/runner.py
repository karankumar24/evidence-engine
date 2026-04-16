"""Async evaluation runner: runs benchmark cases through classify_claim and computes metrics.

Design constraints:
- No HTTP calls, no HTTP test clients, no ASGI server — calls classify_claim() directly as a coroutine
- concurrency semaphore prevents flooding OpenAI API
- Case failures default to needs_review and log a warning — harness never crashes on single failure
- Both gold AND synthetic cases are run; EvalReport is computed from gold-only (enforced in compute_metrics)
"""

import asyncio
import logging

from evidenceengine.classification.classifier import classify_claim
from eval.benchmark.loader import load_benchmark
from eval.benchmark.schema import BenchmarkCase, BenchmarkSuite
from eval.metrics.compute import compute_metrics
from eval.metrics.types import BenchmarkResult, EvalReport

logger = logging.getLogger(__name__)


async def _evaluate_case(case: BenchmarkCase, model: str | None = None) -> BenchmarkResult:
    """Run a single benchmark case through the classifier."""
    # Convert EvidenceSpanFixture list to the dict format classify_claim expects
    evidence_dicts = [
        {"span_text": s.span_text, "relevance_score": s.relevance_score, "rank": s.rank}
        for s in case.evidence_spans
    ]
    try:
        response = await classify_claim(case.claim_text, evidence_dicts)
        predicted = response.verdict_type
    except Exception as exc:
        logger.warning(
            "classify_claim failed for case %s: %s — defaulting to needs_review",
            case.id,
            exc,
        )
        predicted = "needs_review"

    return BenchmarkResult(
        case_id=case.id,
        gold_verdict=case.gold_verdict,
        predicted_verdict=predicted,
        evidence_span_count=len(case.evidence_spans),
        source=case.source,
    )


async def run_evaluation(
    suite: BenchmarkSuite | None = None,
    *,
    model: str | None = None,
    concurrency: int = 5,
) -> tuple[EvalReport, list[BenchmarkResult]]:
    """Run all benchmark cases (gold + synthetic) through the classifier.

    Args:
        suite: BenchmarkSuite to evaluate. If None, loads from fixtures via load_benchmark().
        model: Override the OpenAI model name (passed to environment or future config injection).
        concurrency: Max concurrent classify_claim calls (rate-limit protection).

    Returns:
        Tuple of (EvalReport, list[BenchmarkResult]) — report computed from gold-only results.
    """
    if suite is None:
        suite = load_benchmark()

    all_cases = suite.gold + suite.synthetic
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_eval(case: BenchmarkCase) -> BenchmarkResult:
        async with semaphore:
            return await _evaluate_case(case, model=model)

    results = await asyncio.gather(*[bounded_eval(case) for case in all_cases])
    report = compute_metrics(list(results))
    return report, list(results)
