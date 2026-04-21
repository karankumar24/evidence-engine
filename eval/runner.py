"""Async evaluation runner with per-case persistence + resume + status tracking.

Design constraints:
- No HTTP calls, no test client — calls classify_claim() directly as a coroutine.
- Per-case results appended to a JSONL file as soon as they complete; a crash
  mid-run loses ZERO completed work. `--resume` skips cases already in the file.
- Bounded concurrency to play nice with free-tier rate limits.
- Distinguishes classification status: 'classified' (verdict produced),
  'chain_exhausted' (all models 429/timeout — no honest verdict possible),
  'auth_error' (fatal config), 'other_error' (unknown). compute_metrics()
  excludes non-'classified' results so false_support_rate is honest about
  coverage rather than artificially low because most cases never reached an LLM.
- Both gold AND synthetic cases are run; metrics computed gold-only downstream.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from evidenceengine.classification.classifier import classify_claim
from eval.benchmark.loader import load_benchmark
from eval.benchmark.schema import BenchmarkCase, BenchmarkSuite
from eval.metrics.compute import compute_metrics
from eval.metrics.types import BenchmarkResult, EvalReport

logger = logging.getLogger(__name__)


CaseStatus = Literal["classified", "chain_exhausted", "auth_error", "other_error"]


class PersistedCaseResult(BaseModel):
    """One line in the JSONL output. Strict superset of BenchmarkResult plus
    classification status + error string for non-classified cases.

    `predicted_verdict` is set to 'needs_review' when status != 'classified',
    consistent with the prior runner's behavior, BUT downstream metrics
    computation skips non-classified cases so the rate isn't artificially low.
    """
    case_id: str
    gold_verdict: str
    predicted_verdict: str  # 'needs_review' when status != 'classified'
    evidence_span_count: int
    source: str  # 'gold' | 'synthetic'
    status: CaseStatus
    error: str | None = None  # short error message when status != 'classified'
    confidence: float | None = None  # model self-reported score; None for non-classified

    def to_benchmark_result(self) -> BenchmarkResult:
        return BenchmarkResult(
            case_id=self.case_id,
            gold_verdict=self.gold_verdict,  # type: ignore[arg-type]
            predicted_verdict=self.predicted_verdict,  # type: ignore[arg-type]
            evidence_span_count=self.evidence_span_count,
            source=self.source,  # type: ignore[arg-type]
        )


class RunSummary(BaseModel):
    """Wall-clock + coverage facts about the eval run, for honest reporting
    alongside the metrics."""
    n_total: int
    n_classified: int
    n_chain_exhausted: int
    n_auth_error: int
    n_other_error: int
    output_path: str

    @property
    def coverage(self) -> float:
        return self.n_classified / self.n_total if self.n_total else 0.0


def _load_persisted_results(path: Path) -> dict[str, PersistedCaseResult]:
    """Read existing JSONL into a {case_id: PersistedCaseResult} dict for resume."""
    if not path.exists():
        return {}
    out: dict[str, PersistedCaseResult] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = PersistedCaseResult.model_validate_json(line)
                out[rec.case_id] = rec
            except Exception as exc:  # noqa: BLE001 — corrupted line should not block run
                logger.warning("Skipping unparseable result line: %s", exc)
    return out


def _append_result(path: Path, result: PersistedCaseResult) -> None:
    """Append a single result line + flush so a crash survives partial writes."""
    with path.open("a") as f:
        f.write(result.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def _classify_exception(exc: BaseException) -> tuple[CaseStatus, str]:
    """Map a raised exception to (status, short_error_string).

    'chain_exhausted' is the most common free-tier failure: every model 429'd
    or 404'd across all retry rounds. Don't conflate it with 'genuine model
    judgement = needs_review'.
    """
    msg = str(exc)
    name = type(exc).__name__
    short = f"{name}: {msg[:120]}"

    if "fallback chain exhausted" in msg or "rate limit" in msg.lower() or "429" in msg:
        return "chain_exhausted", short
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return "auth_error", short
    return "other_error", short


async def _evaluate_case(case: BenchmarkCase) -> PersistedCaseResult:
    """Run a single benchmark case. Returns a PersistedCaseResult tagged with status."""
    evidence_dicts = [
        {"span_text": s.span_text, "relevance_score": s.relevance_score, "rank": s.rank}
        for s in case.evidence_spans
    ]
    try:
        response = await classify_claim(case.claim_text, evidence_dicts)
        return PersistedCaseResult(
            case_id=case.id,
            gold_verdict=case.gold_verdict,
            predicted_verdict=response.verdict_type,
            evidence_span_count=len(case.evidence_spans),
            source=case.source,
            status="classified",
            confidence=float(response.confidence_score),
        )
    except Exception as exc:  # noqa: BLE001 — runner must not crash on single-case failure
        status, short = _classify_exception(exc)
        logger.warning("classify_claim failed for case %s [%s]: %s", case.id, status, short)
        return PersistedCaseResult(
            case_id=case.id,
            gold_verdict=case.gold_verdict,
            predicted_verdict="needs_review",
            evidence_span_count=len(case.evidence_spans),
            source=case.source,
            status=status,
            error=short,
        )


async def run_evaluation(
    suite: BenchmarkSuite | None = None,
    *,
    output_path: Path | str = Path("eval/results/latest.jsonl"),
    concurrency: int = 3,
    resume: bool = True,
    max_cases: int | None = None,
) -> tuple[EvalReport, list[PersistedCaseResult], RunSummary]:
    """Run benchmark cases through the classifier with per-case persistence.

    Args:
        suite: BenchmarkSuite. If None, loads from default fixtures.
        output_path: JSONL file to append per-case results to. Resumed from on restart.
        concurrency: Max concurrent classify_claim calls. Lower is friendlier
            to free-tier rate limits. Default 3.
        resume: If True, skip cases already present in output_path.
        max_cases: If set, evaluate at most this many cases (after resume skip).
            Useful for sanity-check runs before a full 220-case sweep.

    Returns:
        (EvalReport, list[PersistedCaseResult], RunSummary).
        EvalReport is computed only over status='classified' cases.

    Raises:
        Nothing — single-case failures are persisted with status set; auth errors
        ARE persisted but the run continues (caller can grep the JSONL).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if suite is None:
        suite = load_benchmark()
    all_cases = list(suite.gold) + list(suite.synthetic)

    persisted = _load_persisted_results(output_path) if resume else {}
    # Retry chain_exhausted cases when resuming — a quota-exhausted run
    # that never got an LLM verdict should NOT be treated as "done." Only
    # 'classified' (and fatal 'auth_error') results are considered final.
    retriable_statuses = {"chain_exhausted", "other_error"}
    stale_ids = {
        cid for cid, rec in persisted.items() if rec.status in retriable_statuses
    }
    if stale_ids:
        logger.info(
            "Resume: retrying %d previously %s case(s)",
            len(stale_ids),
            "/".join(retriable_statuses),
        )
        persisted = {cid: rec for cid, rec in persisted.items() if cid not in stale_ids}
        # Rewrite the JSONL without the stale rows so re-runs append cleanly.
        with output_path.open("w") as f:
            for rec in persisted.values():
                f.write(rec.model_dump_json() + "\n")
    if persisted:
        logger.info(
            "Resume: %d classified/auth-error cases already in %s — skipping these",
            len(persisted), output_path,
        )

    pending = [c for c in all_cases if c.id not in persisted]
    if max_cases is not None:
        pending = pending[:max_cases]
    logger.info(
        "Evaluating %d pending cases (out of %d total) with concurrency=%d",
        len(pending), len(all_cases), concurrency,
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_eval(case: BenchmarkCase) -> PersistedCaseResult:
        async with semaphore:
            result = await _evaluate_case(case)
            _append_result(output_path, result)
            return result

    new_results = await asyncio.gather(*(bounded_eval(c) for c in pending))

    # Merge resumed + new
    all_results = list(persisted.values()) + new_results

    classified_results = [r.to_benchmark_result() for r in all_results if r.status == "classified"]
    report = compute_metrics(classified_results)

    summary = RunSummary(
        n_total=len(all_results),
        n_classified=sum(1 for r in all_results if r.status == "classified"),
        n_chain_exhausted=sum(1 for r in all_results if r.status == "chain_exhausted"),
        n_auth_error=sum(1 for r in all_results if r.status == "auth_error"),
        n_other_error=sum(1 for r in all_results if r.status == "other_error"),
        output_path=str(output_path),
    )
    return report, all_results, summary
