"""SciFact evaluation runner: BM25 + rerank + classify loop with JSONL persistence.

Design mirrors eval/runner.py:
- Per-claim results appended to JSONL immediately; crash loses zero completed work.
- --resume skips claims already written with status="classified".
- Bounded concurrency via asyncio.Semaphore (default=1 for M1 8GB memory safety).
- Retrieval scoped to cited_doc_ids only (Pitfall 5: no full-corpus BM25).
- Uses get_backend().classify() for nli_primary path (not legacy classify_claim).

Memory note (STATE.md): DeBERTa ~870 MB + bge-reranker ~1.1 GB + OS ~1.5 GB ≈ 3.5 GB peak.
Concurrency=1 is the safe default on 8 GB M1 Air.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import bm25s
from pydantic import BaseModel

from evidenceengine.retrieval.reranker import rerank
from evidenceengine.classification.backend import get_backend
from eval.scifact.loader import ScifactClaim, load_scifact

logger = logging.getLogger(__name__)

CaseStatus = Literal["classified", "error"]

DEFAULT_RESULTS_DIR = "eval/results"
DEFAULT_CACHE_DIR = "eval/data/scifact/"
TOP_K_BM25 = 10    # BM25 candidates before reranking
TOP_K_RERANK = 5   # top-k after reranking (evidence spans passed to classifier)


class ScifactCaseResult(BaseModel):
    id: int
    gold_label: str          # "SUPPORT", "CONTRADICT", or "" (NEI)
    predicted_label: str     # EE verdict: "supported", "contradicted", etc.
    confidence: float
    retrieved_spans: list[str]
    retrieval_recall: int    # 1 if any cited abstract sentence in top-k, 0 otherwise
    latency_ms: float
    status: CaseStatus
    error: str | None = None


def _load_existing_results(path: Path) -> dict[int, ScifactCaseResult]:
    """Load existing JSONL → {claim_id: result}. Returns empty dict if file missing."""
    if not path.exists():
        return {}
    results: dict[int, ScifactCaseResult] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                result = ScifactCaseResult(**obj)
                results[result.id] = result
            except Exception as exc:
                logger.warning("Skipping malformed JSONL line: %s", exc)
    return results


def _append_result(path: Path, result: ScifactCaseResult) -> None:
    """Append one result to JSONL with fsync — crash-safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result.model_dump()) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _build_ephemeral_bm25(sentences: list[str]):
    """Build an in-memory BM25 index over a list of sentences."""
    corpus_tokens = bm25s.tokenize(sentences, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.corpus = [{"id": i, "text": s} for i, s in enumerate(sentences)]
    return retriever


async def _retrieve_for_claim(
    claim: ScifactClaim,
    corpus: dict[int, dict],
) -> tuple[list[str], int]:
    """Build ephemeral BM25 over cited abstracts + rerank. Returns (top_spans, retrieval_recall).

    Scoped to cited_doc_ids only (research Pitfall 5). Retrieval_recall=1 if any
    sentence from any cited abstract appears in top-k spans.
    """
    all_sentences: list[str] = []
    for doc_id in claim.cited_doc_ids:
        if doc_id in corpus:
            all_sentences.extend(corpus[doc_id]["abstract"])

    if not all_sentences:
        return [], 0

    retriever = _build_ephemeral_bm25(all_sentences)

    k_bm25 = min(TOP_K_BM25, len(all_sentences))
    from evidenceengine.retrieval.bm25_retriever import query_index  # noqa: PLC0415
    top_texts, _ = query_index(retriever, claim.claim, k=k_bm25)

    if not top_texts:
        return [], 0

    # Rerank with bge-reranker-v2-m3
    reranked = await rerank(claim.claim, top_texts)
    top_k_spans = [r["text"] for r in reranked[:TOP_K_RERANK]]

    # Retrieval recall: did the top-k include any sentence from cited abstracts?
    # Since all sentences ARE from cited abstracts, recall@k = 1 if top_k_spans non-empty.
    retrieval_recall = 1 if top_k_spans else 0

    return top_k_spans, retrieval_recall


async def _evaluate_claim(
    claim: ScifactClaim,
    corpus: dict[int, dict],
    backend,
) -> ScifactCaseResult:
    """Run full pipeline for one claim. Catches all exceptions — never crashes the runner."""
    t0 = time.perf_counter()
    try:
        evidence_spans, retrieval_recall = await _retrieve_for_claim(claim, corpus)
        response = await backend.classify(claim.claim, evidence_spans)
        latency_ms = (time.perf_counter() - t0) * 1000

        return ScifactCaseResult(
            id=claim.id,
            gold_label=claim.gold_label,
            predicted_label=response.verdict,
            confidence=response.confidence_score,
            retrieved_spans=evidence_spans,
            retrieval_recall=retrieval_recall,
            latency_ms=latency_ms,
            status="classified",
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.exception("Claim %d failed: %s", claim.id, exc)
        return ScifactCaseResult(
            id=claim.id,
            gold_label=claim.gold_label,
            predicted_label="needs_review",
            confidence=0.0,
            retrieved_spans=[],
            retrieval_recall=0,
            latency_ms=latency_ms,
            status="error",
            error=str(exc),
        )


async def run_scifact_evaluation(
    split: str = "dev",
    resume: bool = False,
    max_cases: int | None = None,
    concurrency: int = 1,
    results_dir: str = DEFAULT_RESULTS_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict:
    """Run full SciFact evaluation pipeline.

    Args:
        split: "dev" (300 claims) or "train"
        resume: skip claims already in results JSONL with status="classified"
        max_cases: if set, process at most N claims (smoke-test shortcut)
        concurrency: number of parallel classify calls (default 1 — M1 memory safety)
        results_dir: directory for output JSONL
        cache_dir: HuggingFace dataset cache directory

    Returns:
        dict with final metrics from compute_macro_f1()
    """
    from eval.scifact.metrics import compute_macro_f1  # noqa: PLC0415
    from eval.scifact.reliability import reliability_diagram_scifact  # noqa: PLC0415

    results_path = Path(results_dir) / f"scifact-{split}300.jsonl"
    output_prefix = str(Path(results_dir) / f"scifact-{split}300")

    logger.info("Loading SciFact %s split...", split)
    claims, corpus = load_scifact(split=split, cache_dir=cache_dir)

    existing: dict[int, ScifactCaseResult] = {}
    if resume:
        existing = _load_existing_results(results_path)
        logger.info("Resume: %d claims already classified", len(existing))

    # Filter: skip already-classified; apply max_cases
    pending = [c for c in claims if c.id not in existing or existing[c.id].status != "classified"]
    if max_cases is not None:
        pending = pending[:max_cases]

    logger.info("Processing %d claims (concurrency=%d)...", len(pending), concurrency)

    backend = get_backend()
    sem = asyncio.Semaphore(concurrency)

    async def bounded_eval(claim: ScifactClaim) -> ScifactCaseResult:
        async with sem:
            result = await _evaluate_claim(claim, corpus, backend)
            _append_result(results_path, result)
            return result

    new_results = await asyncio.gather(*[bounded_eval(c) for c in pending])

    # Merge existing + new for final metrics
    all_results: dict[int, ScifactCaseResult] = {**existing}
    for r in new_results:
        all_results[r.id] = r

    results_list = [r.model_dump() for r in all_results.values()]

    metrics = compute_macro_f1(results_list)
    logger.info(
        "macro_F1=%.4f  support_F1=%.4f  contradict_F1=%.4f  n_nei=%d",
        metrics["macro_f1"], metrics["support"]["f1"], metrics["contradict"]["f1"], metrics["n_nei"],
    )

    try:
        calibration = reliability_diagram_scifact(results_list, output_prefix)
        logger.info("ECE=%.4f  PNG=%s  CSV=%s", calibration["ece"], calibration["output_png"], calibration["output_csv"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reliability diagram failed (non-fatal): %s", exc)
        calibration = {}

    return {**metrics, "calibration": calibration, "results_path": str(results_path)}
