"""Cross-encoder reranker: lazy singleton + async thread pool wrapper.

The CrossEncoder model (22.7M params) takes 2-5 seconds to load from disk.
We load it once (lazy singleton) and reuse across requests.

Inference is synchronous PyTorch. Running it directly in an async function
would block the FastAPI event loop. We use a single-worker ThreadPoolExecutor
so inference runs in a background thread. max_workers=1 because the model
uses internal BLAS/PyTorch threads — multiple workers cause CPU thrashing.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# sentence_transformers imported lazily inside get_reranker() to avoid blocking startup
from evidenceengine.core.config import settings

_reranker = None
_reranker_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def get_reranker():
    """Lazy singleton — loads model once, reuses across all requests."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import (
                    CrossEncoder,  # noqa: PLC0415 — lazy to avoid blocking startup
                )
                _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


async def rerank(query: str, passages: list[str]) -> list[dict]:
    """Rerank passages by relevance to query using the cross-encoder model.

    Runs model.rank() in a thread pool to avoid blocking the async event loop.
    The cross-encoder scores each (query, passage) pair jointly — significantly
    more accurate than bi-encoder cosine similarity for pairwise relevance.

    Args:
        query: Claim text used as the query.
        passages: List of evidence span texts to rerank (BM25 top-k candidates).

    Returns:
        List of dicts sorted by score descending:
        [{corpus_id: int, score: float, text: str}, ...]
        score is a raw logit (unbounded real; positive = relevant, negative = irrelevant).
        Store as EvidenceSpan.relevance_score — Phase 4 uses rank order, not probabilities.
    """
    if not passages:
        return []

    loop = asyncio.get_event_loop()

    def _predict() -> list[dict]:
        model = get_reranker()
        ranks = model.rank(query, passages, return_documents=True)
        # model.rank() already returns sorted by score descending
        return [{"corpus_id": r["corpus_id"], "score": float(r["score"]), "text": r["text"]}
                for r in ranks]

    return await loop.run_in_executor(_executor, _predict)
