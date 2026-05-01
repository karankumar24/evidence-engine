"""Cross-encoder reranker: lazy singleton + async thread pool wrapper.

The CrossEncoder model (default ``cross-encoder/ms-marco-MiniLM-L6-v2``, 22M
params; configurable via ``settings.reranker_model``) takes a couple of
seconds to load from disk; first run downloads ~80 MB to
``~/.cache/huggingface``. We load it once (lazy singleton) and reuse across
requests. The much larger ``bge-reranker-v2-m3`` (568M params) was tried
earlier but is unusably slow on shared CPU.

Inference is synchronous PyTorch. Running it directly in an async function
would block the FastAPI event loop. We use a single-worker ThreadPoolExecutor
so inference runs in a background thread. max_workers=1 because the model
uses internal BLAS/PyTorch threads — multiple workers cause CPU thrashing.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# sentence_transformers + torch imported lazily inside get_reranker() to avoid
# blocking server startup (cold-boot bottleneck documented in STATE 4306/3778).
from evidenceengine.core.config import settings

_reranker = None
_reranker_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)


def _select_device() -> str:
    """Return 'mps' on Apple Silicon when available, else 'cpu'.

    Mirrors the device-selection pattern in
    ``evidenceengine.classification.nli_classifier._ensure_loaded`` (line 79-81).
    Lazy torch import matches sentence_transformers' own internal pattern.
    """
    import torch  # noqa: PLC0415 — lazy to avoid blocking startup

    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_reranker():
    """Lazy singleton — loads model once, reuses across all requests.

    Uses double-checked locking so the load happens exactly once even under
    concurrent first-requests. Pins the HuggingFace revision (when set in
    ``settings.reranker_model_revision``) for reproducibility, and caps
    max_length at 512 because the default 128 silently truncates long spans.
    """
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import (
                    CrossEncoder,  # noqa: PLC0415 — lazy to avoid blocking startup
                )

                _reranker = CrossEncoder(
                    settings.reranker_model,
                    revision=settings.reranker_model_revision or None,
                    device=_select_device(),
                    max_length=512,
                )
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
