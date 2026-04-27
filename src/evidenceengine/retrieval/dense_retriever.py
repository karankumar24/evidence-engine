"""Dense retrieval using sentence-transformers (BGE-base-en-v1.5 default).

Fail-open singleton: if the model cannot load, every function returns None/[]
and the retrieval pipeline falls back to BM25-only. No import errors propagate
to callers.

Embeddings are cached to disk at INDEX_DIR/{doc_id}_dense_{model_slug}.npy so
repeated runs on the same document skip re-encoding (~3-5s saved per document).
The model name is part of the cache key — swapping models auto-invalidates stale
caches (384-dim MiniLM vs 768-dim BGE are incompatible).

Shape invariant: cache is invalidated whenever the number of spans changes
(document re-indexed), so stale caches never produce wrong top-k results.

BGE-base-en-v1.5 note: QUERY embeddings require a prefix for correct accuracy.
DOCUMENT embeddings (indexed passages) do NOT get the prefix.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from evidenceengine.core.config import settings

logger = logging.getLogger(__name__)

_DENSE_MODEL_NAME = settings.dense_embedding_model

# BGE-base requires this prefix on query text only (not on document passages).
# Without it, the model still runs but loses ~5-8pp retrieval accuracy.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_DENSE_CORPUS_SIZE = settings.dense_retrieval_max_corpus_size

# Cache slug: safe filesystem name derived from model identifier
_CACHE_MODEL_SLUG = _DENSE_MODEL_NAME.replace("/", "_").replace("-", "_").lower()

_model = None
_lock = threading.Lock()
_load_failed = False


def _ensure_loaded():
    """Double-check-locked lazy load. Returns model or None on failure. Never raises."""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    with _lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            _model = SentenceTransformer(_DENSE_MODEL_NAME)
            _model.max_seq_length = 512  # explicit for BGE-base (default 512, was 128 for MiniLM)
            logger.info("Dense retrieval model %s loaded (max_seq_length=512)", _DENSE_MODEL_NAME)
            return _model
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Dense retrieval disabled: could not load %s (%s) — BM25-only fallback active.",
                _DENSE_MODEL_NAME,
                exc,
            )
            _load_failed = True
            return None


def load_or_build_dense(span_texts: list[str], cache_path: str) -> np.ndarray | None:
    """Return (N, D) float32 embedding matrix for span_texts (D=768 for BGE-base, 384 for MiniLM).

    Loads from ``cache_path + "_dense.npy"`` if it exists and the shape matches,
    otherwise encodes and saves. Returns None if the model is unavailable.
    """
    npy_path = Path(f"{cache_path}_dense_{_CACHE_MODEL_SLUG}.npy")
    if npy_path.exists():
        try:
            cached = np.load(str(npy_path))
            if cached.shape[0] == len(span_texts):
                return cached
            logger.debug(
                "Dense cache shape mismatch (%d vs %d) at %s — recomputing",
                cached.shape[0], len(span_texts), npy_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dense cache load failed (%s) — recomputing", exc)

    # Skip dense encoding for very large corpora — too slow on shared CPU.
    # BM25 + cross-encoder reranking handles quality for large documents.
    if len(span_texts) > MAX_DENSE_CORPUS_SIZE:
        logger.info(
            "Dense encoding skipped: corpus size %d exceeds limit %d — BM25-only for this document",
            len(span_texts), MAX_DENSE_CORPUS_SIZE,
        )
        return None

    model = _ensure_loaded()
    if model is None:
        return None

    try:
        embeddings: np.ndarray = model.encode(
            span_texts, convert_to_numpy=True, show_progress_bar=False
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dense encode failed: %s", exc)
        return None

    try:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), embeddings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dense cache save failed: %s — continuing without disk cache", exc)

    return embeddings


def _apply_query_prefix(text: str) -> str:
    """Apply BGE query prefix when using a BGE model. No-op for other models."""
    if "bge" in _DENSE_MODEL_NAME.lower():
        return f"{_BGE_QUERY_PREFIX}{text}"
    return text


def query_dense(
    query_text: str,
    corpus_embeddings: np.ndarray,
    span_texts: list[str],
    k: int,
) -> list[str]:
    """Return top-k span texts ranked by cosine similarity. Empty list on any failure."""
    model = _ensure_loaded()
    if model is None:
        return []
    try:
        query_emb: np.ndarray = model.encode(
            [_apply_query_prefix(query_text)], convert_to_numpy=True, show_progress_bar=False
        )[0]
        # Cosine similarity: normalise both sides then dot product.
        corpus_norm = corpus_embeddings / (
            np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
        )
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        scores = corpus_norm @ query_norm
        top_k_idx = np.argsort(scores)[::-1][:k]
        return [span_texts[i] for i in top_k_idx]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dense query failed: %s — returning empty candidates", exc)
        return []
