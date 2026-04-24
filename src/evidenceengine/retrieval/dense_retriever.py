"""Dense retrieval using sentence-transformers/all-MiniLM-L6-v2.

Fail-open singleton: if the model cannot load, every function returns None/[]
and the retrieval pipeline falls back to BM25-only. No import errors propagate
to callers.

Embeddings are cached to disk at INDEX_DIR/{doc_id}_dense.npy so repeated
runs on the same document skip re-encoding (~3-5s saved per document).

Shape invariant: cache is invalidated whenever the number of spans changes
(document re-indexed), so stale caches never produce wrong top-k results.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DENSE_MODEL_NAME = "all-MiniLM-L6-v2"

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
            logger.info("Dense retrieval model %s loaded", _DENSE_MODEL_NAME)
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
    """Return (N, 384) float32 embedding matrix for span_texts.

    Loads from ``cache_path + "_dense.npy"`` if it exists and the shape matches,
    otherwise encodes and saves. Returns None if the model is unavailable.
    """
    npy_path = Path(cache_path + "_dense.npy")
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
            [query_text], convert_to_numpy=True, show_progress_bar=False
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
