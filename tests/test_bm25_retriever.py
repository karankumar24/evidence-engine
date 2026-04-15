"""Unit tests for BM25 retriever (bm25_retriever.py) — RED phase."""

import os
import pytest


CORPUS = [
    "Carbon emissions increased by 12% over this period.",
    "Sea ice extent has declined significantly in the Arctic.",
    "Global temperatures rose by 1.5 degrees Celsius above pre-industrial levels.",
    "Methane concentration in the atmosphere exceeded 1900 ppb.",
    "Renewable energy capacity grew by 295 GW globally in 2022.",
]


def test_build_index_returns_retriever(tmp_path):
    """build_index(spans, index_dir) returns a bm25s.BM25 object."""
    import bm25s
    from evidenceengine.retrieval.bm25_retriever import build_index

    index_dir = str(tmp_path / "idx")
    retriever = build_index(CORPUS, index_dir)
    assert isinstance(retriever, bm25s.BM25)


def test_query_returns_top_k_strings_and_scores(tmp_path):
    """query_index returns (list[str], list[float]) for a query."""
    from evidenceengine.retrieval.bm25_retriever import build_index, query_index

    index_dir = str(tmp_path / "idx")
    retriever = build_index(CORPUS, index_dir)
    texts, scores = query_index(retriever, "CO2 emissions", k=1)
    assert isinstance(texts, list)
    assert isinstance(scores, list)
    assert len(texts) == 1
    assert len(scores) == 1
    assert isinstance(texts[0], str)
    assert isinstance(scores[0], float)


def test_query_returns_at_most_k(tmp_path):
    """Corpus of 5 spans with k=3 returns exactly 3 results."""
    from evidenceengine.retrieval.bm25_retriever import build_index, query_index

    index_dir = str(tmp_path / "idx")
    retriever = build_index(CORPUS, index_dir)
    texts, scores = query_index(retriever, "climate change emissions", k=3)
    assert len(texts) == 3
    assert len(scores) == 3


def test_scores_are_floats(tmp_path):
    """Scores list must contain plain Python floats (not numpy types)."""
    from evidenceengine.retrieval.bm25_retriever import build_index, query_index

    index_dir = str(tmp_path / "idx")
    retriever = build_index(CORPUS, index_dir)
    _, scores = query_index(retriever, "temperature rise", k=2)
    for s in scores:
        assert type(s) is float, f"Expected float, got {type(s)}"


def test_load_or_build_builds_when_missing(tmp_path):
    """When index_dir does not exist, load_or_build_index calls build_index and creates dir."""
    from evidenceengine.retrieval.bm25_retriever import load_or_build_index

    index_dir = str(tmp_path / "new_index")
    assert not os.path.exists(index_dir)
    retriever = load_or_build_index(CORPUS, index_dir)
    assert os.path.exists(index_dir)
    assert retriever is not None


def test_load_or_build_loads_when_present(tmp_path):
    """When index already exists on disk, load_or_build_index loads without rebuilding."""
    from evidenceengine.retrieval.bm25_retriever import build_index, load_or_build_index

    index_dir = str(tmp_path / "existing_index")
    # Build first
    build_index(CORPUS, index_dir)
    # Load — should succeed without raising
    retriever = load_or_build_index(CORPUS, index_dir)
    assert retriever is not None


def test_loaded_index_same_top_result(tmp_path):
    """Save index to tmp_path, load it, query → same top result as original."""
    from evidenceengine.retrieval.bm25_retriever import build_index, load_or_build_index, query_index

    index_dir = str(tmp_path / "cache_test")
    original = build_index(CORPUS, index_dir)
    texts_original, _ = query_index(original, "carbon CO2 emissions", k=1)

    loaded = load_or_build_index(CORPUS, index_dir)
    texts_loaded, _ = query_index(loaded, "carbon CO2 emissions", k=1)

    assert texts_original[0] == texts_loaded[0]


def test_results_shape_2d_sliced(tmp_path):
    """bm25s.retrieve() returns 2D arrays — implementation must slice [0]."""
    from evidenceengine.retrieval.bm25_retriever import build_index, query_index

    index_dir = str(tmp_path / "shape_test")
    retriever = build_index(CORPUS, index_dir)
    # Should not raise IndexError and should return flat lists
    texts, scores = query_index(retriever, "sea ice Arctic", k=2)
    assert isinstance(texts, list)
    assert isinstance(scores, list)
    # Both should be lists (not nested)
    assert not isinstance(texts[0], (list, tuple))
    assert not isinstance(scores[0], (list, tuple))
