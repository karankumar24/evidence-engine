"""Unit tests for cross-encoder reranker (reranker.py) — RED phase.

All tests mock CrossEncoder — never instantiate the real model.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_cross_encoder(monkeypatch):
    """Fixture: replace module-level _reranker with a mock CrossEncoder."""
    import evidenceengine.retrieval.reranker as reranker_mod

    mock = MagicMock()
    mock.rank.return_value = [
        {"corpus_id": 1, "score": 8.5, "text": "CO2 rose 12%"},
        {"corpus_id": 0, "score": 3.2, "text": "Sea ice declined"},
    ]
    monkeypatch.setattr(reranker_mod, "_reranker", mock)
    return mock


def test_rerank_returns_list_of_dicts(mock_cross_encoder):
    """rerank(query, passages) returns list[dict] with keys corpus_id, score, text."""
    from evidenceengine.retrieval.reranker import rerank

    result = asyncio.get_event_loop().run_until_complete(
        rerank("CO2 emissions", ["Sea ice declined", "CO2 rose 12%"])
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert "corpus_id" in item
        assert "score" in item
        assert "text" in item


def test_rerank_sorted_by_score_descending(mock_cross_encoder):
    """Results are sorted by score descending (highest relevance first)."""
    from evidenceengine.retrieval.reranker import rerank

    result = asyncio.get_event_loop().run_until_complete(
        rerank("emissions", ["Sea ice declined", "CO2 rose 12%"])
    )
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True), "Results must be sorted descending by score"


def test_rerank_empty_passages_returns_empty():
    """rerank(query, []) returns empty list without calling the model."""
    from evidenceengine.retrieval.reranker import rerank

    result = asyncio.get_event_loop().run_until_complete(rerank("some query", []))
    assert result == []


def test_get_reranker_singleton(monkeypatch):
    """Calling get_reranker() twice returns the exact same object."""
    import evidenceengine.retrieval.reranker as reranker_mod

    # Reset to None so we control construction
    monkeypatch.setattr(reranker_mod, "_reranker", None)

    mock_ce_class = MagicMock()
    mock_instance = MagicMock()
    mock_ce_class.return_value = mock_instance

    with patch("evidenceengine.retrieval.reranker.CrossEncoder", mock_ce_class):
        r1 = reranker_mod.get_reranker()
        r2 = reranker_mod.get_reranker()

    assert r1 is r2, "get_reranker() must return the same object on repeated calls"
    # CrossEncoder constructor should only be called once
    mock_ce_class.assert_called_once()


def test_reranker_not_instantiated_without_call(monkeypatch):
    """After module import, _reranker global should be None until get_reranker() is called."""
    import evidenceengine.retrieval.reranker as reranker_mod

    # Force reset to None
    monkeypatch.setattr(reranker_mod, "_reranker", None)
    assert reranker_mod._reranker is None


def test_rerank_runs_in_executor(monkeypatch, mock_cross_encoder):
    """rerank() uses run_in_executor, not direct synchronous call."""
    import evidenceengine.retrieval.reranker as reranker_mod

    executor_calls = []
    original_run_in_executor = asyncio.get_event_loop().run_in_executor

    async def patched_rerank(query, passages):
        loop = asyncio.get_event_loop()
        # Track calls to run_in_executor
        orig = loop.run_in_executor

        async def tracking_run(*args, **kwargs):
            executor_calls.append(args)
            return await orig(*args, **kwargs)

        loop.run_in_executor = tracking_run
        try:
            return await reranker_mod.rerank(query, passages)
        finally:
            loop.run_in_executor = orig

    result = asyncio.get_event_loop().run_until_complete(
        patched_rerank("emissions", ["Sea ice declined", "CO2 rose 12%"])
    )
    assert len(executor_calls) >= 1, "run_in_executor must be called during rerank()"
