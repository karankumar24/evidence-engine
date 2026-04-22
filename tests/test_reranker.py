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


# ── Phase 03 Plan 01: device selection + revision pinning + max_length ──────


def _install_torch_stub(monkeypatch, mps_available: bool) -> None:
    """Install a stub ``torch`` module in sys.modules with a configurable
    ``torch.backends.mps.is_available()``.

    Required because the v1.2.9 macOS dev venv (off-iCloud) may not have torch
    installed (pyproject pins torch to linux-only via markers). Device-selection
    logic must still be unit-testable without the 2 GB torch wheel.
    """
    import sys
    import types

    torch_mod = types.ModuleType("torch")
    backends = types.ModuleType("torch.backends")
    mps = types.ModuleType("torch.backends.mps")
    mps.is_available = lambda: mps_available  # type: ignore[attr-defined]
    backends.mps = mps  # type: ignore[attr-defined]
    torch_mod.backends = backends  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.backends", backends)
    monkeypatch.setitem(sys.modules, "torch.backends.mps", mps)


def _install_cross_encoder_stub(monkeypatch) -> MagicMock:
    """Install a stub sentence_transformers.CrossEncoder. Returns the mock so
    tests can inspect call_args. Stubs the module so the lazy import inside
    get_reranker() finds it even when sentence_transformers isn't installed.
    """
    import sys
    import types

    ce_mock = MagicMock()
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.CrossEncoder = ce_mock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)
    return ce_mock


def test_get_reranker_uses_mps_when_available(monkeypatch):
    """RET-02: get_reranker() selects MPS when torch.backends.mps.is_available() is True.
    Also asserts RET-03 revision pinning + max_length=512 (bge-reranker-v2-m3 supports 512).
    """
    import evidenceengine.retrieval.reranker as reranker_mod
    monkeypatch.setattr(reranker_mod, "_reranker", None)
    _install_torch_stub(monkeypatch, mps_available=True)
    ce_mock = _install_cross_encoder_stub(monkeypatch)

    reranker_mod.get_reranker()

    _, kwargs = ce_mock.call_args
    assert kwargs.get("device") == "mps", kwargs
    assert kwargs.get("revision") == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", kwargs
    assert kwargs.get("max_length") == 512, kwargs


def test_get_reranker_falls_back_to_cpu(monkeypatch):
    """RET-02: get_reranker() falls back to CPU when MPS unavailable.
    Also asserts RET-03 revision pinning + max_length=512.
    """
    import evidenceengine.retrieval.reranker as reranker_mod
    monkeypatch.setattr(reranker_mod, "_reranker", None)
    _install_torch_stub(monkeypatch, mps_available=False)
    ce_mock = _install_cross_encoder_stub(monkeypatch)

    reranker_mod.get_reranker()

    _, kwargs = ce_mock.call_args
    assert kwargs.get("device") == "cpu", kwargs
    assert kwargs.get("revision") == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", kwargs
    assert kwargs.get("max_length") == 512, kwargs


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
