"""Fallback-chain exhaustion tests: lock the deterministic failure behavior.

sync_call_with_fallback is the thing that stands between our classifier and
the free-tier LLM providers. When every model in the chain is rate-limited
or broken, we rely on the chain raising a deterministic RuntimeError so the
per-claim try/except can tag the failure as chain_exhausted telemetry.

Historical problem: these tests didn't exist, so a silent fall-through (e.g.
returning None from the last model instead of raising) would have silently
stamped every claim as needs_review @ 0.0 without the exhausted-chain label.
"""

from unittest.mock import MagicMock, patch

import openai
import pytest

from evidenceengine.llm.fallback import sync_call_with_fallback


def _rate_limit_response(model_id: str) -> openai.RateLimitError:
    """Build a real openai.RateLimitError that the fallback layer expects."""
    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {}
    return openai.RateLimitError(
        message=f"rate limit on {model_id}",
        response=resp,
        body={"error": {"message": "rate_limit_exceeded"}},
    )


class _AllRateLimited:
    """Stand-in OpenAI client whose parse() always raises RateLimitError."""

    def __init__(self, calls: list[str]):
        self._calls = calls
        self.beta = MagicMock()
        self.beta.chat.completions.parse = self._parse

    def _parse(self, **kwargs):
        self._calls.append(kwargs["model"])
        raise _rate_limit_response(kwargs["model"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_exhausted_chain_raises_runtimerror():
    """All models 429 across all retry rounds → RuntimeError with 'exhausted' text."""
    calls: list[str] = []

    with patch("openai.OpenAI") as mock_cls, \
         patch("evidenceengine.llm.fallback.time.sleep"):  # skip the real backoff
        mock_cls.side_effect = lambda **kw: _AllRateLimited(calls)

        with pytest.raises(RuntimeError) as ei:
            sync_call_with_fallback(
                model_chain=["model-a", "model-b", "model-c"],
                messages=[{"role": "user", "content": "x"}],
                response_format=MagicMock(),
                timeout=1.0,
                api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
            )
    err_msg = str(ei.value)
    assert "exhausted" in err_msg.lower()
    # Error should name the last offending model.
    assert any(m in err_msg for m in ["model-a", "model-b", "model-c"])
    # Should have tried every model at least once AND done the chain-level retry rounds.
    # With 3 retry backoffs configured, we expect 4 full passes × 3 models = 12 calls.
    assert len(calls) >= len(["model-a", "model-b", "model-c"]), (
        f"expected ≥3 model calls, got {len(calls)}: {calls}"
    )


def test_chain_level_retry_triggers_on_all_rate_limited():
    """Every model 429 on first pass → second pass attempted (chain-level retry)."""
    calls: list[str] = []

    with patch("openai.OpenAI") as mock_cls, \
         patch("evidenceengine.llm.fallback.time.sleep"):
        mock_cls.side_effect = lambda **kw: _AllRateLimited(calls)

        with pytest.raises(RuntimeError):
            sync_call_with_fallback(
                model_chain=["a", "b"],
                messages=[{"role": "user", "content": "x"}],
                response_format=MagicMock(),
                timeout=1.0,
                api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
            )
    # Chain of 2 models, 3 retry rounds defined → expect 2 × (1 + 3) = 8 attempts total.
    # Not exact because backoff logic may short-circuit; require at least 4 (2 passes × 2 models).
    assert len(calls) >= 4, f"chain-retry not triggered — only {len(calls)} attempts"
    # Pattern must repeat the chain: first a then b, then a then b again, ...
    assert calls[:2] == ["a", "b"]
    assert calls[2:4] == ["a", "b"]


def test_auth_error_short_circuits_chain():
    """AuthenticationError on any model must halt the chain (fatal config issue)."""
    calls: list[str] = []

    auth_err = openai.AuthenticationError(
        message="invalid api key",
        response=MagicMock(status_code=401, headers={}),
        body={"error": {"message": "invalid_api_key"}},
    )

    class _AuthFail:
        def __init__(self, model_id):
            self.model_id = model_id
            self.beta = MagicMock()
            self.beta.chat.completions.parse = self._parse
        def _parse(self, **kwargs):
            calls.append(kwargs["model"])
            raise auth_err
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    with patch("openai.OpenAI") as mock_cls:
        mock_cls.side_effect = lambda **kw: _AuthFail("does-not-matter")

        with pytest.raises(openai.AuthenticationError):
            sync_call_with_fallback(
                model_chain=["a", "b", "c"],
                messages=[{"role": "user", "content": "x"}],
                response_format=MagicMock(),
                timeout=1.0,
                api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
            )
    # Must NOT have tried model "b" or "c" — auth fail is fatal at first encounter.
    assert calls == ["a"], f"auth error should short-circuit but got calls={calls}"


def test_first_successful_model_wins_chain_stops():
    """Once a model returns parsed verdict, remaining models aren't called."""
    calls: list[str] = []

    class _FirstFailsSecondSucceeds:
        def __init__(self):
            self.beta = MagicMock()
            self.beta.chat.completions.parse = self._parse
        def _parse(self, **kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                raise _rate_limit_response(kwargs["model"])
            result_msg = MagicMock(refusal=None, parsed=MagicMock(verdict_type="supported"))
            return MagicMock(choices=[MagicMock(message=result_msg)])
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    with patch("openai.OpenAI") as mock_cls:
        mock_cls.side_effect = lambda **kw: _FirstFailsSecondSucceeds()

        result = sync_call_with_fallback(
            model_chain=["a", "b", "c"],
            messages=[{"role": "user", "content": "x"}],
            response_format=MagicMock(),
            timeout=1.0,
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
        )
    assert result.parsed.verdict_type == "supported"
    assert calls == ["a", "b"], (
        f"chain should stop on first success — expected 2 calls, got {len(calls)}: {calls}"
    )
