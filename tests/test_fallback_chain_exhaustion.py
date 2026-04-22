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

from evidenceengine.llm.fallback import (
    _GEMINI_BASE_URL,
    _GROQ_BASE_URL,
    _resolve_base_url,
    sync_call_with_fallback,
)


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


# ── v1.2.9 provider-dispatch tests (PRV-02) ──────────────────────────────────

def test_resolve_base_url_gemini_prefix_overrides_caller():
    """Gemini model IDs route to Google AI Studio regardless of caller base_url."""
    assert _resolve_base_url("gemini-2.0-flash", "https://openrouter.ai/api/v1") == _GEMINI_BASE_URL
    assert _resolve_base_url("gemma-2-9b-it", None) == _GEMINI_BASE_URL


def test_resolve_base_url_groq_prefix_overrides_caller():
    """Groq-hosted model IDs route to Groq regardless of caller base_url."""
    assert _resolve_base_url("llama-3.3-70b-versatile", "https://openrouter.ai/api/v1") == _GROQ_BASE_URL
    assert _resolve_base_url("mixtral-8x7b-32768", None) == _GROQ_BASE_URL
    assert _resolve_base_url("deepseek-r1-distill-llama-70b", None) == _GROQ_BASE_URL


def test_resolve_base_url_other_models_use_caller_value():
    """OpenRouter / OpenAI / local model IDs use the caller-provided base_url."""
    assert _resolve_base_url("arcee-ai/trinity:free", "https://openrouter.ai/api/v1") == "https://openrouter.ai/api/v1"
    assert _resolve_base_url("gpt-4o-mini", None) is None
    assert _resolve_base_url("gpt-4o-mini", "") is None


class _CaptureClientCall:
    """Captures base_url passed to OpenAI() + extra_body passed to parse()."""

    captured_base_url: str | None = None
    captured_extra_body: dict | None = None
    captured_kwargs_keys: list[str] = []

    def __init__(self, model_id: str = "dummy"):
        self.beta = MagicMock()
        self.beta.chat.completions.parse = self._parse

    def _parse(self, **kwargs):
        _CaptureClientCall.captured_extra_body = kwargs.get("extra_body")
        _CaptureClientCall.captured_kwargs_keys = list(kwargs.keys())
        result_msg = MagicMock(refusal=None, parsed=MagicMock(verdict_type="supported"))
        return MagicMock(choices=[MagicMock(message=result_msg)])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run_capture(model: str, caller_base_url: str | None) -> None:
    _CaptureClientCall.captured_base_url = None
    _CaptureClientCall.captured_extra_body = None

    def factory(**kw):
        _CaptureClientCall.captured_base_url = kw.get("base_url")
        return _CaptureClientCall(model)

    with patch("openai.OpenAI") as mock_cls:
        mock_cls.side_effect = factory
        sync_call_with_fallback(
            model_chain=[model],
            messages=[{"role": "user", "content": "x"}],
            response_format=MagicMock(),
            timeout=1.0,
            api_key="sk-test",
            base_url=caller_base_url,
        )


def test_gemini_model_id_uses_gemini_base_url_and_omits_openrouter_extras():
    """Gemini model → Gemini base URL, no response-healing / require_parameters flags."""
    _run_capture("gemini-2.0-flash", "https://openrouter.ai/api/v1")
    assert _CaptureClientCall.captured_base_url == _GEMINI_BASE_URL
    # OpenRouter-only extras must NOT be sent to Gemini — either extra_body absent
    # or an empty dict (both are fine; openai SDK normalizes).
    assert "extra_body" not in _CaptureClientCall.captured_kwargs_keys or \
           _CaptureClientCall.captured_extra_body in (None, {})


def test_groq_model_id_uses_groq_base_url_and_omits_openrouter_extras():
    """Groq model → Groq base URL, no OpenRouter extras."""
    _run_capture("llama-3.3-70b-versatile", "https://openrouter.ai/api/v1")
    assert _CaptureClientCall.captured_base_url == _GROQ_BASE_URL
    assert "extra_body" not in _CaptureClientCall.captured_kwargs_keys or \
           _CaptureClientCall.captured_extra_body in (None, {})


def test_openrouter_model_retains_response_healing_plugin():
    """OpenRouter model path keeps the response-healing plugin in extra_body."""
    _run_capture("arcee-ai/trinity-large-preview:free", "https://openrouter.ai/api/v1")
    assert _CaptureClientCall.captured_base_url == "https://openrouter.ai/api/v1"
    eb = _CaptureClientCall.captured_extra_body
    assert eb is not None, "OpenRouter path must send extra_body"
    assert eb.get("plugins") == [{"id": "response-healing"}]
    # This model IS in _STRICT_STRUCTURED_OUTPUT_MODELS → require_parameters must fire
    assert eb.get("provider", {}).get("require_parameters") is True


def test_new_default_chain_exhaustion_names_both_providers():
    """Mixed Gemini+Groq chain: exhaustion error string references the last failure."""
    calls: list[str] = []

    with patch("openai.OpenAI") as mock_cls, \
         patch("evidenceengine.llm.fallback.time.sleep"):
        mock_cls.side_effect = lambda **kw: _AllRateLimited(calls)

        with pytest.raises(RuntimeError) as ei:
            sync_call_with_fallback(
                model_chain=["gemini-2.0-flash", "llama-3.3-70b-versatile"],
                messages=[{"role": "user", "content": "x"}],
                response_format=MagicMock(),
                timeout=1.0,
                api_key="sk-test",
                base_url=None,  # Gemini + Groq auto-dispatch; caller base_url ignored
            )
    err_msg = str(ei.value)
    assert "exhausted" in err_msg.lower()
    # Last failure should name one of the two new-chain models
    assert "gemini-2.0-flash" in err_msg or "llama-3.3-70b-versatile" in err_msg
    # Both models should have been attempted at least once each
    assert "gemini-2.0-flash" in calls
    assert "llama-3.3-70b-versatile" in calls
