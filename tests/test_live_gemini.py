"""Live integration test for Gemini 2.0 Flash via sync_call_with_fallback.

Skipped unless GEMINI_API_KEY is set in the environment. Run explicitly with:

    export GEMINI_API_KEY=<key-from-aistudio.google.com/apikey>
    pytest -m live tests/test_live_gemini.py -x -v

Default pytest runs (`pytest tests/`) exclude this file via
`addopts = -m 'not live'` in pyproject.toml, so CI and local dev runs stay
hermetic.

Purpose: proves PRV-02 (Phase 1 Provider Migration) — round-trips a real
structured-output call through Gemini 2.0 Flash using the v1.2.9 dispatch
path in `sync_call_with_fallback`. No other Gemini-specific adapter code
needed: the OpenAI-compatible wire protocol + the per-model-prefix dispatch
in fallback.py handle routing.
"""
import os

import pytest
from pydantic import BaseModel

from evidenceengine.llm.fallback import sync_call_with_fallback


class _TinyFact(BaseModel):
    capital: str
    country: str


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — live Gemini test skipped",
)
def test_gemini_structured_output_roundtrip() -> None:
    """Gemini 2.0 Flash returns a parsed Pydantic object for a trivial fact query.

    Regression guard: a 400 here usually means an OpenRouter-only extra_body key
    (response-healing plugin or provider.require_parameters) leaked through the
    dispatch gate in fallback.py — check `_resolve_base_url` + the
    `is_openrouter` conditional.
    """
    api_key = os.environ["GEMINI_API_KEY"]
    messages = [
        {"role": "system", "content": "You are a terse fact responder. Return only structured JSON."},
        {"role": "user", "content": "What is the capital of France? Respond as JSON."},
    ]

    message = sync_call_with_fallback(
        model_chain=["gemini-2.0-flash"],
        messages=messages,
        response_format=_TinyFact,
        timeout=30.0,
        api_key=api_key,
        base_url=None,  # dispatch auto-resolves Gemini URL via _resolve_base_url
    )

    assert message.parsed is not None, "Gemini returned null parsed — check schema or key"
    assert isinstance(message.parsed, _TinyFact)
    assert "paris" in message.parsed.capital.lower(), (
        f"expected 'Paris' in capital, got: {message.parsed.capital!r}"
    )
    assert message.parsed.country, "country field must be non-empty"
