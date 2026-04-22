"""Verify LLM_* env vars are preferred over legacy OPENAI_* names, and that
the openai_api_key/openai_base_url back-compat properties still work.
"""

from evidenceengine.core.config import Settings


def _settings(env: dict[str, str]) -> Settings:
    # `_env_file=None` disables .env loading so env dict is the sole source.
    return Settings(_env_file=None, **{})  # type: ignore[call-arg]


def test_llm_api_key_wins_over_openai_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy")
    monkeypatch.setenv("LLM_API_KEY", "sk-new")
    s = Settings(_env_file=None)
    assert s.llm_api_key == "sk-new"
    assert s.openai_api_key == "sk-new"  # back-compat property


def test_legacy_openai_api_key_still_works(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-v1-legacy")
    s = Settings(_env_file=None)
    assert s.llm_api_key == "sk-or-v1-legacy"
    assert s.openai_api_key == "sk-or-v1-legacy"


def test_llm_base_url_wins_over_openai_base_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    s = Settings(_env_file=None)
    assert s.llm_base_url == "https://openrouter.ai/api/v1"
    assert s.openai_base_url == "https://openrouter.ai/api/v1"


def test_defaults_are_empty(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    s = Settings(_env_file=None)
    assert s.llm_api_key == ""
    assert s.llm_base_url == ""
    assert s.openai_api_key == ""
    assert s.openai_base_url == ""


# ── v1.2.9 provider-migration additions (PRV-01) ─────────────────────────────

def test_llm_provider_default_is_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = Settings(_env_file=None)
    assert s.llm_provider == "gemini"


def test_llm_provider_override_persists(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    s = Settings(_env_file=None)
    assert s.llm_provider == "groq"


def test_gemini_api_key_from_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "xxx")
    s = Settings(_env_file=None)
    assert s.gemini_api_key == "xxx"


def test_groq_api_key_from_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "yyy")
    s = Settings(_env_file=None)
    assert s.groq_api_key == "yyy"


def test_default_fallback_chain_is_gemini_then_groq(monkeypatch):
    monkeypatch.delenv("MODEL_FALLBACK_CHAIN", raising=False)
    s = Settings(_env_file=None)
    assert s.model_fallback_chain == ["gemini-2.0-flash", "llama-3.3-70b-versatile"]


def test_model_fallback_chain_env_override_still_parses(monkeypatch):
    # Regression: comma-separated override via env must still parse to a list
    # (parse_model_chain validator must remain intact after default change).
    monkeypatch.setenv("MODEL_FALLBACK_CHAIN", "a-model,b-model , c-model")
    s = Settings(_env_file=None)
    assert s.model_fallback_chain == ["a-model", "b-model", "c-model"]
