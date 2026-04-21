"""Application configuration via Pydantic Settings."""

from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    debug: bool = False
    # LLM provider is OpenRouter by default; any OpenAI-compatible endpoint works
    # (Azure, local vLLM, real OpenAI). Accepts either LLM_API_KEY (preferred) or
    # legacy OPENAI_API_KEY — LLM_API_KEY wins when both are set.
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    # NOTE: the default string is a placeholder only — in production we set
    # EXTRACTION_MODEL to an OpenRouter free-tier id (arcee-ai/trinity-...).
    # Kept as "gpt-4o-mini" because openai SDK rejects empty model names; any
    # call with this default WILL fail unless LLM_BASE_URL points to OpenAI
    # proper and the caller has credits. Treat it as "must be overridden".
    extraction_model: str = "gpt-4o-mini"
    retrieval_top_k_bm25: int = 10
    retrieval_top_k_final: int = 5
    # Claim cap strategy: scale with document size so small docs aren't
    # over-capped and large docs don't blow the free-tier LLM budget.
    # Effective cap = min(page_count * max_claims_per_page, max_claims_absolute).
    max_claims_per_page: int = 10
    max_claims_absolute: int = 80
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    index_dir: str = "./indexes"
    classification_model: str = "gpt-4o-mini"
    # Trust-model thresholds — the interaction between these two numbers is
    # the heart of v1.2.6's self-verify defense. DO NOT change one without
    # updating the other and the regression test that locks their band.
    #
    #   * verdict_needs_review_threshold (0.70): any predicted verdict below
    #     this gets overridden to needs_review. Calibration instructions in
    #     the SYSTEM_PROMPT teach the model to keep uncertain claims < 0.70.
    #
    #   * self_verify_supported_cap (0.80): when ALL evidence came from the
    #     same document as the claim (self-verify mode), SUPPORTED verdicts
    #     are hard-capped at 0.80. This leaves exactly 0.10 of headroom above
    #     the threshold so high-quality self-corroboration CAN still pass,
    #     but anything the model was overconfident about gets squeezed
    #     through the needs_review safety net.
    #
    # Invariant: 0 < verdict_needs_review_threshold < self_verify_supported_cap < 1
    verdict_needs_review_threshold: float = 0.70
    self_verify_supported_cap: float = 0.80
    verdict_prompt_version: str = "v1"
    # Local NLI second-opinion: when True, a locally-hosted DeBERTa-v3-NLI
    # model runs AFTER the LLM verdict. High-confidence disagreement
    # (contradiction found by NLI but LLM said supported, or vice versa)
    # forces the verdict to needs_review. It never upgrades. Disable in
    # environments without transformers/torch installed — the classifier
    # detects that and falls back cleanly either way.
    nli_second_opinion_enabled: bool = True
    # LLM HTTP client safety rails — OpenAI SDK default timeout is 600s which
    # compounds with OpenRouter free-tier rate-limits into multi-minute hangs.
    llm_request_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # Fallback model chain — comma-separated in .env, parsed to list.
    # When non-empty, the pipeline tries each model in order on timeout/error/refusal.
    # Falls back gracefully to extraction_model / classification_model if unset.
    # Default is the same live chain shipped in fly.toml so local dev gets the
    # same resilience as prod. Audit monthly against /api/v1/models.
    model_fallback_chain: Annotated[list[str], NoDecode] = Field(
        # Audited against OpenRouter /api/v1/models + live 200/429 probes on
        # 2026-04-21. Diversity across providers so a single provider's daily
        # cap doesn't kill the whole chain. Strict-structured-output models
        # (arcee, nvidia) lead; gemma at the tail as a recovery slot.
        default_factory=lambda: [
            "arcee-ai/trinity-large-preview:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "minimax/minimax-m2.5:free",
            "google/gemma-4-31b-it:free",
        ]
    )
    # Per-model timeout when the fallback chain has >1 model — shorter so failures
    # don't pile up (full chain still gets N × llm_fallback_timeout_seconds budget).
    llm_fallback_timeout_seconds: float = 30.0

    @field_validator("model_fallback_chain", mode="before")
    @classmethod
    def parse_model_chain(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v or []  # type: ignore[return-value]
    cors_origins: Annotated[list[str], NoDecode] = ["http://127.0.0.1:8000", "http://localhost:8000"]
    sentry_dsn: str = ""

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v  # type: ignore[return-value]

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, v: object) -> str:
        # Managed Postgres providers (Fly, Heroku, Supabase) hand out URLs
        # that asyncpg can't consume directly. Normalize:
        #   - `postgres://`  → `postgresql+asyncpg://`
        #   - `postgresql://` (no driver) → `postgresql+asyncpg://`
        #   - strip `sslmode=…`  (psycopg2 flag; asyncpg uses `ssl=` instead,
        #     and Fly's internal `.flycast` network is already encrypted).
        if not isinstance(v, str):
            return v  # type: ignore[return-value]
        url = v
        if url.startswith("postgres://"):
            url = "postgresql+asyncpg://" + url[len("postgres://"):]
        elif url.startswith("postgresql://") and "+asyncpg" not in url.split("://", 1)[0]:
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
        # Translate psycopg2 `sslmode=…` → asyncpg `ssl=…`. asyncpg rejects the
        # `sslmode` kwarg but accepts the same string values under `ssl`.
        # Without this, Fly internal postgres (no TLS) blows up with
        # ConnectionResetError because asyncpg defaults to ssl=prefer.
        if "sslmode=" in url or "ssl=" in url:
            from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
            parts = urlsplit(url)
            new_query: list[tuple[str, str]] = []
            seen_ssl = False
            for k, val in parse_qsl(parts.query, keep_blank_values=True):
                if k == "sslmode":
                    new_query.append(("ssl", val))
                    seen_ssl = True
                elif k == "ssl":
                    new_query.append((k, val))
                    seen_ssl = True
                else:
                    new_query.append((k, val))
            if not seen_ssl:
                new_query.append(("ssl", "disable"))
            url = urlunsplit(parts._replace(query=urlencode(new_query)))
        return url

    # Back-compat shims: existing code references settings.openai_api_key /
    # settings.openai_base_url. Keep those names working after the LLM_* rename.
    @property
    def openai_api_key(self) -> str:
        return self.llm_api_key

    @property
    def openai_base_url(self) -> str:
        return self.llm_base_url


settings = Settings()
