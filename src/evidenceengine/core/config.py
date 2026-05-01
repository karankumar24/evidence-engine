"""Application configuration via Pydantic Settings."""

from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    debug: bool = False
    # LLM provider defaults to Google AI Studio (Gemini 2.0 Flash) as of v1.2.9.
    # Any OpenAI-compatible endpoint works (Gemini OpenAI-compat, Groq, Azure,
    # local vLLM, real OpenAI). `llm_api_key` + `llm_base_url` are authoritative
    # for HTTP routing. `llm_provider` is an informational routing hint used by
    # telemetry and tests. Accepts either LLM_API_KEY (preferred) or legacy
    # OPENAI_API_KEY — LLM_API_KEY wins when both are set.
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    # Routing hint (informational) + provider-specific credentials for the
    # forward-looking live-test + telemetry layers. `llm_api_key` above is what
    # `sync_call_with_fallback` actually uses today; these fields let callers
    # pick the right key per provider without env-var gymnastics.
    llm_provider: str = "gemini"
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    cerebras_api_key: str = Field(default="", validation_alias="CEREBRAS_API_KEY")
    sambanova_api_key: str = Field(default="", validation_alias="SAMBANOVA_API_KEY")
    # Default extraction/classification models target Gemini 2.0 Flash (free
    # tier: 1,500 req/day). Override via EXTRACTION_MODEL / CLASSIFICATION_MODEL
    # env vars; openai SDK rejects empty model names so a sensible default is
    # required. If you point LLM_BASE_URL at real OpenAI, set these to e.g.
    # "gpt-4o-mini" and LLM_API_KEY to an sk-... key.
    extraction_model: str = "gemini-2.0-flash"
    retrieval_top_k_bm25: int = 5
    retrieval_top_k_final: int = 2
    # Dense retrieval: when True, all-MiniLM-L6-v2 embeddings augment BM25 to
    # catch paraphrased evidence. Union of BM25 + dense candidates is reranked,
    # then truncated to retrieval_top_k_final. Set DENSE_RETRIEVAL_ENABLED=false
    # to fall back to BM25-only (e.g. memory-constrained environments).
    dense_retrieval_enabled: bool = True
    # BGE-base-en-v1.5 replaces all-MiniLM-L6-v2: +12 MTEB retrieval points on technical text.
    # Requires query prefix "Represent this sentence for searching relevant passages: " on queries.
    dense_embedding_model: str = "BAAI/bge-base-en-v1.5"
    # Raised from 1500 to cover large ML papers (100-page arXiv papers ~2000-2500 chunks).
    # At 3000 chunks, BGE-base uses ~9MB for embeddings + ~400MB for model = well within 4GB.
    dense_retrieval_max_corpus_size: int = 3000
    # Claim cap: cross-encoder/nli-deberta-v3-small costs ~0.5s per (claim, span) pair.
    # 25 claims × 2 spans × 0.5s = 25s — safely under 60s target.
    # Effective cap = min(page_count * max_claims_per_page, max_claims_absolute).
    max_claims_per_page: int = 4
    max_claims_absolute: int = 25
    # Default reranker: ms-marco-MiniLM-L6-v2 (22M params, pre-downloaded in
    # Docker at build time). Fast on CPU — ~50ms per pair on shared-cpu-2x.
    # BAAI/bge-reranker-v2-m3 (568M, 2.27GB) is the higher-quality alternative
    # but unusable on shared CPU (70s/batch). Override via RERANKER_MODEL env var.
    # revision="" means use the HF default (latest committed weights, no SHA pin).
    # Set RERANKER_MODEL_REVISION to a SHA for reproducibility if pinning matters.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_model_revision: str = ""
    index_dir: str = "./indexes"
    classification_model: str = "gemini-2.0-flash"
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
    # ─── NLI-primary classifier ─────────────────────────────────────────────
    # Production runs ``nli_primary`` only. The historical LLM-primary
    # rollback path (and the NLI second-opinion downgrader that gated it)
    # were deleted in the Phase A simplification refactor. The field is kept
    # for telemetry continuity.
    classifier_backend: Literal["nli_primary"] = "nli_primary"
    # Path to fine-tuned NLI model checkpoint. Empty = use HuggingFace base model.
    # Production: /data/models/scifact-nli (Fly.io volume). Local dev: ./checkpoints/scifact-nli/best
    nli_model_path: str = ""
    # NLI threshold band. Band invariant (locked by test_config.py):
    #   0 < nli_min_confidence_for_verdict
    #     < nli_tiebreaker_threshold
    #     < nli_entailment_supported_threshold
    #   0 < nli_tiebreaker_threshold
    #     < nli_contradiction_contradicted_threshold
    #
    #   * nli_min_confidence_for_verdict (0.50): when max(p_entail, p_neutral,
    #     p_contra) falls below this → needs_review. In practice this threshold
    #     is never hit (NLI softmax always gives some class > 0.50). needs_review
    #     is therefore assigned only by rule-based paths (unresolvable citations,
    #     model load failure), not by the NLI output itself.
    #   * nli_tiebreaker_threshold (0.65): reserved for a future tiebreaker path
    #     where both entail and contra have signal but neither hits 0.80. NOT
    #     currently applied in nli_probs_to_verdict — kept in config so the
    #     band invariant test still exercises the threshold ordering.
    #   * nli_entailment_supported_threshold (0.80): entailment score must
    #     clear this to emit a `supported` verdict on its own.
    #   * nli_contradiction_contradicted_threshold (0.80): symmetric for
    #     `contradicted`.
    nli_tiebreaker_threshold: float = 0.65
    nli_entailment_supported_threshold: float = 0.92   # raised 0.86→0.92 for calibrated precision (DeBERTa overconfident by ~8pp)
    nli_contradiction_contradicted_threshold: float = 0.75  # lowered 0.85→0.75 per 2026-04-30 sweep: +2.4pp 3-class acc, +1.8pp contra recall, FSR unchanged 9.1% (trust preserved per PROJECT.md)
    nli_min_confidence_for_verdict: float = 0.50
    # ── LLM HTTP settings (used by on-demand explanation generation) ──────────
    # The classification hot path is NLI-only. These knobs apply to the
    # on-demand explanation generator (``classification/explanation.py``) which
    # calls ``llm/fallback.sync_call_with_fallback``.
    # OpenAI SDK default timeout is 600s which compounds with rate-limits.
    llm_request_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    # Fallback model chain — comma-separated in .env, parsed to list.
    # When non-empty, the pipeline tries each model in order on timeout/error/refusal.
    # Falls back gracefully to extraction_model / classification_model if unset.
    # Default is the same live chain shipped in fly.toml so local dev gets the
    # same resilience as prod. Audit monthly against /api/v1/models.
    model_fallback_chain: Annotated[list[str], NoDecode] = Field(
        # 4-provider free-tier chain (mirrors Fly production secret).
        # Gemini 2.0 Flash primary (1,500 req/day) → Groq Llama 4 Scout
        # (json_schema-compatible; llama-3.3-70b-versatile is NOT) →
        # SambaNova → Cerebras as deep fallbacks.
        # Override via MODEL_FALLBACK_CHAIN env var (CSV).
        default_factory=lambda: [
            "gemini-2.0-flash",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "sambanova/Meta-Llama-3.3-70B-Instruct",
            "cerebras/qwen-3-235b-a22b-instruct-2507",
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


settings = Settings()
