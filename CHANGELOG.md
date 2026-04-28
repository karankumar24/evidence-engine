# Changelog

All notable changes to EvidenceEngine. Format: [Keep a Changelog](https://keepachangelog.com/),
versioning: [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Domain pivot**: EvidenceEngine retargets to biomedical and clinical research
  papers as the sole supported domain. ML/CS support deferred indefinitely. NLI is
  fine-tuned on SciFact (biomedical) and degrades 15-25pp on ML papers; aligning
  scope with calibration. See `tasks/lessons.md` lesson 59.
- Upload UX adds a subtitle signaling biomedical optimization (PubMed Central,
  PLOS, Frontiers, BMC, NEJM as recommended sources).

### Fixed
- Classification pipeline no longer short-circuits to needs_review when retrieval
  has produced evidence spans (commit `c1c6dac`). Was masking real NLI verdicts
  whenever any citation anchor failed to resolve.
- Retrieval pipeline no longer falls back to searching wrong cited sources when
  citations fail to resolve (commit `ef2afee`). Was producing high-confidence
  false-contradicted verdicts on claims that referenced sources not in the packet.

### Removed
- ML/CS test fixtures: `attention_paper.pdf`, `bert_paper.pdf`,
  `gpt4_technical_report.pdf`, `mistral_paper.pdf`, `qlora_paper.pdf`,
  `t5_paper.pdf`, `whisper_paper.pdf` (~17.5 MB).
- ML-targeted training scripts: `scripts/prepare_ml_nli_data.py`,
  `scripts/fine_tune_ml_nli.py`. SciNLI fine-tune no longer on roadmap.
- ML smoke-test script: `scripts/test_new_pdfs.sh`.
- Trimmed `_KNOWN_JOINS` in `claim_extractor.py` — removed ML-specific entries
  (taskspecific, languagemodel, downstreamtasks, etc.). Universal regex Pass 1.6
  handles the general case.

## [v1.2.9] — 2026-04-21 (in progress)

### Changed
- **LLM provider migration**: Default provider swapped from OpenRouter `:free` chain
  (observed 98% failure rate as of 2026-04-21) to Google AI Studio Gemini 2.0 Flash
  (primary) + Groq Llama 3.3 70B (fallback). Both are OpenAI-compatible endpoints so
  no caller code changed.
- `model_fallback_chain` default is now `["gemini-2.0-flash", "llama-3.3-70b-versatile"]`.
- `extraction_model` and `classification_model` defaults now `gemini-2.0-flash`.
- `.env.example` rewritten with Gemini-first config; legacy OpenRouter values preserved
  as commented migration reference.

### Added
- `Settings.llm_provider` (default `gemini`), `Settings.gemini_api_key`, `Settings.groq_api_key`.
- Per-model provider dispatch in `src/evidenceengine/llm/fallback.py` routes Gemini/Groq
  model IDs to their official base URLs without caller changes:
  - `gemini-*` / `gemma-*` → `https://generativelanguage.googleapis.com/v1beta/openai/`
  - `llama-*` / `llama3-*` / `mixtral-*` / `deepseek-*` → `https://api.groq.com/openai/v1`
  - Everything else → caller-provided `base_url` (OpenAI / OpenRouter / Azure / local vLLM)
- OpenRouter-specific `extra_body` keys (`plugins: response-healing`,
  `provider.require_parameters`) now gated behind an `"openrouter.ai" in base_url` check —
  they were 400'ing on Gemini and Groq endpoints when the chain tried them.
- `tests/test_live_gemini.py` — `pytest -m live` real structured-output round-trip (skipped
  by default; requires `GEMINI_API_KEY`).
- `live` pytest marker registered in `pyproject.toml` with default `addopts = -m 'not live'`
  so hermetic runs stay hermetic.
- 7 new unit tests in `tests/test_fallback_chain_exhaustion.py` covering `_resolve_base_url`
  + OpenRouter-extras gating + mixed-chain exhaustion.

### Preserved (backward compatibility)
- `llm_api_key` (AliasChoices: `LLM_API_KEY`, `OPENAI_API_KEY`) unchanged.
- `llm_base_url` (AliasChoices: `LLM_BASE_URL`, `OPENAI_BASE_URL`) unchanged.
- `openai_api_key` / `openai_base_url` back-compat properties unchanged.
- `sync_call_with_fallback` signature unchanged — callers (`classifier.py`,
  `claim_extractor.py`, `orchestrator.py`) untouched.
- `_RETRIABLE_CODES`, `_RETRY_BACKOFFS`, `_STRICT_STRUCTURED_OUTPUT_MODELS` unchanged.
- All non-LLM config (thresholds, timeouts, retrieval tuning, NLI flag) untouched.

### Migration (v1.2.8 → v1.2.9)
Existing self-hosters: your current `.env` keeps working (all legacy env vars still read).
To adopt the new defaults:

1. Create a Gemini key: https://aistudio.google.com/apikey (free; 1,500 req/day)
2. Create a Groq key: https://console.groq.com/keys (free; 14,400 req/day)
3. Update `.env`:
   ```
   LLM_PROVIDER=gemini
   LLM_API_KEY=<gemini-key>
   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   GEMINI_API_KEY=<gemini-key>
   GROQ_API_KEY=<groq-key>
   EXTRACTION_MODEL=gemini-2.0-flash
   CLASSIFICATION_MODEL=gemini-2.0-flash
   MODEL_FALLBACK_CHAIN=gemini-2.0-flash,llama-3.3-70b-versatile
   ```
4. Verify: `pytest -m live tests/test_live_gemini.py` (after exporting env vars).

**Rollback**: revert the Phase 1 commits — no data migration needed.
