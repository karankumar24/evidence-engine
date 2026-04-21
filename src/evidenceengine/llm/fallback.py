"""Synchronous LLM call with model fallback chain.

Designed to run inside asyncio.to_thread() so network I/O never blocks
the uvicorn event loop. Tries each model in order, moving immediately to
the next on timeout, HTTP error, null parse, or refusal. After a full
chain pass, if at least one model returned 429 (upstream rate limit) and
none succeeded, sleep with exponential backoff and retry the chain — gives
:free providers time to recover.
"""

import logging
import random
import time
from typing import Any, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# HTTP status codes that mean "try the next model" rather than "fatal error".
# 404 is included because OpenRouter returns it when a model name is not
# available (free models rotate in/out), and we want to fall through rather
# than abort the whole run.
_RETRIABLE_CODES = frozenset({400, 404, 408, 429, 502, 503, 529})

# Models that ADVERTISE true OpenAI structured-output support
# (`structured_outputs` in their OpenRouter `supported_parameters`).
# For these we set `provider.require_parameters: true` so OpenRouter only
# routes to providers that honor `json_schema`. For OTHER models (those that
# only have `response_format`) we DROP the flag — otherwise OpenRouter 404s
# saying no provider supports the combination, even though the model itself
# can usually be coaxed into producing JSON via the response-healing plugin.
# Audit monthly against `https://openrouter.ai/api/v1/models` (filter for
# `:free` ids whose `supported_parameters` contains `structured_outputs`).
_STRICT_STRUCTURED_OUTPUT_MODELS = frozenset({
    # Audited against OpenRouter /api/v1/models on 2026-04-21 — these carry
    # `structured_outputs` in their supported_parameters (not just
    # `response_format`). Setting provider.require_parameters:true here keeps
    # OpenRouter from routing to providers that silently drop the strict mode.
    "arcee-ai/trinity-large-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
})

# Chain-level retry: when an entire pass returns "every model 429'd",
# the upstream :free providers are simultaneously throttled. Sleep and
# retry the whole chain a few times — most rate-limits recover in 5-15s.
# Total worst-case wallclock per call ≈ sum(_RETRY_BACKOFFS) seconds + jitter.
_RETRY_BACKOFFS = (5.0, 15.0, 30.0)  # 3 extra rounds = ~50s extra wait


def sync_call_with_fallback(
    model_chain: list[str],
    messages: list[dict],
    response_format: Type[BaseModel],
    timeout: float,
    api_key: str,
    base_url: str | None,
) -> Any:
    """Try each model in model_chain. Return first successful parsed message.

    A "successful" message has parsed != None and refusal == falsy.
    Moves immediately to the next model on:
      - APITimeoutError (timeout)
      - RateLimitError (429)
      - APIStatusError with retriable status code (400, 502, 503, etc.)
      - message.refusal is set
      - message.parsed is None

    If a full chain pass had at least one 429 and no success, sleeps and retries
    the entire chain up to len(_RETRY_BACKOFFS) more times.

    Raises:
        openai.AuthenticationError / PermissionDeniedError — fatal, re-raised.
        RuntimeError — when all models in the chain are exhausted across all rounds.
    """
    import openai
    from openai import OpenAI as _SyncOpenAI  # noqa: PLC0415

    last_failure = "chain is empty"

    # Outer retry: round 0 is the immediate pass; subsequent rounds happen only
    # if the previous pass had any 429 (upstream rate-limit recovery).
    rate_limited_count = 0
    for round_idx in range(len(_RETRY_BACKOFFS) + 1):
        if round_idx > 0:
            if rate_limited_count == 0:
                # Previous pass exhausted the chain without any 429 → no point sleeping.
                break
            backoff = _RETRY_BACKOFFS[round_idx - 1] + random.uniform(0, 2.0)
            logger.warning(
                "llm_fallback: full chain rate-limited (round %d, %d/%d models 429) — "
                "sleeping %.1fs then retrying",
                round_idx, rate_limited_count, len(model_chain), backoff,
            )
            time.sleep(backoff)
        rate_limited_count = 0

        for model in model_chain:
            try:
                with _SyncOpenAI(
                    api_key=api_key,
                    base_url=base_url or None,
                    timeout=timeout,
                    max_retries=0,  # we manage the retry chain ourselves
                ) as client:
                    extra_body: dict[str, Any] = {
                        "plugins": [{"id": "response-healing"}],
                    }
                    if model in _STRICT_STRUCTURED_OUTPUT_MODELS:
                        extra_body["provider"] = {"require_parameters": True}

                    completion = client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        extra_body=extra_body,
                    )
                message = completion.choices[0].message

            except openai.APITimeoutError:
                logger.warning("llm_fallback: %s timed out (%.0fs) — next model", model, timeout)
                last_failure = f"timeout on {model}"
                continue

            except openai.RateLimitError:
                logger.warning("llm_fallback: %s rate-limited (429) — next model", model)
                last_failure = f"rate_limit on {model}"
                rate_limited_count += 1
                continue

            except openai.APIStatusError as exc:
                if exc.status_code == 429:
                    rate_limited_count += 1
                if exc.status_code in _RETRIABLE_CODES:
                    logger.warning(
                        "llm_fallback: %s returned HTTP %d — next model", model, exc.status_code
                    )
                    last_failure = f"HTTP {exc.status_code} on {model}"
                    continue
                # 401 AuthenticationError, 403 PermissionDenied — fatal config issues
                raise

            except Exception as exc:  # noqa: BLE001
                logger.warning("llm_fallback: %s unexpected error (%s) — next model", model, exc)
                last_failure = f"error on {model}: {exc}"
                continue

            if message.refusal:
                logger.warning("llm_fallback: %s refused — next model", model)
                last_failure = f"refusal on {model}"
                continue

            if message.parsed is None:
                logger.warning("llm_fallback: %s null parsed response — next model", model)
                last_failure = f"null_parse on {model}"
                continue

            # Success — log if a fallback model or retry round was used
            if round_idx > 0 or model != model_chain[0]:
                logger.info(
                    "llm_fallback: succeeded with %s (round %d, earlier models/rounds failed)",
                    model, round_idx,
                )
            return message

    raise RuntimeError(f"All models in fallback chain exhausted — last failure: {last_failure}")
