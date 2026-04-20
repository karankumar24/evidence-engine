"""Synchronous LLM call with model fallback chain.

Designed to run inside asyncio.to_thread() so network I/O never blocks
the uvicorn event loop. Tries each model in order, moving immediately to
the next on timeout, HTTP error, null parse, or refusal.
"""

import logging
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
    "arcee-ai/trinity-large-preview:free",
})


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

    Raises:
        openai.AuthenticationError / PermissionDeniedError — fatal, re-raised.
        RuntimeError — when all models in the chain are exhausted.
    """
    import openai
    from openai import OpenAI as _SyncOpenAI  # noqa: PLC0415

    last_failure = "chain is empty"

    for model in model_chain:
        try:
            with _SyncOpenAI(
                api_key=api_key,
                base_url=base_url or None,
                timeout=timeout,
                max_retries=0,  # we manage the retry chain ourselves
            ) as client:
                # Only route to a provider that strictly honors json_schema for
                # models we KNOW support it end-to-end. For models that only
                # advertise `response_format` (looser support), drop the flag —
                # otherwise OpenRouter 404s with "no providers". The
                # response-healing plugin + `if message.parsed is None` fallback
                # below catch the rare cases where a relaxed model returns junk.
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
            continue

        except openai.APIStatusError as exc:
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

        # Success — log if a fallback model was used
        if model != model_chain[0]:
            logger.info("llm_fallback: succeeded with %s (earlier models failed)", model)
        return message

    raise RuntimeError(f"All models in fallback chain exhausted — last failure: {last_failure}")
