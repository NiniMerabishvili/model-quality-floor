"""
Generic rate-limit backoff for provider calls.

Every real provider.generate() call in this codebase is invoked from exactly
two places: cli.py's call_fn (candidate models) and build_judge_fn's inner
_judge (the judge model). Rather than duplicating retry logic in each of the
five providers/*.py files, this wrapper is applied once at those two call
sites -- any provider, present or future, gets the same backoff behavior for
free just by being called through them.
"""

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimitExhaustedError(Exception):
    """Raised when a provider call still hits a rate limit after all backoff retries."""


def _rate_limit_exception_types() -> tuple[type[Exception], ...]:
    # Groq and xAI's Grok both reuse the openai client (see their provider
    # files) against a different base_url, so openai.RateLimitError covers
    # OpenAI, Groq, Grok, and HuggingFace's router alike -- one import, four
    # vendors. Anthropic and Gemini each have their own SDK exception.
    types: list[type[Exception]] = []
    try:
        import openai

        types.append(openai.RateLimitError)
    except ImportError:
        pass
    try:
        import anthropic

        types.append(anthropic.RateLimitError)
    except ImportError:
        pass
    try:
        from google.api_core.exceptions import ResourceExhausted

        types.append(ResourceExhausted)
    except ImportError:
        pass
    return tuple(types)


_RATE_LIMIT_TYPES = _rate_limit_exception_types()


def is_rate_limit_error(exc: Exception) -> bool:
    if _RATE_LIMIT_TYPES and isinstance(exc, _RATE_LIMIT_TYPES):
        return True
    # Fallback for any SDK not explicitly covered above (or a future one):
    # most HTTP-based SDKs expose the status code directly on the exception.
    return getattr(exc, "status_code", None) == 429


def call_with_rate_limit_backoff(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 2.0,
    context: str = "",
    **kwargs: Any,
) -> T:
    """
    Calls fn(*args, **kwargs). If it raises a rate-limit error (429, or the
    Anthropic/OpenAI/Gemini SDK's specific rate-limit exception), waits with
    exponential backoff (2s, 4s, 8s for the default max_retries=3) and
    retries, up to `max_retries` retries (max_retries + 1 total attempts).

    Any non-rate-limit exception is re-raised immediately and unchanged --
    this wrapper only ever intercepts the rate-limit case, per the provider
    contract in providers/base.py ("must not swallow errors silently").

    Raises RateLimitExhaustedError (chained from the last real exception)
    if every attempt is exhausted, so the caller can choose to skip just
    this one trial instead of crashing the whole run.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not is_rate_limit_error(e):
                raise
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Rate limit hit%s (attempt %d/%d) -- retrying in %.0fs. %s",
                    f" [{context}]" if context else "",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    e,
                )
                time.sleep(delay)

    raise RateLimitExhaustedError(
        f"Rate limit retries exhausted{f' for {context}' if context else ''} "
        f"after {max_retries + 1} attempts. Last error: {last_exc}"
    ) from last_exc
