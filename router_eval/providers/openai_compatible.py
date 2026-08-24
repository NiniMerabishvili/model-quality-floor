"""
Shared base for any vendor exposing an OpenAI-compatible Chat Completions
API -- same request/response shape as OpenAI's own /v1/chat/completions,
just a different base_url and API key. OpenAI itself, Groq, xAI's Grok, and
Hugging Face's Inference Providers router all qualify.

This is the one place that request/response logic lives, instead of being
copy-pasted into four separate providers/*.py files. Concrete providers
only need to set two class attributes (base_url, env_var) and, optionally,
override count_tokens() if the vendor exposes a real tokenizer (see
OpenAIProvider, the one exception).
"""

import os
import time

from .base import GenerationResult, ModelProvider


class OpenAICompatibleProvider(ModelProvider):
    base_url: str = ""
    env_var: str = ""

    def __init__(self, model_id: str, api_key: str | None = None) -> None:
        super().__init__(model_id, api_key)
        if not self.base_url or not self.env_var:
            raise NotImplementedError(
                f"{type(self).__name__} must set non-empty 'base_url' and 'env_var' "
                f"class attributes before it can be instantiated."
            )
        import openai  # imported lazily so the package is only required if this provider is actually used

        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get(self.env_var),
            base_url=self.base_url,
        )

    def generate(self, system_prompt: str, user_input: str, max_tokens: int = 1024) -> GenerationResult:
        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.choices[0].message.content
        if text is None:
            # The SDK types this as Optional (e.g. a tool-call-only response
            # with no text part) -- surface it loudly rather than silently
            # returning an empty/garbage string, per the "don't silently
            # fail" contract in providers/base.py.
            raise ValueError(f"{type(self).__name__} ({self.model_id}) returned no text content in its response.")
        if response.usage is None:
            raise ValueError(f"{type(self).__name__} ({self.model_id}) response had no token usage data.")

        return GenerationResult(
            text=text,
            input_tokens=response.usage.prompt_tokens,  # exact count from the API, not estimated
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            raw_response=None,
        )

    def count_tokens(self, text: str) -> int:
        # Rough fallback shared by every vendor here that doesn't expose its
        # own tokenizer -- not used for billed cost estimates, only for the
        # --dry-run preview. Override this in a subclass if the vendor does
        # expose one (see OpenAIProvider.count_tokens).
        return max(1, len(text) // 4)
