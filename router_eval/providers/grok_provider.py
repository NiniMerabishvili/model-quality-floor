"""
xAI Grok provider. Requires: pip install openai (xAI's Chat Completions API
is OpenAI-compatible -- same request/response shape, just a different
base_url and model names, so this reuses the openai package instead of a
separate xAI SDK).
Reads GROK_API_KEY from env if no key is passed explicitly.
"""

import os
import time

from .base import ModelProvider, GenerationResult


class GrokProvider(ModelProvider):
    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id, api_key)
        import openai  # imported lazily so the package is only required if this provider is actually used
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
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

        return GenerationResult(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,   # exact count from the API, not estimated
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            raw_response=None,
        )

    def count_tokens(self, text: str) -> int:
        # xAI doesn't expose a local tokenizer through the OpenAI-compatible
        # surface (tiktoken has no Grok encoding), so this always falls back
        # to the same rough per-character estimate the other providers use
        # only as a last resort -- not used for billed cost estimates, only
        # for the --dry-run preview.
        return max(1, len(text) // 4)
