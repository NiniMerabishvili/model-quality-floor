"""
Groq provider. Requires: pip install openai (Groq's API is OpenAI-compatible
-- same request/response shape, just a different base_url and model names,
so this reuses the openai package instead of the groq-sdk package).

Not to be confused with Grok (xAI's model, providers/grok_provider.py) --
Groq is the LPU hardware inference company hosting Llama/GPT-OSS/etc.
directly on its own chips.

Reads GROQ_API_KEY from env if no key is passed explicitly.
"""

import os
import time

from .base import ModelProvider, GenerationResult


class GroqProvider(ModelProvider):
    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id, api_key)
        import openai  # imported lazily so the package is only required if this provider is actually used
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
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
        # Groq hosts several model families (Llama, GPT-OSS, Qwen) with
        # different tokenizers and doesn't expose a local tokenizer endpoint
        # through this OpenAI-compatible surface, so this falls back to the
        # same rough per-character estimate the other providers use only as
        # a last resort -- not used for billed cost estimates, only for the
        # --dry-run preview.
        return max(1, len(text) // 4)
