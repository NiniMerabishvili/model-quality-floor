"""
OpenAI provider (GPT models). Requires: pip install openai

This file is here specifically to prove the point: the harness was originally
written for two vendors, but adding a third took one file, ~30 lines, and zero
changes anywhere else in the codebase. That's the generalization payoff.
"""

import os
import time

from .base import ModelProvider, GenerationResult


class OpenAIProvider(ModelProvider):
    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id, api_key)
        import openai
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

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
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
            raw_response=None,
        )

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model_id)
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)
