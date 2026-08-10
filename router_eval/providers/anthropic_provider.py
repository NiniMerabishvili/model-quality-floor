"""
Anthropic provider. Requires: pip install anthropic
Reads ANTHROPIC_API_KEY from env if no key is passed explicitly (BYOK --
this key is the user's own, never stored or transmitted anywhere by this tool).
"""

import os
import time

from .base import ModelProvider, GenerationResult


class AnthropicProvider(ModelProvider):
    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id, api_key)
        import anthropic  # imported lazily so the package is only required if this provider is actually used
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, system_prompt: str, user_input: str, max_tokens: int = 1024) -> GenerationResult:
        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(block.text for block in response.content if block.type == "text")
        return GenerationResult(
            text=text,
            input_tokens=response.usage.input_tokens,   # exact count from the API, not estimated
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    def count_tokens(self, text: str) -> int:
        # Anthropic exposes a token counting endpoint; falls back to a rough
        # estimate only if that call fails (e.g. no network in a --dry-run
        # context), and the estimate is clearly not a real API call.
        try:
            result = self._client.messages.count_tokens(
                model=self.model_id,
                messages=[{"role": "user", "content": text}],
            )
            return result.input_tokens
        except Exception:
            return max(1, len(text) // 4)  # rough fallback, not used for billed cost estimates
