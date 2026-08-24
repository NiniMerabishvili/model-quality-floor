"""
Hugging Face Inference Providers. Requires: pip install openai (HF's router
at https://router.huggingface.co/v1 is OpenAI-compatible -- same request/
response shape as OpenAI, just a different base_url, so this reuses the
openai package rather than a separate HF SDK).

Model IDs are "owner/repo" Hub names, optionally with a routing suffix:
":cheapest" picks the lowest-priced live backend for that model at request
time, ":fastest" (the default if no suffix is given) picks the lowest-
latency one, or a specific provider name (e.g. ":together") pins one.
Because ":cheapest"/":fastest" resolve to a *different underlying provider*
per call depending on live pricing/latency, the per-token price billed for
a given call can vary -- configs/models.yaml's price for these entries is
the cheapest live provider seen at the time it was checked, not a fixed
contractual rate the way a single vendor's own pricing page is.

Reads HF_TOKEN from env if no key is passed explicitly.
"""

import os
import time

from .base import ModelProvider, GenerationResult


class HuggingFaceProvider(ModelProvider):
    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id, api_key)
        import openai  # imported lazily so the package is only required if this provider is actually used
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("HF_TOKEN"),
            base_url="https://router.huggingface.co/v1",
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
        # No single local tokenizer applies across every open-weight model
        # this router can point at (Qwen, Llama, Mistral, etc. each use
        # different tokenizers), so this always falls back to the same
        # rough per-character estimate the other providers use only as a
        # last resort -- not used for billed cost estimates, only for the
        # --dry-run preview.
        return max(1, len(text) // 4)
