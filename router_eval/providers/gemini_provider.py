"""
Gemini provider. Requires: pip install google-generativeai
Reads GEMINI_API_KEY from env if no key is passed explicitly.
"""

import os
import time

from .base import GenerationResult, ModelProvider


class GeminiProvider(ModelProvider):
    def __init__(self, model_id: str, api_key: str | None = None):
        super().__init__(model_id, api_key)
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self._genai = genai
        self._model = genai.GenerativeModel(model_id, system_instruction=None)  # system set per-call below

    def generate(self, system_prompt: str, user_input: str, max_tokens: int = 1024) -> GenerationResult:
        # Gemini takes system instructions at model-construction time, not per-call --
        # rebuild the model object per call so a harness run can vary the system
        # prompt across use cases without holding one GeminiProvider per use case.
        model = self._genai.GenerativeModel(self.model_id, system_instruction=system_prompt)

        start = time.perf_counter()
        response = model.generate_content(
            user_input,
            generation_config={"max_output_tokens": max_tokens},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage_metadata
        return GenerationResult(
            text=response.text,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            latency_ms=latency_ms,
            raw_response=None,
        )

    def count_tokens(self, text: str) -> int:
        try:
            return self._model.count_tokens(text).total_tokens
        except Exception:
            return max(1, len(text) // 4)
