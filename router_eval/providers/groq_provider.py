"""
Groq provider. Requires: pip install openai (Groq's API is OpenAI-compatible
-- same request/response shape, just a different base_url and model names,
so this reuses the openai package instead of the groq-sdk package).

Not to be confused with Grok (xAI's model, providers/grok_provider.py) --
Groq is the LPU hardware inference company hosting Llama/GPT-OSS/etc.
directly on its own chips.

Reads GROQ_API_KEY from env if no key is passed explicitly.
"""

from .openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    base_url = "https://api.groq.com/openai/v1"
    env_var = "GROQ_API_KEY"
