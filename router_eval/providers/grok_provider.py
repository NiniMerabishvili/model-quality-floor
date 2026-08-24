"""
xAI Grok provider. Requires: pip install openai (xAI's Chat Completions API
is OpenAI-compatible -- same request/response shape, just a different
base_url and model names, so this reuses the openai package instead of a
separate xAI SDK).
Reads GROK_API_KEY from env if no key is passed explicitly.
"""

from .openai_compatible import OpenAICompatibleProvider


class GrokProvider(OpenAICompatibleProvider):
    base_url = "https://api.x.ai/v1"
    env_var = "GROK_API_KEY"
