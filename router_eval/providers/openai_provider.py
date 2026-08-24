"""
OpenAI provider (GPT models). Requires: pip install openai

This file is here specifically to prove the point: the harness was originally
written for two vendors, but adding a third took one file, ~30 lines, and zero
changes anywhere else in the codebase. That's the generalization payoff.

The only OpenAICompatibleProvider subclass that overrides count_tokens() --
OpenAI is the one vendor here with its own local tokenizer (tiktoken)
available offline, so its --dry-run estimate doesn't need the shared
len(text)//4 fallback the other three providers rely on.
"""

from .openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    base_url = "https://api.openai.com/v1"
    env_var = "OPENAI_API_KEY"

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model_id)
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)
