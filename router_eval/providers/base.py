"""
The provider interface.

Everything else in this codebase (harness, judge, decision engine) talks to
models only through this interface — never through a provider's raw SDK
directly. That's the entire trick behind "add any model, not just Gemini/
Claude": as long as a new provider implements `generate()` with this exact
signature, the harness, the judge, and the CLI all work with it automatically,
with zero changes anywhere else.

Think of it like a power outlet standard. The harness doesn't care if the
electricity comes from solar, coal, or a generator (Anthropic, Google,
OpenAI, a local Llama server) — it just needs a plug that fits the socket.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw_response: dict | None = None  # kept for debugging, not used by the harness


class ModelProvider(ABC):
    """
    One instance per (provider, model_id) pair, e.g. Anthropic + claude-sonnet-5.
    Registered in model_registry.py, looked up by a short key like
    "claude-sonnet-5" so the rest of the codebase never touches SDK-specific
    types.
    """

    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self.api_key = api_key  # None is valid -- means "read from env", see subclasses

    @abstractmethod
    def generate(self, system_prompt: str, user_input: str, max_tokens: int = 1024) -> GenerationResult:
        """
        Must:
        - Actually call the provider's API (no mocking inside a real provider class)
        - Time the call itself (wall-clock around the request, not the SDK's
          own reported latency if it has one -- we want what the harness sees)
        - Return exact token counts from the API response, not estimates.
          If a provider's SDK doesn't return usage, count tokens with that
          provider's own tokenizer -- never approximate with len(text)//4,
          it's off by enough to distort the cost comparison.
        Must NOT:
        - Swallow errors silently. Let exceptions propagate -- the harness
          decides whether a failed call means "retry", "skip", or "abort run",
          not the provider.
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Used for the CLI's --dry-run cost estimate, before any real call is made."""
        raise NotImplementedError
