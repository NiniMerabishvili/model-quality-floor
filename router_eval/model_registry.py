"""
Model registry. Loads configs/models.yaml and turns each entry into a
ready-to-use ModelProvider instance.

Why YAML instead of a Python dict (as in the original pricing.py): adding a
model should never require touching code. A non-engineer teammate -- or
future-you six months from now -- should be able to add a new model or
update a price by editing one line of a text file, not by finding the right
Python dataclass to edit.
"""

import yaml

from .providers.anthropic_provider import AnthropicProvider
from .providers.gemini_provider import GeminiProvider
from .providers.openai_provider import OpenAIProvider

PROVIDER_CLASSES = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    # Adding a 4th vendor (e.g. Mistral, a self-hosted Llama endpoint) means:
    # 1. write providers/mistral_provider.py implementing ModelProvider
    # 2. add one line here
    # Nothing else in this file, or in harness.py/decision_engine.py, changes.
}


class ModelSpec:
    def __init__(self, key, provider_name, model_id, input_per_mtok, output_per_mtok, notes=""):
        self.key = key
        self.provider_name = provider_name
        self.model_id = model_id
        self.input_per_mtok = input_per_mtok
        self.output_per_mtok = output_per_mtok
        self.notes = notes

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )

    def build_provider(self, api_key: str | None = None):
        cls = PROVIDER_CLASSES.get(self.provider_name)
        if cls is None:
            raise ValueError(
                f"No provider class registered for '{self.provider_name}'. "
                f"Known: {list(PROVIDER_CLASSES)}"
            )
        return cls(self.model_id, api_key=api_key)


class ModelRegistry:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        self._specs = {
            key: ModelSpec(key=key, **fields) for key, fields in raw["models"].items()
        }

    def get(self, key: str) -> ModelSpec:
        if key not in self._specs:
            raise KeyError(f"Unknown model key '{key}'. Known: {list(self._specs)}")
        return self._specs[key]

    def keys(self):
        return list(self._specs)
