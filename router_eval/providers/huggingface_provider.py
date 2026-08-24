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

from .openai_compatible import OpenAICompatibleProvider


class HuggingFaceProvider(OpenAICompatibleProvider):
    base_url = "https://router.huggingface.co/v1"
    env_var = "HF_TOKEN"
