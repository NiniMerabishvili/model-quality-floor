"""
router_eval: a BYOK model-routing evaluation harness.

Runs the same set of use cases against multiple LLM providers side by side,
scores each response with an LLM-as-judge against a per-use-case rubric,
gates candidates on a quality floor before ever considering cost or latency,
and reports a routing recommendation with its own confidence level.

See README.md for the full pitch and how to run it.
"""

__version__ = "0.1.0"
