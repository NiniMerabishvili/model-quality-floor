"""
Loads use case definitions from a YAML file into the same UseCase shape the
harness and decision engine already expect. This is the second half of
"generalize away from Freeside": anyone can point --use-cases at their own
YAML file describing their own product's call sites, with no code changes.
"""

from dataclasses import dataclass
import yaml


@dataclass(frozen=True)
class UseCase:
    key: str
    description: str
    system_prompt: str
    sample_inputs: list
    rubric: dict
    latency_class: str
    latency_budget_ms: int
    quality_floor: float
    weight_quality: float
    weight_latency: float
    weight_cost: float


def load_use_cases(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    use_cases = {}
    for key, fields in raw["use_cases"].items():
        use_cases[key] = UseCase(key=key, **fields)
    return use_cases
