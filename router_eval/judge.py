"""
Rubric-based LLM-as-judge scoring. Unchanged from the original design —
this module was already provider-agnostic, since it takes `judge_fn` as a
plain callable(prompt: str) -> str rather than importing any SDK directly.
cli.py is what wires judge_fn to a real ModelProvider.generate() call; this
file has no idea whether the judge is Claude, Gemini, or GPT.

Two documented bias mitigations (see cli.py for how they're enforced):
1. Position bias — each candidate is scored independently against the
   rubric, never "which is better", and trial order is randomized.
2. Self-preference bias — the judge model should not be the same model as
   either candidate being compared in that trial.
"""

import json
import statistics
from dataclasses import dataclass


@dataclass
class RubricScore:
    scores: dict
    rationale: dict

    def mean(self) -> float:
        return statistics.mean(self.scores.values())

    def min_criterion(self) -> tuple[str, float]:
        k = min(self.scores, key=self.scores.get)
        return k, self.scores[k]


def build_judge_prompt(use_case, model_output: str) -> str:
    rubric_lines = "\n".join(f"- {name}: {desc}" for name, desc in use_case.rubric.items())
    return f"""You are scoring a single AI response against a rubric. Score each
criterion independently on a 1-5 scale. Do not let a low score on one
criterion pull down another.

Task context: {use_case.description}

Rubric:
{rubric_lines}

Response to score:
---
{model_output}
---

Return ONLY valid JSON: {{"scores": {{"<criterion>": <1-5 int>, ...}},
"rationale": {{"<criterion>": "<one short sentence>", ...}}}}
"""


def score_response(use_case, model_output: str, judge_fn) -> RubricScore:
    prompt = build_judge_prompt(use_case, model_output)
    raw = judge_fn(prompt)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(cleaned)
    return RubricScore(scores=parsed["scores"], rationale=parsed.get("rationale", {}))
