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
import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from .use_case_loader import UseCase

logger = logging.getLogger(__name__)

RETRY_INSTRUCTION = "\n\nYour last response was not valid JSON. Return ONLY the JSON object, nothing else."


class JudgeParseError(Exception):
    """Raised when the judge's response could not be parsed as valid JSON after all retries."""


@dataclass
class RubricScore:
    scores: dict[str, int]
    rationale: dict[str, str]

    def mean(self) -> float:
        return statistics.mean(self.scores.values())

    def min_criterion(self) -> tuple[str, float]:
        k = min(self.scores, key=lambda name: self.scores[name])
        return k, self.scores[k]


def build_judge_prompt(use_case: UseCase, model_output: str) -> str:
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


def _strip_code_fence(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def score_response(
    use_case: UseCase,
    model_output: str,
    judge_fn: Callable[[str], str],
    max_retries: int = 2,
) -> RubricScore:
    """
    Calls judge_fn(prompt) and parses the result as JSON.

    If parsing fails (malformed JSON, judge added commentary outside the
    JSON block, etc.), retries up to `max_retries` times (3 attempts total
    by default), appending a stronger instruction to the prompt on each
    retry. The raw failed response is logged every time, not just "parse
    failed", so the actual judge output is visible for debugging.

    If every attempt fails, raises JudgeParseError naming the use case and
    the model output being scored -- this is intentional and NOT swallowed
    into a skipped trial or a fake score, per providers/base.py's "don't
    silently fail" contract.
    """
    base_prompt = build_judge_prompt(use_case, model_output)
    last_error = None
    last_raw = None

    for attempt in range(max_retries + 1):
        prompt = base_prompt if attempt == 0 else base_prompt + RETRY_INSTRUCTION
        raw = judge_fn(prompt)
        last_raw = raw
        cleaned = _strip_code_fence(raw)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                "Judge response was not valid JSON (attempt %d/%d) for use case '%s'. " "Raw response: %r",
                attempt + 1,
                max_retries + 1,
                use_case.key,
                raw,
            )
            continue
        return RubricScore(scores=parsed["scores"], rationale=parsed.get("rationale", {}))

    raise JudgeParseError(
        f"Judge failed to return valid JSON after {max_retries + 1} attempts while scoring "
        f"use case '{use_case.key}'. Model output being scored: {model_output!r}. "
        f"Last JSON error: {last_error}. Last raw judge response: {last_raw!r}"
    )
