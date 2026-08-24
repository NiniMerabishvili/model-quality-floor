"""
Evaluation harness core loop, with adaptive early stopping.

The cost problem this solves: running a fixed n_trials=8 for every model on
every use case spends the same token budget whether the two candidates are
neck-and-neck or one is obviously better after 3 trials. That's wasted spend
on the easy cases, in a tool designed specifically to be affordable to run
without a hosted platform footing the bill.

How it works: trials run in small rounds (default 2 at a time). After
min_trials per model, check whether the leading model's quality mean is
clearly ahead of the runner-up (gap > `stop_threshold` combined stdevs).
If so, stop — more trials would only confirm what's already clear. If the
gap is inside the noise band, run another round, up to max_trials. This is
a simplified sequential test, not a formal SPRT — good enough to cut spend
on clear-cut use cases while still being honest about when more data is
needed (see decision_engine.py's own confidence reporting, which still
applies on top of this).
"""

import logging
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from .judge import RubricScore, score_response
from .model_registry import ModelSpec
from .retry import RateLimitExhaustedError
from .use_case_loader import UseCase

logger = logging.getLogger(__name__)

# model_call_fn signature: (model_key, system_prompt, user_input) ->
#   (output_text, latency_ms, input_tokens, output_tokens)
ModelCallFn = Callable[[str, str, str], tuple[str, float, int, int]]
JudgeFn = Callable[[str], str]

# Defaults for run_use_case_adaptive's adaptive-stopping schedule -- named so
# the CLI/tests can reference the same values run_use_case_adaptive falls
# back to, rather than repeating the bare numbers wherever they're overridden.
DEFAULT_MIN_TRIALS = 3
DEFAULT_MAX_TRIALS = 10
DEFAULT_ROUND_SIZE = 2
DEFAULT_STOP_THRESHOLD = 1.0


@dataclass
class CallResult:
    model_key: str
    input_text: str
    output_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    quality: RubricScore


@dataclass
class ModelAggregate:
    model_key: str
    n_trials: int
    mean_quality: float
    min_criterion_mean: tuple[str, float]
    quality_stdev: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_cost_usd: float
    total_cost_usd: float
    quality_floor_met: bool


def percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct / 100 * (len(s) - 1))))
    return s[idx]


def run_use_case_adaptive(
    use_case: UseCase,
    model_keys: list[str],
    model_call_fn: ModelCallFn,
    judge_fn: JudgeFn,
    min_trials: int = DEFAULT_MIN_TRIALS,
    max_trials: int = DEFAULT_MAX_TRIALS,
    round_size: int = DEFAULT_ROUND_SIZE,
    stop_threshold: float = DEFAULT_STOP_THRESHOLD,
) -> tuple[dict[str, list[CallResult]], dict[str, int], dict[str, int]]:
    """
    model_call_fn: callable(model_key, system_prompt, user_input) ->
        (output_text, latency_ms, input_tokens, output_tokens)
    Cost isn't computed inside this loop (it has no ModelSpec/pricing
    reference) — the early-stop check below runs on quality convergence
    only. Cost is calculated afterward in aggregate(), once per finished
    trial set, via the model_spec passed there.

    Returns: (raw_results: dict[model_key -> list[CallResult]],
              trials_run: dict[model_key -> int],
              skipped: dict[model_key -> int]) -- skipped counts trials
    abandoned because a provider call's rate-limit retries were exhausted
    (see retry.py). Those trials are logged and left out of raw_results
    entirely rather than crashing the run; every other exception (a judge
    JSON-parse failure, an auth error, the spend cap, etc.) still
    propagates unchanged and halts the run, per providers/base.py's
    "don't silently fail" contract -- only rate-limit exhaustion gets this
    skip-and-continue treatment.
    """
    raw_results: dict[str, list[CallResult]] = {mk: [] for mk in model_keys}
    skipped: dict[str, int] = {mk: 0 for mk in model_keys}
    sample_inputs = use_case.sample_inputs

    def run_round(n: int) -> None:
        for model_key in model_keys:
            for _ in range(n):
                for sample_input in sample_inputs:
                    try:
                        output, latency_ms, in_tok, out_tok = model_call_fn(
                            model_key, use_case.system_prompt, sample_input
                        )
                        quality = score_response(use_case, output, judge_fn)
                    except RateLimitExhaustedError as e:
                        skipped[model_key] += 1
                        logger.warning(
                            "Skipping one trial for model '%s' on use case '%s' -- %s",
                            model_key,
                            use_case.key,
                            e,
                        )
                        continue
                    raw_results[model_key].append(
                        CallResult(model_key, sample_input, output, latency_ms, in_tok, out_tok, quality)
                    )

    run_round(min_trials)
    trials_run = min_trials

    while trials_run < max_trials:
        aggs = {mk: aggregate(use_case, mk, r) for mk, r in raw_results.items() if r}
        if len(aggs) < 2:
            break
        sorted_aggs = sorted(aggs.values(), key=lambda a: -a.mean_quality)
        top, second = sorted_aggs[0], sorted_aggs[1]
        gap = top.mean_quality - second.mean_quality
        combined_sd = (top.quality_stdev + second.quality_stdev) / 2 or 0.1
        if gap > stop_threshold * combined_sd:
            break  # clearly ahead -- stop spending on this use case
        run_round(round_size)
        trials_run += round_size

    trials_by_model = {mk: len(r) // max(1, len(sample_inputs)) for mk, r in raw_results.items()}
    return raw_results, trials_by_model, skipped


def aggregate(
    use_case: UseCase,
    model_key: str,
    results: list[CallResult],
    model_spec: ModelSpec | None = None,
) -> ModelAggregate:
    quality_means = [r.quality.mean() for r in results]
    latencies = [r.latency_ms for r in results]
    costs = [model_spec.cost(r.input_tokens, r.output_tokens) for r in results] if model_spec else [0.0] * len(results)

    criteria = results[0].quality.scores.keys()
    per_criterion_means = {c: statistics.mean(r.quality.scores[c] for r in results) for c in criteria}
    weakest = min(per_criterion_means.items(), key=lambda kv: kv[1])

    return ModelAggregate(
        model_key=model_key,
        n_trials=len(results),
        mean_quality=statistics.mean(quality_means),
        min_criterion_mean=weakest,
        quality_stdev=statistics.stdev(quality_means) if len(quality_means) > 1 else 0.0,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        mean_cost_usd=statistics.mean(costs) if costs else 0.0,
        total_cost_usd=sum(costs),
        quality_floor_met=weakest[1] >= use_case.quality_floor,
    )
