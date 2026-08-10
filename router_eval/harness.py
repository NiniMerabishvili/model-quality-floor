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

import statistics
from dataclasses import dataclass

from .judge import score_response


@dataclass
class CallResult:
    model_key: str
    input_text: str
    output_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    quality: "RubricScore"


@dataclass
class ModelAggregate:
    model_key: str
    n_trials: int
    mean_quality: float
    min_criterion_mean: tuple
    quality_stdev: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_cost_usd: float
    total_cost_usd: float
    quality_floor_met: bool


def percentile(values, pct):
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct / 100 * (len(s) - 1))))
    return s[idx]


def run_use_case_adaptive(
    use_case, model_keys, model_call_fn, judge_fn,
    min_trials=3, max_trials=10, round_size=2, stop_threshold=1.0,
):
    """
    model_call_fn: callable(model_key, system_prompt, user_input) ->
        (output_text, latency_ms, input_tokens, output_tokens)
    Cost isn't computed inside this loop (it has no ModelSpec/pricing
    reference) — the early-stop check below runs on quality convergence
    only. Cost is calculated afterward in aggregate(), once per finished
    trial set, via the model_spec passed there.

    Returns: (raw_results: dict[model_key -> list[CallResult]], trials_run: dict[model_key -> int])
    """
    raw_results = {mk: [] for mk in model_keys}
    sample_inputs = use_case.sample_inputs

    def run_round(n):
        for model_key in model_keys:
            for _ in range(n):
                for sample_input in sample_inputs:
                    output, latency_ms, in_tok, out_tok = model_call_fn(
                        model_key, use_case.system_prompt, sample_input
                    )
                    quality = score_response(use_case, output, judge_fn)
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

    return raw_results, {mk: len(r) // max(1, len(sample_inputs)) for mk, r in raw_results.items()}


def aggregate(use_case, model_key, results: list, model_spec=None) -> ModelAggregate:
    quality_means = [r.quality.mean() for r in results]
    latencies = [r.latency_ms for r in results]
    costs = (
        [model_spec.cost(r.input_tokens, r.output_tokens) for r in results]
        if model_spec else [0.0] * len(results)
    )

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
