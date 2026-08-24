"""
CLI entry point.

Usage:
  python -m router_eval.cli run \
      --use-cases configs/use_cases.example.yaml \
      --models claude-sonnet-5,gemini-flash,gpt-4o-mini \
      --max-spend 2.00

  python -m router_eval.cli run --use-cases ... --models ... --dry-run

Design choices:

- --models takes ANY comma-separated list of keys from models.yaml, not a
  hardcoded Gemini-vs-Claude pair. Three, four, five models can be compared
  in one run -- the decision engine currently reasons pairwise (see its
  docstring) so with 3+ models it evaluates the top two by quality first, a
  known simplification documented there rather than hidden.

- --dry-run estimates cost from each provider's own tokenizer (via
  count_tokens) BEFORE any real call is made, so a person can see "this
  will cost about $X" and back out before spending anything.

- --max-spend is a hard stop enforced mid-run, not just at the start.
  A dry-run estimate can be wrong if a model responds much longer than
  expected -- the running total is checked after every judged trial.
"""

import argparse
import os
from collections.abc import Callable

from dotenv import load_dotenv

from .decision_engine import Recommendation, recommend
from .harness import ModelAggregate, aggregate, run_use_case_adaptive
from .model_registry import ModelRegistry
from .providers.base import ModelProvider
from .retry import call_with_rate_limit_backoff
from .use_case_loader import UseCase, load_use_cases

# Rough placeholder used only for the --dry-run cost preview, before any
# real call has happened -- refine per use case if you have historical
# output lengths to plug in instead.
DRY_RUN_OUTPUT_TOKEN_ESTIMATE = 200
DRY_RUN_TRIALS_ESTIMATE = 5


def build_judge_fn(judge_provider: ModelProvider, judge_key: str = "judge") -> Callable[[str], str]:
    def _judge(prompt: str) -> str:
        def _do_call() -> str:
            result = judge_provider.generate(
                system_prompt="You are a strict, consistent rubric scorer.",
                user_input=prompt,
            )
            return result.text

        return call_with_rate_limit_backoff(_do_call, context=f"judge={judge_key}")

    return _judge


def estimate_dry_run_cost(
    use_case: UseCase,
    model_keys: list[str],
    registry: ModelRegistry,
    providers: dict[str, ModelProvider],
    n_trials_estimate: int = DRY_RUN_TRIALS_ESTIMATE,
) -> float:
    total = 0.0
    for mk in model_keys:
        spec = registry.get(mk)
        provider = providers[mk]
        for sample in use_case.sample_inputs:
            in_tok = provider.count_tokens(use_case.system_prompt + sample)
            total += spec.cost(in_tok, DRY_RUN_OUTPUT_TOKEN_ESTIMATE) * n_trials_estimate
    return total


def main() -> None:
    # Loads ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY / etc. from a
    # .env file in the current working directory (if present) into the
    # process environment, so providers' os.environ.get(...) lookups see
    # them. A no-op if .env doesn't exist or a variable is already set in
    # the shell -- real shell env vars still take precedence over .env by
    # default. Called here rather than at module import time, so importing
    # this module (e.g. from tests) never has the side effect of mutating
    # the process environment.
    load_dotenv()

    parser = argparse.ArgumentParser(description="Model routing evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--use-cases", required=True, help="Path to use_cases YAML")
    run_p.add_argument("--models-config", default="configs/models.yaml")
    run_p.add_argument("--models", required=True, help="Comma-separated model keys, e.g. claude-sonnet-5,gemini-flash")
    run_p.add_argument(
        "--judge-model",
        default=None,
        help="Model key for the judge. Defaults to a model NOT in --models, to avoid self-preference bias.",
    )
    run_p.add_argument("--max-spend", type=float, default=2.00)
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--out", default="results/report.md")

    args = parser.parse_args()

    registry = ModelRegistry(args.models_config)
    use_cases = load_use_cases(args.use_cases)
    model_keys = [m.strip() for m in args.models.split(",")]

    for mk in model_keys:
        registry.get(mk)  # raises a clear KeyError early if a typo'd model key was passed

    judge_key = args.judge_model or next((k for k in registry.keys() if k not in model_keys), model_keys[0])
    if judge_key in model_keys:
        print(
            f"Warning: judge model '{judge_key}' is also a candidate being scored -- "
            f"self-preference bias is possible. Pass --judge-model to use a separate model."
        )

    providers: dict[str, ModelProvider] = {
        mk: registry.get(mk).build_provider() for mk in set(model_keys + [judge_key])
    }
    judge_fn = build_judge_fn(providers[judge_key], judge_key)

    if args.dry_run:
        total = 0.0
        for uc in use_cases.values():
            total += estimate_dry_run_cost(uc, model_keys, registry, providers)
        print(f"Estimated cost for this run: ${total:.4f} (rough — actual output length varies)")
        return

    spent = 0.0

    def call_fn(model_key: str, system_prompt: str, user_input: str) -> tuple[str, float, int, int]:
        def _do_call() -> tuple[str, float, int, int]:
            nonlocal spent
            result = providers[model_key].generate(system_prompt, user_input)
            spent += registry.get(model_key).cost(result.input_tokens, result.output_tokens)
            if spent > args.max_spend:
                raise RuntimeError(
                    f"Spend cap exceeded: ${spent:.4f} > ${args.max_spend:.2f}. "
                    f"Run halted mid-evaluation -- partial results are not written."
                )
            return result.text, result.latency_ms, result.input_tokens, result.output_tokens

        return call_with_rate_limit_backoff(_do_call, context=f"model={model_key}")

    all_recs: dict[str, Recommendation] = {}
    all_aggs: dict[str, dict[str, ModelAggregate]] = {}
    all_skipped: dict[str, dict[str, int]] = {}
    for uc_key, uc in use_cases.items():
        raw, trials, skipped = run_use_case_adaptive(uc, model_keys, call_fn, judge_fn)
        aggs = {mk: aggregate(uc, mk, r, model_spec=registry.get(mk)) for mk, r in raw.items() if r}
        rec = recommend(uc, aggs)
        all_recs[uc_key] = rec
        all_aggs[uc_key] = aggs
        all_skipped[uc_key] = skipped
        skip_note = f" (skipped due to rate limits: {skipped})" if any(skipped.values()) else ""
        print(f"{uc_key}: {trials} trials/model, recommendation = {rec.winner} ({rec.confidence}){skip_note}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(render_report(all_recs, all_aggs, use_cases, spent, all_skipped))
    print(f"\nTotal spend: ${spent:.4f}")
    print(f"Report written to {args.out}")


def render_report(
    all_recs: dict[str, Recommendation],
    all_aggs: dict[str, dict[str, ModelAggregate]],
    use_cases: dict[str, UseCase],
    total_spend: float,
    all_skipped: dict[str, dict[str, int]] | None = None,
) -> str:
    all_skipped = all_skipped or {}
    lines = ["# Model Router — Evaluation Report", "", f"Total spend this run: ${total_spend:.4f}", ""]
    for uc_key, rec in all_recs.items():
        uc = use_cases[uc_key]
        lines += [
            f"## {uc_key}",
            f"_{uc.description.strip()}_",
            "",
            f"**Recommendation: `{rec.winner or 'NO MODEL CLEARS FLOOR'}`** (confidence: {rec.confidence})",
            "",
        ]
        lines.append("| model | mean quality | weakest criterion | p50 lat (ms) | p95 lat (ms) | mean cost/call |")
        lines.append("|---|---|---|---|---|---|")
        for mk, agg in all_aggs[uc_key].items():
            crit, val = agg.min_criterion_mean
            flag = "OK" if agg.quality_floor_met else "BELOW FLOOR"
            lines.append(
                f"| {mk} | {agg.mean_quality:.2f} | {crit}={val:.2f} ({flag}) | "
                f"{agg.p50_latency_ms:.0f} | {agg.p95_latency_ms:.0f} | ${agg.mean_cost_usd:.5f} |"
            )
        lines.append("")
        lines.append("**Reasoning:**")
        for step in rec.reasoning:
            lines.append(f"- {step}")
        if rec.fallback_note:
            lines.append(f"- ⚠️ {rec.fallback_note}")
        skipped = {mk: n for mk, n in all_skipped.get(uc_key, {}).items() if n}
        if skipped:
            skip_desc = ", ".join(f"{mk}: {n}" for mk, n in skipped.items())
            lines.append(
                f"- ⚠️ Some trials were skipped after exhausting rate-limit retries ({skip_desc}) -- "
                f"the numbers above reflect only the trials that succeeded."
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
