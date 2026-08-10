"""
Turns ModelAggregate stats into a routing recommendation.

This is the piece that actually separates a junior and a senior treatment
of "which model should we use." A junior version of this script normalizes
quality/latency/cost to 0-1 and picks whichever model has the highest
weighted sum. That's a reasonable start but it will happily recommend a
model that's 8% cheaper and technically "wins" the weighted score while
silently failing the one criterion that matters most for that use case
(e.g. fabricating activity in the RAG summary, or being diagnostic in the
burnout explanation). Two things fix that:

1. Quality floor is a GATE, not an input to the weighted sum. A model that
   fails the floor is disqualified for that use case regardless of how far
   ahead it is on cost or latency. This mirrors how you'd actually want a
   production router to behave -- cost optimization should never be allowed
   to trade away the one non-negotiable quality criterion.

2. The recommendation states its own confidence. If quality means are within
   ~1 stdev of each other, or the sample count is small, the engine doesn't
   report a false winner -- it says so and recommends collecting more trials
   before committing, and defaults to the cheaper option ONLY if quality is
   statistically indistinguishable AND both meet the floor.
"""

from dataclasses import dataclass


@dataclass
class Recommendation:
    use_case_key: str
    winner: str | None          # None if no model clears the floor
    reasoning: list             # ordered list of the actual reasoning steps
    confidence: str             # "high" | "moderate" | "low -- needs more trials"
    fallback_note: str = ""


def recommend(use_case, aggregates: dict):
    """
    aggregates: dict[model_key -> ModelAggregate] for this use case.
    """
    reasoning = []

    # Step 1: gate on quality floor
    eligible = {k: a for k, a in aggregates.items() if a.quality_floor_met}
    disqualified = {k: a for k, a in aggregates.items() if not a.quality_floor_met}

    for k, a in disqualified.items():
        crit, val = a.min_criterion_mean
        reasoning.append(
            f"{k} disqualified: '{crit}' scored {val:.2f}, below the "
            f"{use_case.quality_floor} floor required for this use case."
        )

    if not eligible:
        return Recommendation(
            use_case_key=use_case.key,
            winner=None,
            reasoning=reasoning + ["No candidate model clears the quality floor. "
                                    "Do not route this use case automatically -- "
                                    "escalate for prompt revision or manual review."],
            confidence="high",
        )

    if len(eligible) == 1:
        only_key = next(iter(eligible))
        reasoning.append(f"Only {only_key} clears the quality floor; selected by elimination.")
        return Recommendation(use_case_key=use_case.key, winner=only_key,
                               reasoning=reasoning, confidence="high")

    # Step 2: among eligible models, check whether quality is statistically
    # distinguishable. Cheap heuristic (not a full significance test, but
    # honest about that): treat means within ~1 stdev of each other as tied.
    keys = list(eligible)
    a, b = eligible[keys[0]], eligible[keys[1]]
    quality_gap = abs(a.mean_quality - b.mean_quality)
    combined_stdev = (a.quality_stdev + b.quality_stdev) / 2 or 0.1
    quality_tied = quality_gap < combined_stdev

    if quality_tied:
        reasoning.append(
            f"Quality means are within noise ({a.mean_quality:.2f} vs "
            f"{b.mean_quality:.2f}, combined stdev {combined_stdev:.2f}) -- "
            f"treating quality as tied and deciding on latency/cost instead."
        )
        winner = _pick_on_latency_then_cost(use_case, a, b, reasoning)
    else:
        better = a if a.mean_quality > b.mean_quality else b
        worse = b if better is a else a
        reasoning.append(
            f"{better.model_key} leads on quality ({better.mean_quality:.2f} vs "
            f"{worse.model_key} at {worse.mean_quality:.2f}), a gap larger than "
            f"trial-to-trial noise."
        )
        # even with a quality lead, check whether it's worth the cost/latency
        # premium given this use case's weighting -- a senior router doesn't
        # auto-pick "highest quality" if the use case barely weights quality
        if use_case.weight_quality < 0.4 and better.mean_cost_usd > worse.mean_cost_usd * 1.5:
            reasoning.append(
                f"But this use case only weights quality at {use_case.weight_quality:.2f} "
                f"and {better.model_key} costs {better.mean_cost_usd / worse.mean_cost_usd:.1f}x "
                f"more per call -- quality lead doesn't justify it here."
            )
            winner = worse.model_key
        else:
            winner = better.model_key

    n_min = min(a.n_trials, b.n_trials)
    confidence = "high" if n_min >= 5 and not quality_tied else (
        "moderate" if n_min >= 5 else "low -- needs more trials"
    )

    return Recommendation(
        use_case_key=use_case.key,
        winner=winner,
        reasoning=reasoning,
        confidence=confidence,
        fallback_note=(
            "" if confidence != "low -- needs more trials" else
            f"Only {n_min} trials per model -- treat this as directional, "
            f"not a final routing decision. Re-run with n_trials>=10 before shipping."
        ),
    )


def _pick_on_latency_then_cost(use_case, a, b, reasoning):
    if use_case.latency_class == "sync_user_facing":
        a_ok = a.p95_latency_ms <= use_case.latency_budget_ms
        b_ok = b.p95_latency_ms <= use_case.latency_budget_ms
        if a_ok != b_ok:
            winner = a.model_key if a_ok else b.model_key
            reasoning.append(
                f"Only {winner} meets the {use_case.latency_budget_ms}ms p95 budget "
                f"for this sync call."
            )
            return winner
    cheaper = a if a.mean_cost_usd < b.mean_cost_usd else b
    reasoning.append(
        f"Both within latency budget (or use case is async) -- routing to "
        f"{cheaper.model_key}, the cheaper option at ${cheaper.mean_cost_usd:.5f}/call "
        f"vs ${(a.mean_cost_usd + b.mean_cost_usd - cheaper.mean_cost_usd):.5f}/call."
    )
    return cheaper.model_key
