"""
Tests for decision_engine.py's recommend().

These use FakeUseCase and ModelAggregate objects built directly (see
conftest.py) rather than running the harness against a real use_cases.yaml
or real API calls -- fast, isolated, deterministic.
"""

from router_eval.decision_engine import recommend


def test_disqualified_model_loses_even_if_cheaper_and_faster(make_use_case, make_aggregate):
    uc = make_use_case(quality_floor=4.0)
    cheap_but_bad = make_aggregate(
        "cheap_bad_model",
        mean_quality=4.5,
        min_criterion_mean=("accuracy", 3.0),
        quality_floor_met=False,
        mean_cost_usd=0.00001,
        p95_latency_ms=200,
    )
    good_but_pricier = make_aggregate(
        "good_model",
        mean_quality=4.2,
        min_criterion_mean=("accuracy", 4.2),
        quality_floor_met=True,
        mean_cost_usd=0.0005,
        p95_latency_ms=900,
    )
    rec = recommend(uc, {"cheap_bad_model": cheap_but_bad, "good_model": good_but_pricier})

    assert rec.winner == "good_model"
    assert any("cheap_bad_model disqualified" in step for step in rec.reasoning)


def test_all_models_fail_floor_returns_none_and_escalation_message(make_use_case, make_aggregate):
    uc = make_use_case(quality_floor=4.5)
    a = make_aggregate("model_a", mean_quality=4.0, min_criterion_mean=("accuracy", 3.5), quality_floor_met=False)
    b = make_aggregate("model_b", mean_quality=3.9, min_criterion_mean=("accuracy", 3.2), quality_floor_met=False)

    rec = recommend(uc, {"model_a": a, "model_b": b})

    assert rec.winner is None
    assert rec.confidence == "high"
    assert any("manual review" in step.lower() for step in rec.reasoning)


def test_tied_quality_routes_to_cheaper_option_using_real_report_numbers(make_use_case, make_aggregate):
    # Mirrors the real task_triage run in results/report.md: gemini-flash-lite
    # at 4.73 vs groq-gpt-oss-120b at 4.69, combined stdev 0.38, both clearing
    # the 4.0 floor -- decision_engine should call quality tied and route on
    # cost (both p95 latencies exceed the 800ms sync budget, so latency
    # doesn't decide it either -- it falls all the way through to cost).
    uc = make_use_case(quality_floor=4.0, latency_class="sync_user_facing", latency_budget_ms=800)
    gemini = make_aggregate(
        "gemini-flash-lite",
        mean_quality=4.73,
        min_criterion_mean=("classification_accuracy", 4.18),
        quality_floor_met=True,
        quality_stdev=0.38,
        n_trials=9,
        p50_latency_ms=642,
        p95_latency_ms=879,
        mean_cost_usd=0.00006,
    )
    groq = make_aggregate(
        "groq-gpt-oss-120b",
        mean_quality=4.69,
        min_criterion_mean=("classification_accuracy", 4.18),
        quality_floor_met=True,
        quality_stdev=0.38,
        n_trials=9,
        p50_latency_ms=779,
        p95_latency_ms=1026,
        mean_cost_usd=0.00016,
    )

    rec = recommend(uc, {"gemini-flash-lite": gemini, "groq-gpt-oss-120b": groq})

    assert rec.winner == "gemini-flash-lite"
    assert any("within noise" in step.lower() for step in rec.reasoning)
    assert rec.confidence == "moderate"


def test_real_quality_lead_picks_higher_quality_model(make_use_case, make_aggregate):
    uc = make_use_case(quality_floor=4.0, weight_quality=0.6)
    better = make_aggregate(
        "model_a",
        mean_quality=4.9,
        min_criterion_mean=("accuracy", 4.9),
        quality_floor_met=True,
        quality_stdev=0.05,
        mean_cost_usd=0.0002,
    )
    worse = make_aggregate(
        "model_b",
        mean_quality=4.0,
        min_criterion_mean=("accuracy", 4.0),
        quality_floor_met=True,
        quality_stdev=0.05,
        mean_cost_usd=0.0001,
    )

    rec = recommend(uc, {"model_a": better, "model_b": worse})

    assert rec.winner == "model_a"


def test_real_quality_lead_but_low_weight_and_big_cost_premium_picks_cheaper(make_use_case, make_aggregate):
    uc = make_use_case(quality_floor=4.0, weight_quality=0.2)
    better_but_pricier = make_aggregate(
        "model_a",
        mean_quality=4.9,
        min_criterion_mean=("accuracy", 4.9),
        quality_floor_met=True,
        quality_stdev=0.05,
        mean_cost_usd=0.0010,
    )
    cheaper = make_aggregate(
        "model_b",
        mean_quality=4.0,
        min_criterion_mean=("accuracy", 4.0),
        quality_floor_met=True,
        quality_stdev=0.05,
        mean_cost_usd=0.0002,
    )

    rec = recommend(uc, {"model_a": better_but_pricier, "model_b": cheaper})

    assert rec.winner == "model_b"


def test_low_confidence_when_n_trials_small(make_use_case, make_aggregate):
    uc = make_use_case(quality_floor=4.0)
    a = make_aggregate(
        "model_a",
        mean_quality=4.5,
        min_criterion_mean=("accuracy", 4.5),
        quality_floor_met=True,
        quality_stdev=0.1,
        n_trials=3,
        mean_cost_usd=0.0001,
    )
    b = make_aggregate(
        "model_b",
        mean_quality=4.45,
        min_criterion_mean=("accuracy", 4.45),
        quality_floor_met=True,
        quality_stdev=0.1,
        n_trials=3,
        mean_cost_usd=0.0002,
    )

    rec = recommend(uc, {"model_a": a, "model_b": b})

    assert rec.confidence == "low -- needs more trials"
    assert "3 trials" in rec.fallback_note
