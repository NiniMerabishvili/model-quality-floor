"""
Shared fixtures for the decision_engine.py / judge.py test suite.

FakeUseCase is a minimal stand-in for use_case_loader.UseCase, carrying only
the fields decision_engine.py and judge.py actually read. Tests build one
directly instead of loading configs/use_cases.example.yaml, so they stay
fast and isolated from any real YAML file on disk.
"""

import pytest

from router_eval.harness import ModelAggregate


class FakeUseCase:
    def __init__(
        self,
        key="test_use_case",
        description="A minimal test use case.",
        rubric=None,
        quality_floor=4.0,
        weight_quality=0.35,
        weight_latency=0.40,
        weight_cost=0.25,
        latency_class="sync_user_facing",
        latency_budget_ms=800,
    ):
        self.key = key
        self.description = description
        self.rubric = rubric if rubric is not None else {"accuracy": "5 = perfect; 1 = wrong"}
        self.quality_floor = quality_floor
        self.weight_quality = weight_quality
        self.weight_latency = weight_latency
        self.weight_cost = weight_cost
        self.latency_class = latency_class
        self.latency_budget_ms = latency_budget_ms


@pytest.fixture
def make_use_case():
    """Factory fixture: make_use_case(**overrides) -> FakeUseCase."""
    return FakeUseCase


@pytest.fixture
def make_aggregate():
    """Factory fixture for building ModelAggregate instances without running the harness."""

    def _make(
        model_key,
        mean_quality,
        min_criterion_mean,
        quality_floor_met,
        quality_stdev=0.0,
        n_trials=9,
        p50_latency_ms=500.0,
        p95_latency_ms=700.0,
        mean_cost_usd=0.0001,
        total_cost_usd=None,
    ):
        return ModelAggregate(
            model_key=model_key,
            n_trials=n_trials,
            mean_quality=mean_quality,
            min_criterion_mean=min_criterion_mean,
            quality_stdev=quality_stdev,
            p50_latency_ms=p50_latency_ms,
            p95_latency_ms=p95_latency_ms,
            mean_cost_usd=mean_cost_usd,
            total_cost_usd=total_cost_usd if total_cost_usd is not None else mean_cost_usd * n_trials,
            quality_floor_met=quality_floor_met,
        )

    return _make
