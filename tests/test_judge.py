"""
Tests for judge.py's score_response(), including the JSON-parse retry path
(Fix 1). judge_fn is faked here as a plain callable, matching judge.py's own
contract that it never knows or cares which real SDK the judge model is.
"""

import json

import pytest

from router_eval.judge import JudgeParseError, RubricScore, score_response


def test_valid_json_response_parses_into_rubric_score(make_use_case):
    uc = make_use_case(rubric={"accuracy": "5 = perfect; 1 = wrong"})
    payload = json.dumps({"scores": {"accuracy": 5}, "rationale": {"accuracy": "Spot on."}})

    score = score_response(uc, "some model output", judge_fn=lambda prompt: payload)

    assert isinstance(score, RubricScore)
    assert score.scores == {"accuracy": 5}
    assert score.rationale == {"accuracy": "Spot on."}


def test_json_wrapped_in_code_fences_is_stripped(make_use_case):
    uc = make_use_case()
    payload = "```json\n" + json.dumps({"scores": {"accuracy": 4}, "rationale": {}}) + "\n```"

    score = score_response(uc, "some model output", judge_fn=lambda prompt: payload)

    assert score.scores == {"accuracy": 4}


def test_malformed_json_retries_then_recovers(make_use_case):
    uc = make_use_case()
    calls = []

    def flaky_judge(prompt):
        calls.append(prompt)
        if len(calls) < 3:
            return "this is not json at all"
        return json.dumps({"scores": {"accuracy": 3}, "rationale": {}})

    score = score_response(uc, "some model output", judge_fn=flaky_judge)

    assert score.scores == {"accuracy": 3}
    assert len(calls) == 3  # 2 failures + 1 success, within the default max_retries=2
    # retry prompts should carry the stronger instruction; the first attempt shouldn't
    assert "not valid JSON" not in calls[0]
    assert "not valid JSON" in calls[1]
    assert "not valid JSON" in calls[2]


def test_exhausting_all_retries_raises_judge_parse_error(make_use_case):
    uc = make_use_case(key="my_use_case")
    calls = []

    def always_broken_judge(prompt):
        calls.append(prompt)
        return "still not json, sorry"

    with pytest.raises(JudgeParseError) as excinfo:
        score_response(uc, "the model output being scored", judge_fn=always_broken_judge)

    assert len(calls) == 3  # 1 initial + 2 retries, the default max_retries=2
    assert "my_use_case" in str(excinfo.value)
    assert "the model output being scored" in str(excinfo.value)
