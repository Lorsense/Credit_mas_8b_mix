"""Semantic control stays independent of value-entropy inputs and permissions."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALUE = load("semantic_control_value", "verl/utils/value_credit.py")
CONTROL = load("semantic_control_controller", "verl/utils/entropy_control.py")


def batch():
    return {
        "traj_uid": ["t"] * 3, "uid": ["q"] * 3, "value_question": ["question"] * 3,
        "value_action_index": [0, 1, 2], "value_max_solver_turns": [2] * 3,
        "agent_id": ["solver", "verifier", "solver"], "role_turn_index": [0, 0, 1],
        "value_action_text": ["s1", "<verify>reject</verify>", "s2"],
        "is_action_valid": [True] * 3, "pass": [True] * 3,
        "pure_entropy_response_tokens": [10] * 3,
        "rewards": [1., 1., 1.], "advantages": [.2, .3, .4], "pure_coefficient": [.8, 1., 1.2],
    }


def predictions():
    return [{"sem": p, "semantic_available": True, "action_semantic_available": i > 0}
            for i, p in enumerate([.8, .5, .3, .2])]


def attach(b=None, p=None, ready=True):
    b = batch() if b is None else b
    VALUE.attach_value_predictions(b, {"t": predictions() if p is None else p}, ready,
                                   prediction_mode="semantic_only")
    return b


def test_semantic_attachment_uses_initial_and_excludes_terminal_without_entropy():
    b = batch()
    original = copy.deepcopy(b)
    attach(b)
    np.testing.assert_array_equal(b["value_semantic_available"], [True, True, False])
    np.testing.assert_allclose(b["control_delta"], [-.3, -.2, 0])
    assert b["control_value_before"][0] == .8
    assert not b["value_absolute_available"].any()
    assert not b["value_entropy_available"].any()
    assert np.isnan(b["value_full_after"]).all()
    assert np.isnan(b["control_value_after"][-1])
    assert set(b["value_prediction_source"]) == {"semantic_only"}
    for key in original:
        assert b[key] == original[key]


def test_semantic_attachment_ignores_low_coverage_and_invalid_unused_predictions():
    baseline = attach()
    b, p = batch(), predictions()
    b["top16_entropy_mean"] = [.9, .01, .99]
    b["top16_entropy"] = [{"coverage": 0.0}] * 3
    for row in p:
        row.update(abs=float("nan"), full=float("inf"), absolute_available=False, temporal_available=False)
    actual = attach(b, p)
    np.testing.assert_array_equal(actual["value_semantic_available"], baseline["value_semantic_available"])
    np.testing.assert_array_equal(actual["control_delta"], baseline["control_delta"])


def test_semantic_record_build_and_attachment_do_not_parse_unused_entropy():
    b = batch()
    b["top16_entropy_mean"] = ["not a number"] * 3
    b["top16_entropy"] = [{"coverage": "bad unused telemetry"}] * 3
    records, _ = VALUE.build_trajectory_records(b, 2, prediction_mode="semantic_only")
    assert len(records) == 1
    assert records[0]["actions"][0]["entropy_token_count"] == 10
    assert records[0]["actions"][0]["entropy_mean"] is None
    assert records[0]["actions"][0]["entropy_coverage"] is None
    actual = attach(b)
    np.testing.assert_array_equal(actual["value_semantic_available"], [True, True, False])
    legacy, _ = VALUE.build_trajectory_records(b, 2)
    assert not legacy  # preserve the original entropy-aware validation


@pytest.mark.parametrize("failure", ["missing", "nonfinite", "out_of_range", "invalid", "truncated", "tokens", "before_unavailable", "after_unavailable", "not_ready"])
def test_semantic_attachment_fails_closed_without_fabricated_values(failure):
    b, p = batch(), predictions()
    if failure == "missing":
        p[1].pop("sem")
    elif failure == "nonfinite":
        p[1]["sem"] = float("nan")
    elif failure == "out_of_range":
        p[1]["sem"] = 2.
    elif failure == "invalid":
        b["is_action_valid"][0] = False
    elif failure == "truncated":
        b["pure_entropy_truncated"] = [True, False, False]
    elif failure == "tokens":
        b["pure_entropy_response_tokens"][0] = 0
    elif failure == "before_unavailable":
        p[0]["semantic_available"] = False
    elif failure == "after_unavailable":
        p[1]["action_semantic_available"] = False
    actual = attach(b, p, ready=failure != "not_ready")
    assert not actual["value_semantic_available"][0]
    assert np.isnan(actual["control_value_after"][0])
    assert actual["control_delta"][0] == 0


def metadata():
    n = 16
    return {
        "traj_uid": np.asarray([f"t{i}" for i in range(n)], dtype=object),
        "agent_id": np.asarray(["Solver Agent"] * 8 + ["Verifier Agent"] * 8, dtype=object),
        "role_turn_index": np.zeros(n, dtype=np.int64),
        "is_action_valid": np.ones(n, dtype=bool),
        "pure_entropy_truncated": np.zeros(n, dtype=bool),
        "pure_entropy_response_tokens": np.full(n, 10),
        "control_delta": np.full(n, -.2),
        "control_available": np.ones(n, dtype=bool),
        "value_semantic_available": np.ones(n, dtype=bool),
        "value_prediction_source": np.full(n, "semantic_only", dtype=object),
    }


def controller(**kwargs):
    return CONTROL.EntropyController({"enabled": True, "prediction_mode": "semantic_only", **kwargs})


def calibrated(**kwargs):
    obj = controller(**kwargs)
    obj.prepare(metadata(), np.ones(16), step=1, ready=False, reliability={})
    return obj


def test_semantic_gate_has_independent_role_permission_and_first_batch_ramp():
    obj = calibrated()
    fields, metrics = obj.prepare(metadata(), np.full(16, 2.), step=2, ready=True,
                                 reliability={"solver": 1., "verifier": 0.})
    np.testing.assert_allclose(fields["entropy_control_weight"][:8], .25 / 20)
    assert not fields["entropy_control_weight"][8:].any()
    assert metrics["entropy_control/semantic_prediction_coverage"] == 1
    assert metrics["entropy_control/blocked_unqualified_fraction"] == .5
    assert metrics["entropy_control/gate_above_cap_fraction"] == .5
    assert metrics["entropy_control/Solver Agent/ramp"] == .05
    assert metrics["entropy_control/Verifier Agent/ramp"] == 0


def test_entropy_residuals_flags_and_temporal_permission_cannot_change_semantic_gate():
    expected, _ = calibrated().prepare(metadata(), np.full(16, 2.), step=2, ready=True)
    meta = metadata()
    meta.update(value_credit_delta=np.full(16, .99), value_entropy_delta=np.full(16, float("nan")),
                value_absolute_available=np.zeros(16, bool), value_entropy_available=np.zeros(16, bool))
    actual, _ = calibrated().prepare(meta, np.full(16, 2.), step=2, ready=True, temporal_reliability=float("nan"))
    np.testing.assert_array_equal(actual["entropy_control_weight"], expected["entropy_control_weight"])


@pytest.mark.parametrize("failure", ["unready", "missing_prediction", "no_risk", "no_negative_progress", "uncalibrated"])
def test_semantic_controller_reports_why_gate_is_zero(failure):
    obj, meta, entropy = calibrated(), metadata(), np.full(16, 2.)
    if failure == "missing_prediction":
        meta.pop("control_delta")
    elif failure == "no_risk":
        entropy[:] = 1.
    elif failure == "no_negative_progress":
        meta["control_delta"][:] = .1
    elif failure == "uncalibrated":
        meta["role_turn_index"][:] = 1
    fields, metrics = obj.prepare(meta, entropy, step=2, ready=failure != "unready")
    assert not fields["entropy_control_weight"].any()
    key = {"unready": "unqualified", "missing_prediction": "missing_prediction",
           "no_risk": "no_risk", "no_negative_progress": "no_negative_progress",
           "uncalibrated": "uncalibrated"}[failure]
    assert metrics[f"entropy_control/blocked_{key}_fraction"] == 1


def test_semantic_strength_is_linear_and_zero_explicitly_disables_gates():
    results = []
    for strength in (0., .25, 1.):
        fields, _ = calibrated(semantic_strength=strength).prepare(metadata(), np.full(16, 2.), step=2, ready=True)
        results.append(fields["entropy_control_weight"])
    assert not results[0].any()
    np.testing.assert_allclose(results[2], 4 * results[1])


def test_semantic_resume_restores_caps_and_ramp_and_rejects_mode_or_strength_changes():
    original = calibrated()
    original.prepare(metadata(), np.full(16, 2.), step=2, ready=True)
    state = json.loads(json.dumps(original.state_dict()))
    restored = controller()
    restored.load_state_dict(state)
    expected, em = original.prepare(metadata(), np.full(16, 1.8), step=3, ready=True)
    actual, am = restored.prepare(metadata(), np.full(16, 1.8), step=3, ready=True)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert em == am
    with pytest.raises(ValueError, match="prediction_mode"):
        CONTROL.EntropyController({"enabled": True}).load_state_dict(state)
    with pytest.raises(ValueError, match="semantic_strength"):
        controller(semantic_strength=1.).load_state_dict(state)


def test_old_controller_state_defaults_to_entropy_aware_only():
    old = CONTROL.EntropyController({"enabled": True}).state_dict()
    old["config"].pop("prediction_mode")
    old["config"].pop("semantic_strength")
    CONTROL.EntropyController({"enabled": True}).load_state_dict(old)
    with pytest.raises(ValueError, match="prediction_mode"):
        controller().load_state_dict(old)


@pytest.mark.parametrize("strength", [-1., 1.01, float("nan"), float("inf")])
def test_invalid_semantic_strength_rejected(strength):
    with pytest.raises(ValueError, match="semantic_strength"):
        controller(semantic_strength=strength)


def test_controller_rejects_mismatched_prediction_source():
    meta = metadata()
    meta["value_prediction_source"][:] = "entropy_aware"
    with pytest.raises(ValueError, match="prediction source"):
        calibrated().prepare(meta, np.full(16, 2.), step=2, ready=True)
