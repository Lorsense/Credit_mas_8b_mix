"""CPU-only contracts for independent semantic deployment and online learning."""
import copy
import hashlib
import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("semantic_worker_helpers", Path(__file__).with_name("test_credit_value.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
module = helpers.module


def scorer(**kwargs):
    settings = dict(prediction_mode="semantic_only", min_train_trajectories=4,
                    min_train_questions=4, min_val_trajectories=4,
                    min_val_questions=4, min_val_per_class=2,
                    semantic_min_val_prefixes=4, semantic_min_val_questions=4,
                    semantic_min_val_per_class=2, train_batch_size=7)
    return helpers.scorer(**{**settings, **kwargs})


def assert_equal(first, second):
    first = first.state_dict() if hasattr(first, "state_dict") else first
    second = second.state_dict() if hasattr(second, "state_dict") else second
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def trained_semantic(head):
    """Known predictive semantic weights; no expensive pretraining required."""
    with torch.no_grad():
        for parameter in head.semantic.parameters():
            parameter.zero_()
        head.semantic[0].weight.fill_(1)
        head.semantic[1].weight[0, 0] = 1
        head.semantic[-1].weight[0, 0] = 4
        head.semantic[-1].bias.fill_(-2)


def records_with_tokens(uid="online", label=1):
    rec = helpers.record(uid, label)
    for action in rec["actions"]:
        action["entropy_token_count"] = 8
    return rec


def synthetic_rows(s, *, legacy=False, verifier_bad=False, count=32):
    """Balanced distinct trajectories/questions on both sides of the hash split."""
    rows, found = [], {False: 0, True: 0}
    index = 0
    while min(found.values()) < count // 2:
        question = f"semantic-question-{index}"
        index += 1
        side = s._is_validation(question)
        if found[side] >= count // 2:
            continue
        label = found[side] % 2
        found[side] += 1
        row = helpers.encoded(s, records_with_tokens(str(index), label))
        row["question_key"] = question
        row["features"].zero_()
        for prefix, role in enumerate(row["prefix_roles"]):
            sign = 2 * label - 1
            row["features"][prefix, 0] = -sign if verifier_bad and role == "verifier" else sign
        # No usable entropy. Validity, truncation and token counts remain raw
        # action metadata and are not entropy qualification requirements.
        row["absolute"][:, [0, 1, 5, 6, 7, 11]] = 0
        row["temporal"].zero_()
        row["abs_mask"].zero_()
        row["temp_mask"].zero_()
        if legacy:
            row.pop("semantic_available", None)
            row.pop("semantic_metadata_schema", None)
        rows.append(row)
    return rows


def split(s, rows):
    return ([row for row in rows if not s._is_validation(row["question_key"])],
            [row for row in rows if s._is_validation(row["question_key"])])


def old_failed_checkpoint(tmp_path, *, verifier_bad=False):
    source = helpers.scorer(allow_absolute_only_pretrain=True, holdout_salt="source-split", validation_fraction=.35)
    rows = synthetic_rows(source, legacy=True, verifier_bad=verifier_bad)
    source._fit_scaler(rows)
    source._train(rows[:4], 1, "semantic")
    trained_semantic(source.candidate_head)
    source._replay = OrderedDict((row["traj_uid"], row) for row in rows)
    source.step, source.bad_windows = 27, 2
    path = tmp_path / "failed_offline.pt"
    source.save(path)
    # Simulate an actual pre-feature checkpoint, not a newly serialized mode.
    state = torch.load(path, weights_only=False)
    state.pop("prediction_mode", None)
    state["config"].pop("prediction_mode", None)
    state.pop("semantic_reliability", None)
    state.pop("replay_feature_storage", None)
    torch.save(state, path)
    return source, path


def fail_if_called(*args, **kwargs):
    raise AssertionError("semantic path called an entropy or historical encoding dependency")


def test_old_failed_candidate_can_qualify_semantic_without_importing_history(tmp_path, monkeypatch):
    source, path = old_failed_checkpoint(tmp_path)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    fresh = scorer(seed=13, learning_rate=.0003)
    initial_rng = fresh._rng.get_state().clone()
    unused_heads = [copy.deepcopy(head) for head in (fresh.candidate_head.absolute,
                   fresh.candidate_head.temporal, fresh.control_head, fresh.temporal_control_head)]
    for name in ("_encode", "_fit_scaler", "_train", "_quality_passes"):
        monkeypatch.setattr(fresh, name, fail_if_called)
    result = fresh.load(path, warm_start=True)
    assert result["ready"] and fresh.ready and fresh.version == 1
    assert fresh.semantic_reliability == {"solver": 1., "verifier": 1.}
    assert fresh.reliability == fresh.temporal_reliability == {"solver": 0., "verifier": 0.}
    assert not fresh.pretrained  # Does not claim the old entropy gate passed.
    assert_equal(source.candidate_head.semantic, fresh.candidate_head.semantic)
    assert_equal(source.candidate_head.semantic, fresh.deployed_head.semantic)
    assert any(not torch.equal(value, fresh.deployed_head.semantic.state_dict()[key])
               for key, value in source.deployed_head.semantic.state_dict().items())
    for before, after in zip(unused_heads, (fresh.candidate_head.absolute, fresh.candidate_head.temporal,
                                           fresh.control_head, fresh.temporal_control_head)):
        assert_equal(before, after)
    assert not fresh._replay and not fresh._pending and fresh.step == 0
    assert not fresh.optimizer.state
    assert torch.equal(initial_rng, fresh._rng.get_state())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


def test_same_failed_checkpoint_preserves_original_entropy_aware_behavior(tmp_path):
    _, path = old_failed_checkpoint(tmp_path)
    fresh = helpers.scorer(holdout_salt="source-split", validation_fraction=.35)
    loaded = fresh.load(path, warm_start=True)
    assert not loaded["ready"] and fresh.version == 0
    with pytest.raises(ValueError, match="entropy-qualified"):
        fresh.load(path)


def test_solver_permission_is_not_blocked_by_bad_verifier(tmp_path):
    _, path = old_failed_checkpoint(tmp_path, verifier_bad=True)
    fresh = scorer()
    fresh.load(path, warm_start=True)
    assert fresh.ready and fresh.version == 1
    assert fresh.semantic_reliability == {"solver": 1., "verifier": 0.}
    assert fresh.temporal_reliability == {"solver": 0., "verifier": 0.}


def test_unknown_or_missing_legacy_metadata_cannot_silently_qualify(tmp_path):
    _, path = old_failed_checkpoint(tmp_path)
    state = torch.load(path, weights_only=False)
    for _, row in state["replay"]:
        row.pop("absolute")
    torch.save(state, path)
    fresh = scorer()
    result = fresh.load(path, warm_start=True)
    assert not result["ready"] and not fresh.ready
    assert fresh.semantic_reliability == {"solver": 0., "verifier": 0.}
    assert fresh.startup_semantic_report
    assert not fresh._replay


def test_semantic_prediction_does_not_execute_entropy_branches_or_scaler(monkeypatch):
    s = scorer()
    row = helpers.encoded(s, records_with_tokens())
    assert s.scaler is None
    monkeypatch.setattr(s.candidate_head.absolute, "forward", fail_if_called)
    monkeypatch.setattr(s.candidate_head.temporal, "forward", fail_if_called)
    monkeypatch.setattr(s, "_batch", fail_if_called)
    expected = s._predict_semantic(s.candidate_head, row)
    actual = s._predict_semantic(s.candidate_head, {"features": row["features"]})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad


def test_prepare_missing_top16_retains_valid_semantic_before_after(monkeypatch):
    s = scorer()
    s.ready, s.version = True, 1
    s.semantic_reliability = {"solver": 1., "verifier": 1.}
    rec = records_with_tokens()
    for action in rec["actions"]:
        action.pop("entropy_mean")
        action["entropy_coverage"] = 0
    monkeypatch.setattr(s.deployed_head.absolute, "forward", fail_if_called)
    monkeypatch.setattr(s.deployed_head.temporal, "forward", fail_if_called)
    served = s.prepare([rec])["values"][rec["traj_uid"]]
    assert served[0]["semantic_available"]  # Initial may be an action's before.
    assert not served[0]["action_semantic_available"]
    assert all(row["action_semantic_available"] for row in served[1:-1])
    assert not served[-1]["action_semantic_available"]
    assert all(not row.get("action_absolute_available", False) for row in served)


@pytest.mark.parametrize("change", [{"valid": False}, {"truncated": True}, {"entropy_token_count": 0}])
def test_invalid_truncated_or_empty_actions_are_excluded(change):
    s = scorer()
    s.ready = True
    s.semantic_reliability = {"solver": 1., "verifier": 1.}
    rec = records_with_tokens()
    rec["actions"][0].update(change)
    served = s.prepare([rec])["values"][rec["traj_uid"]]
    assert not served[1]["action_semantic_available"]


def test_online_updates_only_candidate_semantic_with_no_entropy_dependency(monkeypatch):
    s = scorer(semantic_min_val_prefixes=10000)
    rows = synthetic_rows(s)
    train, val = split(s, rows)
    monkeypatch.setattr(s, "_ingest", lambda: (train, val))
    monkeypatch.setattr(s, "_fit_scaler", fail_if_called)
    monkeypatch.setattr(s, "_quality_passes", fail_if_called)
    for head in (s.candidate_head, s.control_head, s.temporal_control_head, s.deployed_head):
        monkeypatch.setattr(head.absolute, "forward", fail_if_called)
        monkeypatch.setattr(head.temporal, "forward", fail_if_called)
    before = {name: copy.deepcopy(getattr(s, name).state_dict()) for name in
              ("encoder", "candidate_head", "deployed_head", "control_head", "temporal_control_head")}
    metrics = s.update()
    assert metrics["semantic_train_loss"] > 0
    assert metrics["semantic_train_prefixes"] > 0
    assert not s.ready
    after = s.candidate_head.state_dict()
    assert any(not torch.equal(value, after[key]) for key, value in before["candidate_head"].items()
               if key.startswith("semantic."))
    for key, value in before["candidate_head"].items():
        if not key.startswith("semantic."):
            torch.testing.assert_close(value, after[key], rtol=0, atol=0)
    for name in ("encoder", "deployed_head", "control_head", "temporal_control_head"):
        assert_equal(before[name], getattr(s, name))
    trainable = {id(p) for p in s.candidate_head.semantic.parameters()}
    assert {id(p) for group in s.optimizer.param_groups for p in group["params"]} == trainable
    assert all(not p.requires_grad and p.grad is None for p in s.candidate_head.absolute.parameters())
    assert all(not p.requires_grad and p.grad is None for p in s.candidate_head.temporal.parameters())


def test_semantic_qualification_counts_distinct_trajectories_not_prefixes():
    s = scorer(semantic_min_val_per_class=8)
    trained_semantic(s.candidate_head)
    train, val = split(s, synthetic_rows(s))
    # Many correct prefixes but only one true trajectory per outcome.
    tiny = [next(row for row in val if row["label"] == label) for label in (0, 1)]
    duplicated = [copy.deepcopy(row) for row in tiny for _ in range(20)]
    metrics = {}
    s._validate_semantic_deploy(train, duplicated, metrics)
    assert not s.ready and not any(s.semantic_reliability.values())


def test_semantic_insufficient_data_retains_a_working_deployment():
    s = scorer()
    s.ready, s.version = True, 3
    s.semantic_reliability = {"solver": 1., "verifier": 0.}
    before = copy.deepcopy(s.deployed_head)
    metrics = s.update()
    assert metrics["ready"] == 1 and metrics["version"] == 3
    assert metrics["status"] == "insufficient_data_retained_deployment"
    assert s.semantic_reliability == {"solver": 1., "verifier": 0.}
    assert_equal(before, s.deployed_head)


def test_semantic_exact_resume_restores_pending_rng_and_only_semantic_optimizer(tmp_path, monkeypatch):
    original = scorer(dropout=.3, semantic_min_val_prefixes=10000)
    rows = synthetic_rows(original)
    train, val = split(original, rows)
    monkeypatch.setattr(original, "_ingest", lambda: (train, val))
    original.update()
    original._pending = OrderedDict((row["traj_uid"], row) for row in rows[:2])
    original._replay = OrderedDict((row["traj_uid"], row) for row in rows[2:])
    original.role_bad_windows = {"solver": 2, "verifier": 1}
    path = tmp_path / "semantic_online.pt"
    original.save(path)
    resumed = scorer(dropout=.3, semantic_min_val_prefixes=10000)
    loaded = resumed.load(path, resume=True)
    assert not loaded["ready"]  # Temporarily unqualified runs can resume exactly.
    assert resumed.step == original.step
    assert resumed.role_bad_windows == original.role_bad_windows
    assert list(resumed._pending) == list(original._pending)
    assert list(resumed._replay) == list(original._replay)
    assert resumed.semantic_reliability == original.semantic_reliability
    assert torch.equal(resumed._rng.get_state(), original._rng.get_state())
    monkeypatch.setattr(resumed, "_ingest", lambda: (train, val))
    first, second = original.update(), resumed.update()
    assert first["semantic_train_loss"] == second["semantic_train_loss"]
    for name in ("candidate_head", "deployed_head", "control_head", "temporal_control_head"):
        assert_equal(getattr(original, name), getattr(resumed, name))
    assert torch.equal(resumed._rng.get_state(), original._rng.get_state())


def test_cross_mode_exact_resume_and_wrong_encoder_are_rejected(tmp_path):
    s = scorer()
    path = tmp_path / "semantic.pt"
    s.save(path)
    with pytest.raises(ValueError, match="mode|prediction"):
        helpers.scorer().load(path, resume=True)
    with pytest.raises(ValueError, match="encoder identity"):
        scorer(model_path="different-fixed-8b").load(path, resume=True)
    source, old_path = old_failed_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="mode|prediction"):
        scorer().load(old_path, resume=True)
    assert s.feature_dim == s.encoder.config.hidden_size + 8


def test_semantic_qualified_initialization_rejects_entropy_qualification(tmp_path):
    source, path = old_failed_checkpoint(tmp_path)
    state = torch.load(path, weights_only=False)
    state["ready"] = state["pretrained"] = True
    torch.save(state, path)
    with pytest.raises(ValueError, match="mode|semantic|prediction"):
        scorer().load(path)


def test_insufficient_active_solver_validation_prevents_shared_head_replacement():
    s = scorer()
    trained_semantic(s.candidate_head)
    s.deployed_head.semantic.load_state_dict(s.candidate_head.semantic.state_dict())
    s.ready, s.version = True, 1
    s.semantic_reliability = {"solver": 1., "verifier": 0.}
    train, val = split(s, synthetic_rows(s))
    for row in val:
        for i, role in enumerate(row["prefix_roles"]):
            if role == "solver":
                row["semantic_available"][i] = False
    before = copy.deepcopy(s.deployed_head)
    metrics = {}
    s._validate_semantic_deploy(train, val, metrics)
    assert metrics["candidate_verifier_sem_auc"] == 1
    assert metrics["status"] == "insufficient_data_retained_deployment"
    assert s.ready and s.version == 1
    assert s.semantic_reliability == {"solver": 1., "verifier": 0.}
    assert_equal(before, s.deployed_head)


def test_bad_windows_are_role_specific_distinct_and_require_enough_data():
    s = scorer(miscalibration_patience=3)
    trained_semantic(s.candidate_head)
    s.deployed_head.semantic.load_state_dict(s.candidate_head.semantic.state_dict())
    s.ready, s.version = True, 1
    s.semantic_reliability = {"solver": 1., "verifier": 1.}
    train, val = split(s, synthetic_rows(s))
    for row in val:
        for i, role in enumerate(row["prefix_roles"]):
            if role == "solver":
                row["features"][i, 0] *= -1
    for window in range(3):
        current = copy.deepcopy(val)
        for row in current:
            row["traj_uid"] += f"-window-{window}"
        metrics = {}
        s._validate_semantic_deploy(train, current, metrics)
        counters = s.role_bad_windows.copy()
        if window < 2:
            assert counters == {"solver": window + 1, "verifier": 0}
            assert s.semantic_reliability == {"solver": 1., "verifier": 1.}
            s._validate_semantic_deploy(train, current, {})
            assert s.role_bad_windows == counters
            s._validate_semantic_deploy(train, current[:1], {})
            assert s.role_bad_windows == counters
        else:
            assert metrics["disabled"] == 1
    assert s.ready
    assert s.semantic_reliability == {"solver": 0., "verifier": 1.}


def test_ready_semantic_qualified_import_and_exact_resume(tmp_path):
    _, old_path = old_failed_checkpoint(tmp_path)
    source = scorer()
    source.load(old_path, warm_start=True)
    assert source.ready
    source.prepare([records_with_tokens("online")])
    source.step = 6
    source.role_bad_windows = {"solver": 1, "verifier": 0}
    path = tmp_path / "ready_semantic.pt"
    source.save(path)
    fresh = scorer()
    result = fresh.load(path)
    assert result["ready"] and fresh.semantic_reliability == source.semantic_reliability
    assert fresh.version == source.version
    assert not fresh._pending and not fresh._replay and fresh.step == 0
    assert_equal(source.deployed_head.semantic, fresh.candidate_head.semantic)
    assert_equal(source.deployed_head.semantic, fresh.deployed_head.semantic)
    resumed = scorer()
    result = resumed.load(path, resume=True)
    assert result["ready"]
    assert resumed.step == 6 and list(resumed._pending) == ["online"]
    assert resumed.role_bad_windows == source.role_bad_windows
    assert resumed.semantic_reliability == source.semantic_reliability
    assert_equal(source.deployed_head, resumed.deployed_head)
    assert resumed.startup_semantic_report == source.startup_semantic_report


def test_scaled_legacy_cache_never_uses_raw_metadata_positions(tmp_path):
    _, path = old_failed_checkpoint(tmp_path)
    state = torch.load(path, weights_only=False)
    state["replay_feature_storage"] = "scaled"
    torch.save(state, path)
    s = scorer()
    result = s.load(path, warm_start=True)
    assert not result["ready"] and s.version == 0
    assert s.startup_semantic_report["cache_metadata_unrecoverable_prefixes"] > 0
    assert s.semantic_reliability == {"solver": 0., "verifier": 0.}
