"""Failed offline candidate initialization and paired online qualification."""
import copy
import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("warm_start_helpers", Path(__file__).with_name("test_credit_value.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
module = helpers.module


def failed_checkpoint(tmp_path):
    source = helpers.scorer(allow_absolute_only_pretrain=True)
    rows = [helpers.encoded(source, helpers.record(str(i), i % 2)) for i in range(4)]
    source._fit_scaler(rows)
    source._train(rows, 2, "semantic")
    source._train(rows, 2, "absolute")
    source._train(rows, 1, "semantic", True)
    source._pending = OrderedDict((row["traj_uid"], row) for row in rows)
    source._replay = copy.deepcopy(source._pending)
    source.step = 19
    source.bad_windows = 2
    source.last_validation_fingerprint = "offline-window"
    path = tmp_path / "rejected.pt"
    source.save(path)
    assert any(not torch.equal(value, source.deployed_head.state_dict()[key])
               for key, value in source.candidate_head.state_dict().items())
    return source, path


def assert_model_equal(first, second):
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def paired_scorer(**kwargs):
    s = helpers.scorer(min_train_trajectories=2, min_val_trajectories=2,
                       min_train_questions=2, min_val_questions=2, min_val_per_class=1,
                       min_entropy_val_prefixes=2, **kwargs)
    s.training_protocol = module._PAIRED_TRAINING_PROTOCOL
    s.deployment_mode = "absolute_bootstrap"
    return s


def null_entropy_rows(s):
    rows = []
    for i in range(8):
        row = helpers.encoded(s, helpers.record(str(i), i % 2))
        row["question_key"] = f"independent-{i}"
        row["absolute"][:, [0, 6]] = 0
        row["temporal"][:, [0, 1, 2, 3, 6, 7, 8, 9]] = 0
        rows.append(row)
    s._fit_scaler(rows)
    return rows


def test_warm_start_uses_candidate_and_starts_new_unqualified_runtime(tmp_path):
    source, path = failed_checkpoint(tmp_path)
    fresh = helpers.scorer(learning_rate=.0003, replay_max_trajectories=16, seed=13)
    initial_rng = fresh._rng.get_state().clone()
    initial_encoder = copy.deepcopy(fresh.encoder)
    result = fresh.load(path, warm_start=True)

    assert result["warm_started"] and not result["ready"]
    assert fresh.warm_started and not fresh.pretrained
    assert fresh.version == fresh.step == fresh.bad_windows == 0
    assert fresh.last_validation_fingerprint is None
    assert not fresh._pending and not fresh._replay
    assert fresh.reliability == fresh.temporal_reliability == {"solver": 0.0, "verifier": 0.0}
    assert fresh.training_protocol == module._PAIRED_TRAINING_PROTOCOL
    assert fresh.deployment_mode == "absolute_bootstrap"
    assert fresh.scaler == source.scaler
    for head in (fresh.candidate_head, fresh.control_head, fresh.temporal_control_head):
        assert_model_equal(head, source.candidate_head)
    assert_model_equal(fresh.encoder, initial_encoder)
    assert all(not p.requires_grad for p in fresh.encoder.parameters())
    assert torch.equal(fresh._rng.get_state(), initial_rng)
    for optimizer in (fresh.optimizer, fresh.control_optimizer, fresh.temporal_control_optimizer):
        assert not optimizer.state
        assert optimizer.param_groups[0]["lr"] == .0003

    prepared = fresh.prepare([helpers.record("online", 0)])
    assert prepared["warm_started"] and prepared["metrics"]["warm_started"] == 1
    assert not prepared["ready"] and prepared["values"] == {}
    assert list(fresh._pending) == ["online"]


def test_failed_checkpoint_still_requires_explicit_warm_start(tmp_path):
    _, path = failed_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="entropy-qualified"):
        helpers.scorer().load(path)
    with pytest.raises(ValueError, match="cannot be combined"):
        helpers.scorer().load(path, resume=True, warm_start=True)


@pytest.mark.parametrize("setting,value,match", [
    ("model_path", "different-encoder", "encoder identity"),
    ("scaler_clip", 4., "fixed entropy scaling"),
    ("entropy_hidden_dim", 16, "entropy_hidden_dim"),
])
def test_warm_start_preserves_encoder_shape_and_scaler_checks(tmp_path, setting, value, match):
    _, path = failed_checkpoint(tmp_path)
    with pytest.raises(ValueError, match=match):
        helpers.scorer(**{setting: value}).load(path, warm_start=True)


def test_paired_ablation_masks_entropy_but_preserves_auxiliary_metadata():
    s = paired_scorer()
    row = helpers.encoded(s)
    row["absolute"][:, 2] = torch.arange(len(row["features"])) + 1
    row["temporal"][:, 4] = 2
    s._fit_scaler([row])
    expected = s._predict(s.control_head, row, no_entropy=True)
    changed = copy.deepcopy(row)
    changed["absolute"][:, [0, 6]] += .5
    changed["temporal"][:, [0, 1, 2, 3, 6, 7, 8, 9]] += .7
    torch.testing.assert_close(expected, s._predict(s.control_head, changed, no_entropy=True), rtol=0, atol=0)
    changed["absolute"][:, 2] += 10
    assert not torch.equal(expected, s._predict(s.control_head, changed, no_entropy=True))
    temporal = s._predict(s.temporal_control_head, row, no_temporal=True)
    changed = copy.deepcopy(row)
    changed["temporal"][:, [0, 1, 2, 3, 6, 7, 8, 9]] -= 1
    torch.testing.assert_close(temporal, s._predict(s.temporal_control_head, changed, no_temporal=True), rtol=0, atol=0)
    changed["absolute"][:, 0] += .5
    assert not torch.equal(temporal, s._predict(s.temporal_control_head, changed, no_temporal=True))


def test_paired_null_inputs_give_identical_controls_even_with_dropout():
    s = paired_scorer(dropout=.35)
    rows = null_entropy_rows(s)
    global_rng = torch.random.get_rng_state().clone()
    for _ in range(2):
        metrics = s._train_paired_online(rows, 2)
        assert_model_equal(s.candidate_head, s.control_head)
        assert_model_equal(s.candidate_head, s.temporal_control_head)
        assert metrics["train_loss"] == metrics["control_train_loss"]
        assert metrics["train_loss"] == metrics["temporal_control_train_loss"]
    assert torch.equal(global_rng, torch.random.get_rng_state())


def test_online_candidate_learns_while_null_entropy_cannot_open_gate(monkeypatch):
    s = paired_scorer(dropout=.2)
    rows = null_entropy_rows(s)
    before = copy.deepcopy(s.candidate_head)
    encoder = copy.deepcopy(s.encoder)
    monkeypatch.setattr(s, "_ingest", lambda: (rows, rows))
    metrics = s.update()
    assert metrics["status"] == "candidate_rejected"
    assert metrics["candidate_matched_control_gain"] == 0
    assert not s.ready and s.version == 0
    assert s.reliability == {"solver": 0., "verifier": 0.}
    assert any(not torch.equal(value, s.candidate_head.state_dict()[key])
               for key, value in before.state_dict().items())
    assert_model_equal(encoder, s.encoder)
    assert_model_equal(s.candidate_head.semantic, s.control_head.semantic)
    assert_model_equal(s.candidate_head.semantic, s.temporal_control_head.semantic)


def test_online_entropy_signal_can_qualify_and_deploy_after_warm_start(tmp_path):
    _, path = failed_checkpoint(tmp_path)
    s = paired_scorer(learning_rate=.01, train_epochs=60)
    s.load(path, warm_start=True)
    for i in range(128):
        row = helpers.encoded(s, helpers.record(str(i), i % 2))
        row["features"].zero_()
        row["absolute"][:, [0, 6]] = .1 + .8 * (i % 2)
        row["question_key"] = f"independent-{i}"
        s._pending[str(i)] = row
    assert not s.ready
    metrics = s.update()
    assert metrics["candidate_absolute_gain"] > 0
    assert metrics["candidate_matched_control_gain"] > 0
    assert metrics["ready"] == 1 and s.ready and s.version == 1
    assert any(s.reliability.values())
    assert_model_equal(s.candidate_head, s.deployed_head)


def test_nonterminal_own_role_scope_controls_training_and_served_values():
    s = paired_scorer()
    record = helpers.record()
    record["actions"][3]["valid"] = False
    row = helpers.encoded(s, record)
    s._fit_scaler([row])
    scoped = s._control_training_data(row)
    assert scoped["abs_mask"].tolist() == [False, True, True, True, False, False]
    assert scoped["temp_mask"].tolist() == [False, False, False, True, False, False]
    s.ready = True
    s.temporal_reliability = {"solver": 1., "verifier": 1.}
    served = s.prepare([record])["values"][record["traj_uid"]]
    for index in (0, 4, 5):
        assert served[index]["sem"] == served[index]["abs"] == served[index]["full"]
        assert not served[index]["action_absolute_available"]
        assert not served[index]["action_temporal_available"]


def test_terminal_only_rows_cannot_train_entropy_residuals():
    s = paired_scorer()
    row = helpers.encoded(s)
    s._fit_scaler([row])
    for key in ("features", "absolute", "temporal", "abs_mask", "temp_mask", "prefix_terminal"):
        row[key] = row[key][-1:]
    row["prefix_roles"] = ["solver"]
    before = copy.deepcopy(s.candidate_head)
    assert s._train([row], 2, "entropy") == 0
    assert_model_equal(before, s.candidate_head)


def test_warm_started_checkpoint_resumes_paired_training_exactly(tmp_path):
    _, path = failed_checkpoint(tmp_path)
    original = helpers.scorer(dropout=.3, learning_rate=.0004)
    original.load(path, warm_start=True)
    rows = [helpers.encoded(original, helpers.record(str(i), i % 2)) for i in range(4)]
    original._train_paired_online(rows, 2)
    original._pending = OrderedDict((row["traj_uid"], row) for row in rows[:1])
    original._replay = OrderedDict((row["traj_uid"], row) for row in rows[1:])
    original.step = 3
    checkpoint = tmp_path / "online.pt"
    original.save(checkpoint)
    resumed = helpers.scorer(dropout=.3, learning_rate=.0004)
    loaded = resumed.load(checkpoint, resume=True)
    assert loaded["warm_started"] and not loaded["ready"]
    assert resumed.training_protocol == original.training_protocol
    assert resumed.step == original.step == 3
    assert list(resumed._pending) == list(original._pending)
    assert list(resumed._replay) == list(original._replay)
    first = original._train_paired_online(rows, 2)
    second = resumed._train_paired_online(rows, 2)
    assert first == second
    for name in ("candidate_head", "control_head", "temporal_control_head", "deployed_head"):
        assert_model_equal(getattr(original, name), getattr(resumed, name))
    assert torch.equal(original._rng.get_state(), resumed._rng.get_state())


@pytest.mark.parametrize("field,value", [("training_protocol", "unknown"), ("warm_started", "yes")])
def test_checkpoint_rejects_invalid_warm_start_protocol_metadata(tmp_path, field, value):
    _, path = failed_checkpoint(tmp_path)
    state = torch.load(path, weights_only=False)
    state[field] = value
    torch.save(state, path)
    with pytest.raises(ValueError, match="protocol|warm_started"):
        helpers.scorer().load(path, warm_start=True)
