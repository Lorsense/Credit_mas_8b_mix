"""Exercise actual trainer adapters/checkpoint methods without Ray or GPUs."""
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


VALUE = module("mix_integration_value", "verl/utils/value_credit.py")
CONTROL = module("mix_integration_control", "verl/utils/entropy_control.py")
PURE = module("mix_integration_pure", "tests/trainer/ppo/test_sparse_entropy_credit_integration.py")


class Config(dict):
    __getattr__ = dict.__getitem__


def methods():
    source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    selected = [n for n in source.body if isinstance(n, ast.FunctionDef)
                and n.name in {"value_scorer_metrics", "validate_value_initialization"}]
    trainer = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    names = {"_prepare_value_credit", "_prepare_entropy_control", "_save_checkpoint", "_load_checkpoint"}
    selected += [n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name in names]
    tree = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected, type_ignores=[]))
    namespace = {"np": np, "torch": torch, "os": os, "json": json,
                 "ray": SimpleNamespace(get=lambda value: value),
                 "find_latest_ckpt_path": lambda folder: None,
                 "build_trajectory_records": VALUE.build_trajectory_records,
                 "attach_value_predictions": VALUE.attach_value_predictions}
    exec(compile(tree, "actual_mix_trainer_methods", "exec"), namespace)
    return namespace


METHODS = methods()


def config(path="unused"):
    return Config(agent=Config(orchestra=Config(math=Config(max_loop_num=3))),
                  algorithm=Config(entropy_credit=Config(value=Config(allow_missing_resume=False))),
                  trainer=Config(default_local_dir=str(path), default_hdfs_dir=None,
                                 resume_mode="resume_path", resume_from_path=str(Path(path) / "global_step_2"),
                                 del_local_ckpt_after_load=False))


def prepared_batch():
    batch, _ = PURE.prepared(PURE.fixture(include_verifier=True))
    batch, _ = PURE.finalized(batch)
    meta = batch.non_tensor_batch
    n = len(batch)
    is_verifier = meta["agent_id"] == "Verifier Agent"
    meta["value_action_index"] = 2 * meta["role_turn_index"] + is_verifier.astype(int)
    meta["value_action_text"] = np.array([
        ("<verify>reject</verify>" if turn == 0 else "<verify>approve</verify>") if verifier else "solution"
        for turn, verifier in zip(meta["role_turn_index"], is_verifier)], dtype=object)
    meta["value_question"] = np.array(["exact question"] * n, dtype=object)
    meta["value_max_solver_turns"] = np.full(n, 3)
    meta["action_full_entropy"] = np.ones(n)
    return batch


def prediction(records):
    values = {}
    for record in records:
        values[record["traj_uid"]] = [
            {"sem": .8 - .1 * i, "abs": .8 - .1 * i, "full": .8 - .15 * i,
             "action_absolute_available": i > 0, "action_temporal_available": i > 2}
            for i in range(len(record["actions"]) + 1)]
    return dict(ready=True, values=values, reliability={"solver": 1, "verifier": 1},
                temporal_reliability={"solver": 1, "verifier": 0}, metrics={"version": 4})


def test_actual_trainer_adapters_preserve_pure_factors_and_attach_detached_action_controls():
    batch = prepared_batch()
    pure_factor = batch.non_tensor_batch["entropy_credit_final_multiplier"].copy()
    advantages = batch.batch["advantages"].clone()
    controller = CONTROL.EntropyController(dict(enabled=True, min_calibration_actions=2, ramp_steps=1))
    trainer = SimpleNamespace(config=config(), global_steps=1, entropy_controller=controller,
                              value_scorer=SimpleNamespace(prepare=SimpleNamespace(remote=prediction)))
    METHODS["_prepare_value_credit"](trainer, batch)
    METHODS["_prepare_entropy_control"](trainer, batch)
    assert not batch.batch["entropy_control_weight"].any()
    trainer.global_steps = 2
    batch.non_tensor_batch["action_full_entropy"][:] = 4.0
    METHODS["_prepare_value_credit"](trainer, batch)
    METHODS["_prepare_entropy_control"](trainer, batch)
    assert batch.batch["entropy_control_weight"].max() > 0
    first_solver = (batch.non_tensor_batch["agent_id"] == "Solver Agent") & (batch.non_tensor_batch["role_turn_index"] == 0)
    assert batch.batch["entropy_control_weight"][first_solver].min() > 0  # absolute-only base brake
    for key in ("entropy_control_weight", "entropy_control_cap", "entropy_control_valid"):
        assert batch.batch[key].shape == (len(batch),)
        assert not batch.batch[key].requires_grad
    np.testing.assert_array_equal(batch.non_tensor_batch["entropy_credit_final_multiplier"], pure_factor)
    torch.testing.assert_close(batch.batch["advantages"], advantages, rtol=0, atol=0)


def test_actual_checkpoint_saves_and_requires_value_and_controller_state(tmp_path):
    events = []
    def actor_save(path, *args, **kwargs):
        Path(path).mkdir(parents=True)
    def value_save(path):
        Path(path).write_text("test-value", encoding="utf-8")
    loader = SimpleNamespace(state_dict=lambda: {"position": 2}, load_state_dict=lambda state: events.append(("data", state)))
    controller = CONTROL.EntropyController(dict(enabled=True, min_calibration_actions=2))
    trainer = SimpleNamespace(config=config(tmp_path), global_steps=2, use_critic=False,
        wg_to_agents_mapping={"solver": [], "verifier": []}, train_dataloader=loader,
        entropy_controller=controller,
        actor_rollout_wg={role: SimpleNamespace(save_checkpoint=actor_save,
            load_checkpoint=lambda path, **kwargs: events.append(("actor", path))) for role in ("solver", "verifier")},
        value_scorer=SimpleNamespace(save=SimpleNamespace(remote=value_save),
            load=SimpleNamespace(remote=lambda path, **kwargs: events.append(("value", kwargs)))))
    batch = prepared_batch()
    controller.prepare(batch.non_tensor_batch, np.ones(len(batch)), 2, False)
    METHODS["_save_checkpoint"](trainer)
    controller_state = controller.state_dict()
    trainer.entropy_controller = CONTROL.EntropyController(dict(enabled=True, min_calibration_actions=2))
    METHODS["_load_checkpoint"](trainer)
    assert trainer.entropy_controller.state_dict() == controller_state
    assert events[0] == ("value", {"resume": True})
    assert sum(event[0] == "actor" for event in events) == 2
    assert events[-1] == ("data", {"position": 2})
    (tmp_path / "global_step_2" / "entropy_controller.json").unlink()
    with pytest.raises(FileNotFoundError, match="entropy calibration"):
        METHODS["_load_checkpoint"](trainer)


def test_main_loop_scores_before_policy_and_trains_value_after_entire_actor_loop():
    source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    trainer = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    fit = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
    calls = [(ast.unparse(n.func), n.lineno) for n in ast.walk(fit) if isinstance(n, ast.Call)]
    def one(name):
        found = [line for callee, line in calls if callee == name]
        assert len(found) == 1
        return found[0]
    score = one("self._prepare_value_credit")
    control = one("self._prepare_entropy_control")
    update = one("self.value_scorer.update.remote")
    actor = one("self.actor_rollout_wg[wg_id].update_actor")
    assert score < control < actor < update
    actor_loops = [n for n in ast.walk(fit) if isinstance(n, ast.For) and n.lineno < actor <= n.end_lineno]
    innermost = min(actor_loops, key=lambda n: n.end_lineno - n.lineno)
    assert update > innermost.end_lineno


@pytest.mark.parametrize("step,raises", [(0, True), (50, False)])
def test_startup_requires_ready_initialization_but_resumes_temporarily_disabled_value(step, raises):
    source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    trainer_cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    fit = next(n for n in trainer_cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
    block = next(n for n in fit.body if isinstance(n, ast.If) and ast.unparse(n.test) == "self.value_scorer is not None")
    trainer = SimpleNamespace(global_steps=step, config=config(),
                              value_scorer=SimpleNamespace(prepare=SimpleNamespace(remote=lambda rows: {"ready": False})))
    tree = ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[]))
    namespace = {"self": trainer, "ray": SimpleNamespace(get=lambda value: value),
                 "validate_value_initialization": METHODS["validate_value_initialization"],
                 "value_scorer_metrics": METHODS["value_scorer_metrics"],
                 "logger": SimpleNamespace(log=lambda **kwargs: None)}
    if raises:
        with pytest.raises(ValueError, match="offline pretraining"):
            exec(compile(tree, "actual_startup_gate", "exec"), namespace)
    else:
        exec(compile(tree, "actual_startup_gate", "exec"), namespace)


@pytest.mark.parametrize("warm_started", [False, True])
def test_candidate_startup_requires_loaded_weights_without_deployment(warm_started):
    state = {"ready": False, "warm_started": warm_started}
    value = {"initialization_mode": "candidate", "require_pretrained": True}
    if warm_started:
        METHODS["validate_value_initialization"](value, state)
        assert not state["ready"]
    else:
        with pytest.raises(ValueError, match="successfully loaded candidate"):
            METHODS["validate_value_initialization"](value, state)


def test_qualified_startup_does_not_accept_candidate_provenance_as_qualification():
    with pytest.raises(ValueError, match="offline pretraining"):
        METHODS["validate_value_initialization"](
            {"initialization_mode": "qualified"}, {"ready": False, "warm_started": True})


def test_unknown_value_initialization_mode_fails():
    with pytest.raises(ValueError, match="initialization_mode"):
        METHODS["validate_value_initialization"]({"initialization_mode": "bypass"}, {"ready": True})


def test_semantic_startup_requires_matching_mode_and_accepts_independent_qualification():
    cfg = {"initialization_mode": "candidate", "prediction_mode": "semantic_only"}
    with pytest.raises(ValueError, match="prediction_mode"):
        METHODS["validate_value_initialization"](cfg, {"ready": True, "warm_started": True})
    for ready in (False, True):
        METHODS["validate_value_initialization"](
            cfg, {"prediction_mode": "semantic_only", "ready": ready, "warm_started": True})


def test_evaluation_does_not_construct_a_value_worker():
    source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    trainer_cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    method = next(n for n in trainer_cls.body if isinstance(n, ast.FunctionDef) and n.name == "_init_value_scorer")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_eval_value_skip", "exec"), namespace)
    trainer = SimpleNamespace(value_credit_enabled=True, value_scorer=None,
                              config=SimpleNamespace(trainer={"val_only": True}))
    # No Ray or model imports are available here; evaluation must return first.
    namespace[method.name](trainer)
    assert trainer.value_scorer is None


def test_semantic_trainer_routes_role_permissions_without_entropy_reliability():
    batch = prepared_batch()
    pure_factor = batch.non_tensor_batch["entropy_credit_final_multiplier"].copy()
    advantages = batch.batch["advantages"].clone()
    cfg = config()
    cfg.algorithm.entropy_credit.value["prediction_mode"] = "semantic_only"
    def semantic_prediction(records):
        scored = prediction(records)
        scored["reliability"] = {"solver": 0, "verifier": 0}
        scored["semantic_reliability"] = {"solver": 1, "verifier": 0}
        for values in scored["values"].values():
            for i, item in enumerate(values):
                item.update(sem=.8 - .2 * i, semantic_available=True,
                            action_semantic_available=i > 0 and i < len(values) - 1,
                            action_absolute_available=False, action_temporal_available=False)
                item.pop("abs")
                item.pop("full")
        return scored
    controller = CONTROL.EntropyController(dict(enabled=True, prediction_mode="semantic_only",
                                                semantic_strength=.25, min_calibration_actions=2, ramp_steps=1))
    trainer = SimpleNamespace(config=cfg, global_steps=1, entropy_controller=controller,
                              value_scorer=SimpleNamespace(prepare=SimpleNamespace(remote=semantic_prediction)))
    METHODS["_prepare_value_credit"](trainer, batch)
    assert trainer.value_scorer_reliability == {"solver": 1, "verifier": 0}
    METHODS["_prepare_entropy_control"](trainer, batch)
    trainer.global_steps = 2
    batch.non_tensor_batch["action_full_entropy"][:] = 4.0
    METHODS["_prepare_value_credit"](trainer, batch)
    METHODS["_prepare_entropy_control"](trainer, batch)
    solver = batch.non_tensor_batch["agent_id"] == "Solver Agent"
    assert batch.batch["entropy_control_weight"][solver].max() > 0
    assert not batch.batch["entropy_control_weight"][~solver].any()
    assert batch.meta_info["entropy_control_prediction_mode"] == "semantic_only"
    assert batch.meta_info["entropy_control_semantic_strength"] == .25
    np.testing.assert_array_equal(batch.non_tensor_batch["entropy_credit_final_multiplier"], pure_factor)
    torch.testing.assert_close(batch.batch["advantages"], advantages, rtol=0, atol=0)
