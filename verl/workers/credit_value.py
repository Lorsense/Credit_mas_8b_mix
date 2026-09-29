"""Frozen causal prefix encoder and independently trained terminal-value head.

This module deliberately does not import Ray or ``verl``. ``prepare`` scores with
an already deployed head; ``update`` learns from the prepared batch only after
its policy update. Checkpoints are trusted local torch files, not model exports.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import os
from collections import OrderedDict
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# Keep this worker usable for offline pretraining without importing Ray,
# DataProto, TensorDict or the rest of the training runtime.
_metadata_spec = importlib.util.spec_from_file_location("_prefix_entropy_metadata", Path(__file__).resolve().parents[1] / "utils/value_credit.py")
_metadata = importlib.util.module_from_spec(_metadata_spec)
_metadata_spec.loader.exec_module(_metadata)
FEATURE_SCHEMA = _metadata.FEATURE_SCHEMA
ABS_FEATURE_NAMES = _metadata.ABS_FEATURE_NAMES
TEMP_FEATURE_NAMES = _metadata.TEMP_FEATURE_NAMES
prefix_entropy_features = _metadata.prefix_entropy_features


_DEFAULTS = {
    "prediction_mode": "entropy_aware",
    "semantic_min_val_prefixes": 32,
    "semantic_min_val_questions": 8,
    "semantic_min_val_per_class": 8,
    "semantic_min_val_auc": 0.55,
    "semantic_min_brier_improvement": 0.0,
    "device": "cpu",
    "torch_dtype": "float32",
    "attn_implementation": "sdpa",
    "trust_remote_code": False,
    "max_length": 16384,
    "head_hidden_dim": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "train_epochs": 5,
    "train_batch_size": 256,
    "replay_max_trajectories": 2048,
    "validation_fraction": 0.2,
    "holdout_salt": "prefix-value-v1",
    "seed": 0,
    "min_train_trajectories": 64,
    "min_val_trajectories": 32,
    "min_val_per_class": 8,
    "min_train_questions": 8,
    "min_val_questions": 8,
    "min_val_auc": 0.55,
    "min_brier_improvement": 0.0,
    "candidate_brier_tolerance": 0.0,
    "disable_miscalibrated": True,
    "disable_ready_when_insufficient": False,
    "cpu_num_threads": 1,
    "entropy_hidden_dim": 64,
    "dropout": 0.1,
    "min_entropy_brier_gain": 0.0,
    "min_temporal_brier_gain": 0.0,
    "min_entropy_val_prefixes": 32,
    "miscalibration_patience": 3,
    "min_entropy_coverage": 0.9,
    "scaler_clip": 5.0,
    "scaler_floor": 0.001,
    "allow_absolute_only_pretrain": False,
}
_STRUCTURAL_DIM = 8
_CHECKPOINT_VERSION = 2
_PAIRED_TRAINING_PROTOCOL = "paired_nonterminal_v1"
_SEMANTIC_TRAINING_PROTOCOL = "semantic_nonterminal_v1"
_SEMANTIC_METADATA_SCHEMA = "complete-action-metadata-v1"


class EntropyValueHead(nn.Module):
    """Three supervised residual logits with explicit feature masking controls."""
    def __init__(self, feature_dim, hidden_dim, entropy_hidden_dim, dropout):
        super().__init__()
        self.semantic = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
                                      nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        def residual(dim):
            return nn.Sequential(nn.LayerNorm(feature_dim + dim),
                                 nn.Linear(feature_dim + dim, entropy_hidden_dim), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(entropy_hidden_dim, 1))
        self.absolute = residual(len(ABS_FEATURE_NAMES))
        self.temporal = residual(len(ABS_FEATURE_NAMES) + len(TEMP_FEATURE_NAMES))

    def forward(self, features, absolute, temporal, abs_mask, temp_mask, no_entropy=False, no_temporal=False,
                preserve_entropy_metadata=False):
        if preserve_entropy_metadata and (no_entropy or no_temporal):
            # The paired ablation changes entropy numbers only. Length, coverage,
            # validity and history availability remain identical in every arm.
            temporal = temporal.clone()
            temporal[:, [0, 1, 2, 3, 6, 7, 8, 9]] = 0
            if no_entropy:
                absolute = absolute.clone()
                absolute[:, [0, 6]] = 0
        elif no_entropy:
            absolute, temporal = torch.zeros_like(absolute), torch.zeros_like(temporal)
        elif no_temporal:
            temporal = torch.zeros_like(temporal)
        sem = self.semantic(features).squeeze(-1)
        abs_logit = sem + self.absolute(torch.cat((features, absolute), -1)).squeeze(-1) * abs_mask
        full = abs_logit + self.temporal(torch.cat((features, absolute, temporal), -1)).squeeze(-1) * temp_mask
        return torch.stack((sem, abs_logit, full), -1)


class _InvalidRecord(ValueError):
    pass


def _validate_backbone_config(config) -> None:
    if config.model_type != "qwen3":
        raise ValueError("prefix-value encoder must be a causal Qwen3 backbone; bidirectional encoders leak future actions")
    scaling = getattr(config, "rope_scaling", None) or {}
    rope_type = str(scaling.get("rope_type", scaling.get("type", "default"))).lower()
    if "dynamic" in rope_type or rope_type == "longrope":
        raise ValueError("sequence-length-dependent RoPE is incompatible with exact causal-prefix feature reuse")


def _cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _weighted_auc(scores: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> float:
    """Weighted Mann-Whitney AUROC, with half credit for tied predictions."""
    order = torch.argsort(scores)
    scores, labels, weights = scores[order], labels[order], weights[order]
    positive = float(weights[labels == 1].sum())
    negative = float(weights[labels == 0].sum())
    if positive == 0 or negative == 0:
        return float("nan")
    cumulative_negative = 0.0
    concordant = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        group_labels = labels[start:end]
        group_weights = weights[start:end]
        group_positive = float(group_weights[group_labels == 1].sum())
        group_negative = float(group_weights[group_labels == 0].sum())
        concordant += group_positive * (cumulative_negative + 0.5 * group_negative)
        cumulative_negative += group_negative
        start = end
    return concordant / (positive * negative)


class PrefixValueScorer:
    """One fixed encoder and two distinct copies of a shared value head.

    ``save(path)`` and ``load(path)`` accept a checkpoint FILE path, e.g.
    ``prefix_value.pt``. No transformer weights are duplicated in the checkpoint;
    the same immutable ``model_path`` (and preferably ``revision``) must remain
    available on resume. Subclasses may override ``_load_backbone`` for tests.
    """

    def __init__(self, config: dict):
        self.config = {**_DEFAULTS, **dict(config)}
        self._validate_config()
        self.prediction_mode = self.config["prediction_mode"]
        self.device = torch.device(self.config["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Prefix value scorer requested CUDA, but CUDA is unavailable")
        if self.device.type == "cpu":
            torch.set_num_threads(int(self.config["cpu_num_threads"]))
        self.encoder, self.tokenizer = self._load_backbone()
        self.encoder.to(self.device).eval()
        self.encoder.requires_grad_(False)
        hidden_size = int(self.encoder.config.hidden_size)
        self.feature_dim = hidden_size + _STRUCTURAL_DIM
        self.encoder_identity = {
            "model_path": str(self.config["model_path"]),
            "revision": self.config.get("revision"),
            "model_type": getattr(self.encoder.config, "model_type", None),
            "commit_hash": getattr(self.encoder.config, "_commit_hash", None),
            "hidden_size": hidden_size,
            "vocab_size": getattr(self.encoder.config, "vocab_size", None),
            "serialization": "math-prefix-segments-v2",
            "structural_dim": _STRUCTURAL_DIM,
            "entropy_schema": FEATURE_SCHEMA,
            "entropy_definition": "action-mean-top16-renormalized-div-log16-valid-tokens-v1",
        }
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.config["seed"]))
            self.candidate_head = EntropyValueHead(self.feature_dim, int(self.config["head_hidden_dim"]),
                                                   int(self.config["entropy_hidden_dim"]), float(self.config["dropout"])).to(self.device)
        self.deployed_head = copy.deepcopy(self.candidate_head).eval()
        self.deployed_head.requires_grad_(False)
        self.control_head = copy.deepcopy(self.candidate_head)
        self.temporal_control_head = copy.deepcopy(self.candidate_head)
        self.optimizer = torch.optim.AdamW(
            self.candidate_head.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        self._rng = torch.Generator(device="cpu").manual_seed(int(self.config["seed"]))
        self._replay: OrderedDict[str, dict] = OrderedDict()
        self._pending: OrderedDict[str, dict] = OrderedDict()
        self.control_optimizer = torch.optim.AdamW(self.control_head.parameters(),
                                                   lr=float(self.config["learning_rate"]), weight_decay=float(self.config["weight_decay"]))
        self.temporal_control_optimizer = torch.optim.AdamW(self.temporal_control_head.parameters(),
                                                            lr=float(self.config["learning_rate"]), weight_decay=float(self.config["weight_decay"]))
        self.scaler = None
        self.bad_windows = 0
        self.role_bad_windows = {"solver": 0, "verifier": 0}
        self.temporal_bad_windows = {"solver": 0, "verifier": 0}
        self.last_validation_fingerprint = None
        self.pretrained = False
        # This checkpoint-owned mode survives initialization into main training,
        # whose runtime config need not repeat the offline bootstrap option.
        self.deployment_mode = "absolute_bootstrap" if self.config["allow_absolute_only_pretrain"] else "full"
        self.pretraining_mode = "absolute" if self.config["allow_absolute_only_pretrain"] else "full"
        self._absolute_pretrain_active = False
        # Checkpoint-owned opt-in: existing qualified checkpoints keep their
        # original training/ablation behavior unless explicitly warm-started.
        self.training_protocol = "legacy"
        self.warm_started = False
        self.reliability = {"solver": 0.0, "verifier": 0.0}
        self.temporal_reliability = {"solver": 0.0, "verifier": 0.0}
        self.ready = False
        self.version = 0
        self.step = 0
        self.last_metrics: dict = {}
        self.frozen_parameters = sum(parameter.numel() for parameter in self.encoder.parameters())
        self.head_parameters = sum(parameter.numel() for parameter in self.candidate_head.parameters())
        self.semantic_reliability = {"solver": 0.0, "verifier": 0.0}
        self.semantic_validation_fingerprints = {"solver": None, "verifier": None}
        self.startup_semantic_report = {}
        if self.prediction_mode == "semantic_only":
            self._configure_semantic_training()

    def _configure_semantic_training(self):
        self.training_protocol = _SEMANTIC_TRAINING_PROTOCOL
        self.deployment_mode = "semantic_only"
        self.candidate_head.requires_grad_(False)
        self.candidate_head.semantic.requires_grad_(True)
        for head in (self.encoder, self.deployed_head, self.control_head, self.temporal_control_head):
            head.eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.candidate_head.semantic.parameters(),
                                          lr=float(self.config["learning_rate"]),
                                          weight_decay=float(self.config["weight_decay"]))

    def _validate_config(self) -> None:
        if self.config["prediction_mode"] not in ("entropy_aware", "semantic_only"):
            raise ValueError("prediction_mode must be entropy_aware or semantic_only")
        for name in ("semantic_min_val_prefixes", "semantic_min_val_questions", "semantic_min_val_per_class"):
            value = self.config[name]
            if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != float(value) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(float(self.config["semantic_min_val_auc"])) or not 0 <= float(self.config["semantic_min_val_auc"]) <= 1:
            raise ValueError("semantic_min_val_auc must be finite and in [0, 1]")
        if not math.isfinite(float(self.config["semantic_min_brier_improvement"])) or float(self.config["semantic_min_brier_improvement"]) < 0:
            raise ValueError("semantic_min_brier_improvement must be finite and nonnegative")
        if not isinstance(self.config["allow_absolute_only_pretrain"], bool):
            raise ValueError("allow_absolute_only_pretrain must be boolean")
        if not self.config.get("model_path"):
            raise ValueError("prefix_value.model_path must identify the fixed initial encoder checkpoint")
        if self.config["torch_dtype"] not in ("float32", "float16", "bfloat16"):
            raise ValueError("torch_dtype must be float32, float16, or bfloat16")
        for name in (
            "max_length", "head_hidden_dim", "train_epochs", "train_batch_size",
            "replay_max_trajectories", "min_train_trajectories", "min_val_trajectories",
            "min_val_per_class", "min_train_questions", "min_val_questions", "cpu_num_threads",
            "entropy_hidden_dim", "min_entropy_val_prefixes", "miscalibration_patience",
        ):
            if int(self.config[name]) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < float(self.config["validation_fraction"]) < 1:
            raise ValueError("validation_fraction must lie strictly between zero and one")
        if not 0 <= float(self.config["min_val_auc"]) <= 1:
            raise ValueError("min_val_auc must be in [0, 1]")
        if float(self.config["learning_rate"]) <= 0:
            raise ValueError("learning_rate must be positive")
        if float(self.config["min_brier_improvement"]) < 0:
            raise ValueError("min_brier_improvement cannot be negative")
        if float(self.config["candidate_brier_tolerance"]) < 0:
            raise ValueError("candidate_brier_tolerance cannot be negative")
        for name in ("min_entropy_brier_gain", "min_temporal_brier_gain"):
            if not math.isfinite(float(self.config[name])) or float(self.config[name]) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= float(self.config["dropout"]) < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 0 <= float(self.config["min_entropy_coverage"]) <= 1:
            raise ValueError("min_entropy_coverage must be in [0,1]")
        if min(float(self.config["scaler_clip"]), float(self.config["scaler_floor"])) <= 0:
            raise ValueError("scaler_clip and scaler_floor must be positive")

    def _load_backbone(self):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        kwargs = {"trust_remote_code": bool(self.config["trust_remote_code"])}
        if self.config.get("revision") is not None:
            kwargs["revision"] = self.config["revision"]
        model_config = AutoConfig.from_pretrained(self.config["model_path"], **kwargs)
        _validate_backbone_config(model_config)
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_path"], **kwargs)
        encoder = AutoModel.from_pretrained(
            self.config["model_path"],
            torch_dtype=getattr(torch, self.config["torch_dtype"]),
            attn_implementation=self.config["attn_implementation"],
            **kwargs,
        )
        return encoder, tokenizer

    def _is_validation(self, question_key: str) -> bool:
        payload = (str(self.config["holdout_salt"]) + "\0" + str(question_key)).encode("utf-8")
        fraction = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / (1 << 64)
        return fraction < float(self.config["validation_fraction"])

    @staticmethod
    def _state_features(solver_count: int, verifier_count: int, budget: int,
                        next_role: str, last_valid: bool) -> list[float]:
        remaining = max(0, budget - solver_count)
        return [
            math.log1p(solver_count), math.log1p(verifier_count),
            math.log1p(remaining), math.log1p(budget),
            float(next_role == "solver"), float(next_role == "verifier"),
            float(next_role == "terminal"), float(last_valid),
        ]

    def _segments_and_states(self, record: dict):
        question = str(record["question"])
        budget = int(record["max_solver_turns"])
        if budget < 1:
            raise ValueError("max_solver_turns must be positive")
        actions = record["actions"]
        if not isinstance(actions, list):
            raise ValueError("actions must be a list")
        segments = ["[PROBLEM]\n" + question + "\n[INTERACTION]\n"]
        states = [self._state_features(0, 0, budget, "solver", True)]
        solver_count = verifier_count = 0
        for action in actions:
            role = str(action["role"]).lower()
            text = str(action["text"])
            valid = bool(action.get("valid", True))
            if role == "solver":
                solver_count += 1
                next_role = "terminal" if solver_count >= budget else "verifier"
            elif role == "verifier":
                verifier_count += 1
                verdict = text
                # Match the math controller: approve takes priority, unrecognized
                # verdicts stop, and only reject asks for another solver action.
                if "<verify>approve</verify>" in verdict:
                    next_role = "terminal"
                elif "<verify>reject</verify>" in verdict and solver_count < budget:
                    next_role = "solver"
                else:
                    next_role = "terminal"
            else:
                raise ValueError(f"unknown action role: {role!r}")
            segments.append(f"\n[{role.upper()}]\n{text}\n[END_ACTION]\n")
            states.append(self._state_features(solver_count, verifier_count, budget, next_role, valid))
        return segments, states

    def _encode(self, record):
        try:
            segments, states = self._segments_and_states(record)
            entropy = (prefix_entropy_features(record["actions"], float(self.config["min_entropy_coverage"]))
                       if self.prediction_mode == "entropy_aware" else None)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise _InvalidRecord(str(error)) from error
        token_ids, boundaries = [], []
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is not None:
            token_ids.append(int(bos))
        max_length = int(self.config["max_length"])
        total = len(token_ids)
        for segment_index, segment in enumerate(segments):
            ids = self.tokenizer.encode(segment, add_special_tokens=False)
            if not ids:
                raise _InvalidRecord("empty tokenized prefix segment")
            total += len(ids)
            # Never score the interior of a truncated action. Earlier complete
            # boundaries still receive labels/predictions and keep their state.
            if len(token_ids) + len(ids) <= max_length and len(boundaries) == segment_index:
                token_ids.extend(ids)
                boundaries.append(len(token_ids) - 1)
        count = len(boundaries)
        if not count:
            return None, total
        self.encoder.eval()
        with torch.inference_mode():
            ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            output = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
            semantic = output.last_hidden_state[0, boundaries].float().cpu()
        features = torch.cat((semantic.clone(), torch.tensor(states[:count], dtype=torch.float32)), -1).half().contiguous()
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("nonfinite encoder features")
        action_ok = [True]
        for action in record["actions"][:count - 1]:
            response_tokens = action.get("response_token_count", action.get("entropy_token_count"))
            if response_tokens is None:
                response_tokens = len(self.tokenizer.encode(str(action["text"]), add_special_tokens=False))
            action_ok.append(bool(_metadata._binary(action.get("valid", True))
                                  and not _metadata._binary(action.get("truncated", False))
                                  and math.isfinite(float(response_tokens)) and float(response_tokens) > 0))
        return {"features": features,
                "absolute": torch.from_numpy(entropy["absolute"][:count]).clone() if entropy else torch.zeros(count, len(ABS_FEATURE_NAMES)),
                "temporal": torch.from_numpy(entropy["temporal"][:count]).clone() if entropy else torch.zeros(count, len(TEMP_FEATURE_NAMES)),
                "abs_mask": torch.from_numpy(entropy["absolute_available"][:count]).clone() if entropy else torch.zeros(count, dtype=torch.bool),
                "temp_mask": torch.from_numpy(entropy["temporal_available"][:count]).clone() if entropy else torch.zeros(count, dtype=torch.bool),
                "prefix_roles": ["initial"] + [a["role"].lower() for a in record["actions"][:count - 1]],
                "prefix_terminal": torch.tensor([bool(s[6]) for s in states[:count]]),
                "semantic_metadata_schema": _SEMANTIC_METADATA_SCHEMA,
                "semantic_available": torch.tensor(action_ok, dtype=torch.bool)}, total

    @staticmethod
    def _flatten(rows):
        keys = ("features", "absolute", "temporal", "abs_mask", "temp_mask", "prefix_terminal")
        result = {key: torch.cat([row[key] for row in rows]) for key in keys}
        result["labels"] = torch.cat([torch.full((len(row["features"]),), row["label"]) for row in rows])
        # No inverse trajectory-length weights: future stopping length must not
        # change the success-probability target at an already observed prefix.
        result["weights"] = torch.ones(len(result["labels"]))
        result["roles"] = sum([row["prefix_roles"] for row in rows], [])
        return result

    def _fit_scaler(self, train_rows):
        """Fit once on training questions only; frozen throughout online RL."""
        data = self._flatten(train_rows)
        scaler = {"schema": FEATURE_SCHEMA, "clip": float(self.config["scaler_clip"])}
        for key, names in (("absolute", ABS_FEATURE_NAMES), ("temporal", TEMP_FEATURE_NAMES)):
            x = data[key].float()
            centers, scales = [], []
            for j, name in enumerate(names):
                block_mask = x[:, (j // 6) * 6 + 5].bool()
                values = x[block_mask, j]
                # Binary validity/missingness keep their exact meaning.
                binary = name.endswith(("valid", "truncated", "available"))
                median = values.median() if len(values) and not binary else torch.tensor(0.0)
                mad = (values - median).abs().median() if len(values) and not binary else torch.tensor(1.0)
                centers.append(float(median))
                scales.append(max(float(mad) * 1.4826, float(self.config["scaler_floor"])) if not binary else 1.0)
            scaler[key] = {"names": list(names), "center": centers, "scale": scales}
        self.scaler = scaler

    def _batch(self, data, indices):
        result = {key: data[key][indices].to(self.device, dtype=torch.float32)
                  for key in ("features", "absolute", "temporal", "abs_mask", "temp_mask")}
        if self.scaler is None:
            raise RuntimeError("entropy feature scaler has not been fitted")
        for key in ("absolute", "temporal"):
            spec = self.scaler[key]
            x = result[key]
            observed = x.clone()
            x = ((x - torch.tensor(spec["center"], device=self.device)) /
                 torch.tensor(spec["scale"], device=self.device)).clamp(-self.scaler["clip"], self.scaler["clip"])
            for offset in (0, 6):
                x[:, offset:offset + 6] *= observed[:, offset + 5:offset + 6]
            result[key] = x
        return result

    def _control_training_data(self, data):
        """Use the same own-role, nonterminal entropy domain for fit and serving."""
        if self.training_protocol != _PAIRED_TRAINING_PROTOCOL:
            return data
        roles = data.get("roles", data.get("prefix_roles"))
        if roles is None or len(roles) != len(data["features"]):
            raise ValueError("paired entropy training requires prefix roles")
        nonterminal = ~data["prefix_terminal"].bool()
        own_abs = torch.tensor([bool(data["absolute"][i, (0 if role == "solver" else 6) + 5])
                                if role in ("solver", "verifier") else False
                                for i, role in enumerate(roles)], dtype=torch.bool)
        own_temp = torch.tensor([bool(data["temporal"][i, (0 if role == "solver" else 6) + 5])
                                 if role in ("solver", "verifier") else False
                                 for i, role in enumerate(roles)], dtype=torch.bool)
        return {**data, "abs_mask": nonterminal & own_abs,
                "temp_mask": nonterminal & own_abs & own_temp}

    def _predict(self, head, data, no_entropy=False, no_temporal=False):
        data = self._control_training_data(data)
        head.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(data["features"]), int(self.config["train_batch_size"])):
                batch = self._batch(data, slice(start, start + int(self.config["train_batch_size"])))
                scores.append(torch.sigmoid(head(
                    **batch, no_entropy=no_entropy, no_temporal=no_temporal,
                    preserve_entropy_metadata=self.training_protocol == _PAIRED_TRAINING_PROTOCOL)).cpu())
        return torch.cat(scores).clone()

    def _predict_semantic(self, head, data):
        """No residual forward, entropy scaling or entropy masks in this path."""
        head.semantic.eval()
        result = []
        size = int(self.config["train_batch_size"])
        with torch.inference_mode():
            for start in range(0, len(data["features"]), size):
                features = data["features"][start:start + size].to(self.device, dtype=torch.float32)
                result.append(torch.sigmoid(head.semantic(features).squeeze(-1)).cpu())
        return torch.cat(result).clone() if result else torch.empty(0)

    @staticmethod
    def _route_predictions(scores, data, temporal_permissions):
        """Only a qualified just-completed role may deploy its temporal logit.

        Feature availability alone is not a permission. In particular, a
        Verifier prefix may contain old Solver temporal features while the
        Verifier branch has never received nonterminal temporal validation.
        """
        if temporal_permissions is None:
            return scores
        roles = data.get("roles", data.get("prefix_roles"))
        if roles is None or len(roles) != len(scores):
            raise ValueError("prefix roles are required for conditional value deployment")
        enabled = torch.tensor([float(temporal_permissions.get(role, 0.0)) > 0 for role in roles],
                               dtype=torch.bool, device=scores.device)
        result = scores.clone()
        result[:, 2] = torch.where(enabled, scores[:, 2], scores[:, 1])
        return result

    def _deployment_stage(self):
        if self.prediction_mode == "semantic_only":
            return "semantic_only"
        if self.deployment_mode == "full":
            return "full"
        enabled = sum(float(value) > 0 for value in self.temporal_reliability.values())
        return "full" if enabled == 2 else "hybrid" if enabled else "absolute"

    def _deployment_metrics(self):
        return {"deployment_stage": self._deployment_stage(), "pretraining_mode": self.pretraining_mode,
                "prediction_mode": self.prediction_mode,
                "semantic_only": float(self.prediction_mode == "semantic_only"),
                "semantic_ready": float(self.prediction_mode == "semantic_only" and self.ready),
                **{"semantic_reliability_" + role: value for role, value in self.semantic_reliability.items()},
                "absolute_bootstrap_mode": float(self.deployment_mode == "absolute_bootstrap"),
                "warm_started": float(self.warm_started),
                "paired_entropy_training": float(self.training_protocol == _PAIRED_TRAINING_PROTOCOL),
                "deployment_temporal_roles": sum(float(value) > 0 for value in self.temporal_reliability.values())}

    def prepare(self, records):
        started = time.perf_counter()
        values, seen = {}, set()
        encoded = invalid = overlong = partial = tokens = score_only = 0
        for record in records:
            try:
                uid = str(record["traj_uid"])
                if not uid or uid in seen:
                    raise _InvalidRecord("duplicate or empty trajectory identity")
                seen.add(uid)
                label = float(record["label"])
                if label not in (0.0, 1.0):
                    raise _InvalidRecord("terminal label must be binary")
                train_eligible = record.get("train_eligible", True)
                if not isinstance(train_eligible, bool):
                    raise _InvalidRecord("train_eligible must be boolean")
                # Canonical text defines split across all run/source identifiers.
                qkey = hashlib.sha256(str(record["question"]).strip().encode()).hexdigest()
                row, total = self._encode(record)
            except (KeyError, TypeError, ValueError, OverflowError):
                invalid += 1
                continue
            if row is None:
                overlong += 1
                continue
            n = len(row["features"])
            partial += int(n < len(record["actions"]) + 1)
            if self.ready and self.prediction_mode == "semantic_only":
                prediction = self._predict_semantic(self.deployed_head, row)
                available = self._semantic_mask(row, include_initial=True)
                values[uid] = [
                    {"sem": float(score), "control_value": float(score), "prediction_source": "semantic_only",
                     "semantic_available": bool(available[i]) and math.isfinite(float(score)),
                     "action_semantic_available": bool(i and available[i]) and math.isfinite(float(score)),
                     "absolute_available": False, "temporal_available": False,
                     "action_absolute_available": False, "action_temporal_available": False}
                    for i, score in enumerate(prediction)
                ]
            elif self.ready:
                prediction = self._predict(self.deployed_head, row)
                control_row = self._control_training_data(row)
                if self.deployment_mode == "absolute_bootstrap":
                    prediction = self._route_predictions(prediction, row, self.temporal_reliability)
                if not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError("nonfinite deployed prefix value prediction")
                values[uid] = []
                for i, score in enumerate(prediction):
                    offset = 0 if row["prefix_roles"][i] == "solver" else 6
                    values[uid].append(dict(zip(("sem", "abs", "full"), score.tolist()),
                                            absolute_available=bool(control_row["abs_mask"][i]),
                                            temporal_available=bool(control_row["temp_mask"][i]),
                                            action_absolute_available=bool(i and row["absolute"][i, offset + 5]
                                                                           and control_row["abs_mask"][i]),
                                            action_temporal_available=bool(i and row["temporal"][i, offset + 5]
                                                                           and control_row["temp_mask"][i])))
            if train_eligible:
                self._pending[uid] = {**row, "traj_uid": uid, "question_key": qkey, "label": label}
                self._pending.move_to_end(uid)
                while len(self._pending) > int(self.config["replay_max_trajectories"]):
                    self._pending.popitem(last=False)
            else:
                score_only += 1
            encoded += 1
            tokens += total
        metrics = {"ready": float(self.ready), "version": self.version, "records": len(records),
                   "encoded_trajectories": encoded, "score_only_trajectories": score_only, "skipped_invalid": invalid,
                   "skipped_overlong": overlong, "partial_trajectories": partial,
                   "encoded_tokens": tokens, "coverage": encoded / len(records) if records else 0.0,
                   "prepare_seconds": time.perf_counter() - started,
                   "pending_trajectories": len(self._pending), "frozen_parameters": self.frozen_parameters,
                   "head_parameters": self.head_parameters,
                   "matched_control_parameters": 2 * self.head_parameters,
                   "trainable_head_parameters": (sum(p.numel() for p in self.candidate_head.semantic.parameters())
                                                 if self.prediction_mode == "semantic_only" else 3 * self.head_parameters)}
        metrics.update(self._deployment_metrics())
        metrics.update({"reliability_" + role: value for role, value in self.reliability.items()})
        metrics.update({"temporal_reliability_" + role: value for role, value in self.temporal_reliability.items()})
        return {"ready": self.ready, "version": self.version, "values": values, "warm_started": self.warm_started,
                "prediction_mode": self.prediction_mode, "semantic_ready": self.prediction_mode == "semantic_only" and self.ready,
                "semantic_reliability": self.semantic_reliability.copy(),
                "startup_semantic_report": copy.deepcopy(self.startup_semantic_report),
                "reliability": self.reliability.copy(), "temporal_reliability": self.temporal_reliability.copy(), "metrics": metrics}

    def _train(self, rows, epochs, stage="all", control=False):
        # Dropout and minibatch order both descend from the serialized worker
        # generator. A scorer update does not consume another component's RNG.
        seed = int(torch.randint(0, 2 ** 31, (1,), generator=self._rng).item())
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            return self._train_impl(rows, epochs, stage, control)

    def _train_impl(self, rows, epochs, stage="all", control=False):
        if self.prediction_mode == "semantic_only":
            if control or stage not in ("all", "semantic"):
                raise ValueError("semantic_only trains only candidate.semantic")
            return self._train_semantic_impl(rows, epochs)
        data = self._control_training_data(self._flatten(rows))
        if control == "temporal":
            head, optimizer = self.temporal_control_head, self.temporal_control_optimizer
        elif control:
            head, optimizer = self.control_head, self.control_optimizer
        else:
            head, optimizer = self.candidate_head, self.optimizer
        for name, parameter in head.named_parameters():
            if stage == "absolute":
                parameter.requires_grad_(name.startswith("absolute."))
            else:
                parameter.requires_grad_(stage == "all" or (name.startswith("semantic.") == (stage == "semantic")))
        head.train()
        if stage in ("entropy", "absolute"):
            head.semantic.eval()
        if stage == "absolute":
            head.temporal.eval()
        total_loss = total_count = 0.0
        for _ in range(int(epochs)):
            indices = torch.randperm(len(data["labels"]), generator=self._rng)
            for start in range(0, len(indices), int(self.config["train_batch_size"])):
                ix = indices[start:start + int(self.config["train_batch_size"])]
                batch = self._batch(data, ix)
                logits = head(**batch, no_entropy=control is True, no_temporal=control == "temporal",
                              preserve_entropy_metadata=self.training_protocol == _PAIRED_TRAINING_PROTOCOL)
                labels = data["labels"][ix].to(self.device)
                losses = F.binary_cross_entropy_with_logits(logits, labels[:, None].expand_as(logits), reduction="none")
                mask = torch.stack((torch.ones_like(batch["abs_mask"]), batch["abs_mask"], batch["temp_mask"]), -1)
                if stage == "semantic":
                    mask[:, 1:] = 0
                elif stage == "entropy":
                    mask[:, 0] = 0
                elif stage == "absolute":
                    mask[:, 0] = 0
                    mask[:, 2] = 0
                if not bool(mask.any()):
                    continue
                loss = (losses * mask).sum() / mask.sum()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite entropy value loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach()) * len(ix)
                total_count += len(ix)
        head.eval().requires_grad_(True)
        return total_loss / max(total_count, 1)

    def _evaluate(self, head, rows, prior, control_scores=None, temporal_control_scores=None,
                  temporal_permissions=None):
        data = self._flatten(rows)
        raw_scores = self._predict(head, data).double()
        scores = self._route_predictions(raw_scores, data, temporal_permissions)
        labels = data["labels"].double()
        errors = (scores - labels[:, None]).square()
        raw_errors = (raw_scores - labels[:, None]).square()
        nonterminal = ~data["prefix_terminal"].bool()
        # Control is evaluated on the just-completed action's own role, not
        # merely because the other role has some older entropy in the prefix.
        own_abs = torch.tensor([bool(data["absolute"][i, (0 if r == "solver" else 6) + 5]) if r != "initial" else False
                                for i, r in enumerate(data["roles"])])
        own_temp = torch.tensor([bool(data["temporal"][i, (0 if r == "solver" else 6) + 5]) if r != "initial" else False
                                 for i, r in enumerate(data["roles"])])
        entropy_scope = nonterminal & own_abs
        temporal_scope = nonterminal & own_temp
        def mean(x, mask):
            return float(x[mask].mean()) if bool(mask.any()) else float("nan")
        result = {"brier": float(errors[:, 2].mean()), "prior_brier": float((labels - prior).square().mean()),
                  "auc": _weighted_auc(scores[:, 2], labels, torch.ones_like(labels)),
                  "abs_auc": _weighted_auc(scores[:, 1], labels, torch.ones_like(labels)),
                  "nonterminal_brier": mean(errors[:, 2], nonterminal),
                  "entropy_prefixes": int(entropy_scope.sum()), "temporal_prefixes": int(temporal_scope.sum()),
                  "absolute_gain": mean(errors[:, 0] - errors[:, 1], entropy_scope),
                  "temporal_gain": mean(raw_errors[:, 1] - raw_errors[:, 2], temporal_scope)}
        active_control = None if control_scores is None else self._route_predictions(control_scores, data, temporal_permissions)
        control_error = None if active_control is None else (active_control[:, 2].double() - labels).square()
        temporal_control_error = None if temporal_control_scores is None else (temporal_control_scores[:, 2].double() - labels).square()
        result["matched_control_gain"] = mean(control_error - errors[:, 2], entropy_scope) if control_error is not None else float("nan")
        result["abs_matched_control_gain"] = (mean((control_scores[:, 1].double() - labels).square() - errors[:, 1], entropy_scope)
                                              if control_scores is not None else float("nan"))
        result["matched_temporal_gain"] = mean(temporal_control_error - raw_errors[:, 2], temporal_scope) if temporal_control_error is not None else float("nan")
        for j, branch in enumerate(("sem", "abs", "full")):
            result[f"{branch}_brier"] = float(errors[:, j].mean())
        for role in ("solver", "verifier"):
            scope = entropy_scope & torch.tensor([r == role for r in data["roles"]])
            temporal_role = temporal_scope & torch.tensor([r == role for r in data["roles"]])
            result[f"{role}_prefixes"] = int(scope.sum())
            result[f"{role}_brier"] = mean(errors[:, 2], scope)
            result[f"{role}_abs_brier"] = mean(errors[:, 1], scope)
            result[f"{role}_prior_brier"] = mean((labels - prior).square(), scope)
            result[f"{role}_matched_gain"] = mean(control_error - errors[:, 2], scope) if control_error is not None else float("nan")
            result[f"{role}_absolute_gain"] = mean(errors[:, 0] - errors[:, 1], scope)
            result[f"{role}_abs_matched_gain"] = (mean((control_scores[:, 1].double() - labels).square() - errors[:, 1], scope)
                                                    if control_scores is not None else float("nan"))
            result[f"{role}_temporal_prefixes"] = int(temporal_role.sum())
            result[f"{role}_temporal_gain"] = mean(raw_errors[:, 1] - raw_errors[:, 2], temporal_role)
            result[f"{role}_matched_temporal_gain"] = mean(temporal_control_error - raw_errors[:, 2], temporal_role) if temporal_control_error is not None else float("nan")
        return result

    def _role_quality(self, metrics, role):
        return (metrics[f"{role}_prefixes"] >= int(self.config["min_entropy_val_prefixes"])
                and metrics[f"{role}_brier"] < metrics[f"{role}_prior_brier"]
                and metrics[f"{role}_matched_gain"] > float(self.config["min_entropy_brier_gain"]))

    def _role_temporal_quality(self, metrics, role):
        return (self._role_quality(metrics, role)
                and metrics[f"{role}_temporal_prefixes"] >= int(self.config["min_entropy_val_prefixes"])
                and metrics[f"{role}_temporal_gain"] > float(self.config["min_temporal_brier_gain"])
                and metrics[f"{role}_matched_temporal_gain"] > float(self.config["min_temporal_brier_gain"]))

    def _quality_passes(self, metrics):
        return (math.isfinite(metrics["auc"]) and metrics["auc"] >= float(self.config["min_val_auc"])
                and metrics["brier"] < metrics["prior_brier"] - float(self.config["min_brier_improvement"])
                and metrics["entropy_prefixes"] >= int(self.config["min_entropy_val_prefixes"])
                and metrics["temporal_prefixes"] >= int(self.config["min_entropy_val_prefixes"])
                and metrics["absolute_gain"] > float(self.config["min_entropy_brier_gain"])
                and metrics["temporal_gain"] > float(self.config["min_temporal_brier_gain"])
                and metrics["matched_control_gain"] > float(self.config["min_entropy_brier_gain"])
                and metrics["matched_temporal_gain"] > float(self.config["min_temporal_brier_gain"]))

    def _bootstrap_quality_passes(self, metrics):
        """Absolute entropy is a required learned signal, never an optional input."""
        return (math.isfinite(metrics["auc"]) and metrics["auc"] >= float(self.config["min_val_auc"])
                and metrics["brier"] < metrics["prior_brier"] - float(self.config["min_brier_improvement"])
                and math.isfinite(metrics["abs_auc"]) and metrics["abs_auc"] >= float(self.config["min_val_auc"])
                and metrics["abs_brier"] < metrics["prior_brier"] - float(self.config["min_brier_improvement"])
                and metrics["entropy_prefixes"] >= int(self.config["min_entropy_val_prefixes"])
                and metrics["absolute_gain"] > float(self.config["min_entropy_brier_gain"])
                and metrics["abs_matched_control_gain"] > float(self.config["min_entropy_brier_gain"])
                and metrics["matched_control_gain"] > float(self.config["min_entropy_brier_gain"]))

    def _bootstrap_role_quality(self, metrics, role):
        return (self._role_quality(metrics, role)
                and metrics[f"{role}_abs_brier"] < metrics[f"{role}_prior_brier"]
                and metrics[f"{role}_absolute_gain"] > float(self.config["min_entropy_brier_gain"])
                and metrics[f"{role}_abs_matched_gain"] > float(self.config["min_entropy_brier_gain"]))

    def _bootstrap_temporal_quality(self, metrics, role):
        return (self._bootstrap_role_quality(metrics, role)
                and self._role_temporal_quality(metrics, role))

    @staticmethod
    def _semantic_mask(row, include_initial=False):
        """Availability is action metadata, never entropy coverage or a label."""
        n = len(row["features"])
        available = row.get("semantic_available")
        if (row.get("semantic_metadata_schema") != _SEMANTIC_METADATA_SCHEMA
                or not torch.is_tensor(available) or available.dtype != torch.bool or available.shape != (n,)):
            return torch.zeros(n, dtype=torch.bool)
        terminal = row.get("prefix_terminal")
        roles = row.get("prefix_roles", [])
        if not torch.is_tensor(terminal) or terminal.shape != (n,) or len(roles) != n:
            return torch.zeros(n, dtype=torch.bool)
        mask = available.cpu().clone() & ~terminal.cpu().bool()
        mask &= torch.tensor([r in ("solver", "verifier") or (include_initial and i == 0 and r == "initial")
                              for i, r in enumerate(roles)], dtype=torch.bool)
        return mask

    def _train_semantic_impl(self, rows, epochs):
        features = torch.cat([r["features"] for r in rows])
        labels = torch.cat([torch.full((len(r["features"]),), float(r["label"])) for r in rows])
        self.candidate_head.eval()
        self.candidate_head.semantic.train()
        total_loss = total_count = 0.0
        for _ in range(int(epochs)):
            indices = torch.randperm(len(labels), generator=self._rng)
            for start in range(0, len(indices), int(self.config["train_batch_size"])):
                ix = indices[start:start + int(self.config["train_batch_size"])]
                logits = self.candidate_head.semantic(features[ix].to(self.device, dtype=torch.float32)).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels[ix].to(self.device))
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.detach()) * len(ix)
                total_count += len(ix)
        self.candidate_head.eval()
        return total_loss / total_count if total_count else float("nan")

    def _semantic_row_predictions(self, head, rows):
        """Stream cached features in bounded CPU/GPU batches, no re-encoding."""
        size = int(self.config["train_batch_size"])
        pending, count, scores = [], 0, []
        for row in rows:
            features = row["features"]
            for start in range(0, len(features), size):
                piece = features[start:start + size]
                pending.append(piece)
                count += len(piece)
                if count >= size:
                    scores.append(self._predict_semantic(head, {"features": torch.cat(pending)}))
                    pending, count = [], 0
        if pending:
            scores.append(self._predict_semantic(head, {"features": torch.cat(pending)}))
        return torch.cat(scores).double() if scores else torch.empty(0, dtype=torch.float64)

    def _semantic_priors(self, rows):
        counts = {key: 0 for key in ("all", "nonterminal", "solver", "verifier")}
        sums = {key: 0.0 for key in counts}
        for row in rows:
            mask = self._semantic_mask(row)
            scope_counts = {"all": len(row["features"]), "nonterminal": int(mask.sum())}
            for role in ("solver", "verifier"):
                scope_counts[role] = sum(bool(mask[i]) for i, r in enumerate(row["prefix_roles"]) if r == role)
            for key, n in scope_counts.items():
                counts[key] += n
                sums[key] += n * float(row["label"])
        return {key: sums[key] / n if n else float("nan") for key, n in counts.items()}

    def _evaluate_semantic(self, head, rows, train_rows):
        priors = self._semantic_priors(train_rows)
        predictions = self._semantic_row_predictions(head, rows)
        labels = torch.cat([torch.full((len(r["features"]),), float(r["label"]), dtype=torch.float64) for r in rows]) if rows else torch.empty(0, dtype=torch.float64)
        scope = torch.cat([self._semantic_mask(r) for r in rows]) if rows else torch.empty(0, dtype=torch.bool)
        roles = [role for row in rows for role in row["prefix_roles"]]
        errors = (predictions - labels).square()

        def statistics(mask, prior, prefix):
            n = int(mask.sum())
            y, p = labels[mask], predictions[mask]
            finite = bool(torch.isfinite(p).all()) and bool(((p >= 0) & (p <= 1)).all())
            return {prefix + "prefixes": n,
                    prefix + "brier": float(errors[mask].mean()) if n and finite else float("nan"),
                    prefix + "prior_brier": float((y - prior).square().mean()) if n else float("nan"),
                    prefix + "auc": _weighted_auc(p, y, torch.ones_like(y)) if n and finite else float("nan")}

        result = statistics(torch.ones(len(labels), dtype=torch.bool), priors["all"], "sem_")
        # Preserve candidate_sem_brier's original all-prefix definition.
        result["prior_brier"] = result["sem_prior_brier"]
        result.update(statistics(scope, priors["nonterminal"], "sem_nonterminal_"))
        for role in ("solver", "verifier"):
            mask = scope & torch.tensor([r == role for r in roles], dtype=torch.bool)
            result.update(statistics(mask, priors[role], role + "_sem_"))
            questions, trajectories, positive, negative = set(), set(), set(), set()
            for row in rows:
                own = self._semantic_mask(row)
                if any(bool(own[i]) for i, r in enumerate(row["prefix_roles"]) if r == role):
                    questions.add(str(row["question_key"]))
                    uid = str(row["traj_uid"])
                    trajectories.add(uid)
                    (positive if row["label"] == 1 else negative).add(uid)
            result.update({role + "_sem_questions": len(questions), role + "_sem_trajectories": len(trajectories),
                           role + "_sem_success_trajectories": len(positive), role + "_sem_failure_trajectories": len(negative)})
        return result

    def _semantic_role_enough(self, metrics, role):
        return (metrics[role + "_sem_prefixes"] >= int(self.config["semantic_min_val_prefixes"])
                and metrics[role + "_sem_questions"] >= int(self.config["semantic_min_val_questions"])
                and metrics[role + "_sem_success_trajectories"] >= int(self.config["semantic_min_val_per_class"])
                and metrics[role + "_sem_failure_trajectories"] >= int(self.config["semantic_min_val_per_class"]))

    def _semantic_role_quality(self, metrics, role):
        auc, brier, prior = (metrics[role + "_sem_" + name] for name in ("auc", "brier", "prior_brier"))
        return (self._semantic_role_enough(metrics, role)
                and all(math.isfinite(x) for x in (auc, brier, prior))
                and auc >= float(self.config["semantic_min_val_auc"])
                and brier < prior - float(self.config["semantic_min_brier_improvement"]))

    def _validate_semantic_deploy(self, train, val, metrics):
        candidate = self._evaluate_semantic(self.candidate_head, val, train)
        metrics.update({"candidate_" + k: v for k, v in candidate.items()})
        current = self._evaluate_semantic(self.deployed_head, val, train) if self.version else None
        if current is not None:
            metrics.update({"deployed_" + k: v for k, v in current.items()})
        passes = {}
        for role in self.semantic_reliability:
            enough = self._semantic_role_enough(candidate, role)
            passes[role] = self._semantic_role_quality(candidate, role)
            reason = "qualified" if passes[role] else "insufficient_data" if not enough else "quality_failed"
            metrics[f"semantic_{role}_qualification"] = reason
            for label in ("qualified", "insufficient_data", "quality_failed"):
                metrics[f"semantic_{role}_qualification_{label}"] = float(reason == label)
            if not enough:
                continue
            ids = sorted(str(r["traj_uid"]) for r in val if any(
                bool(self._semantic_mask(r)[i]) for i, p in enumerate(r["prefix_roles"]) if p == role))
            fingerprint = hashlib.sha256("\0".join(ids).encode()).hexdigest()
            if current is not None and fingerprint != self.semantic_validation_fingerprints[role]:
                self.role_bad_windows[role] = (0 if self._semantic_role_quality(current, role)
                                               else self.role_bad_windows[role] + 1)
                if (self.config["disable_miscalibrated"] and self.semantic_reliability[role]
                        and self.role_bad_windows[role] >= int(self.config["miscalibration_patience"])):
                    self.semantic_reliability[role] = 0.0
                    metrics["disabled"] = 1
            self.semantic_validation_fingerprints[role] = fingerprint
        # One shared deployed head: never overwrite a still-authorized role's
        # predictor with unvalidated/worse weights. An unqualified other role
        # has no veto. Insufficient windows retain the current frozen copy.
        active = [r for r, permission in self.semantic_reliability.items() if permission]
        proposed = [r for r, passed in passes.items() if passed]
        comparator_roles = set(active + proposed)
        improves = current is None or all(
            passes[r] and candidate[r + "_sem_brier"] <= current[r + "_sem_brier"] + float(self.config["candidate_brier_tolerance"])
            for r in comparator_roles if r in active or math.isfinite(current[r + "_sem_brier"]))
        if proposed and improves:
            self.deployed_head.semantic.load_state_dict(self.candidate_head.semantic.state_dict())
            self.deployed_head.eval().requires_grad_(False)
            self.version += 1
            self.semantic_reliability = {r: float(passes[r]) for r in passes}
            self.role_bad_windows = {r: 0 for r in passes}
            metrics.update(status="deployed", deployed=1)
        else:
            metrics["status"] = "candidate_worse" if proposed else "candidate_rejected"
            if active and any(not self._semantic_role_enough(candidate, r) for r in active):
                metrics["status"] = "insufficient_data_retained_deployment"
        self.ready = any(self.semantic_reliability.values())
        self.reliability = {r: 0.0 for r in self.reliability}
        self.temporal_reliability = {r: 0.0 for r in self.temporal_reliability}
        metrics.update({"semantic_bad_windows_" + r: v for r, v in self.role_bad_windows.items()})
        metrics.update(self._deployment_metrics())

    def _ingest(self):
        for uid, row in self._pending.items():
            self._replay[uid] = row
            self._replay.move_to_end(uid)
        self._pending.clear()
        while len(self._replay) > int(self.config["replay_max_trajectories"]):
            self._replay.popitem(last=False)
        train, val = [], []
        for row in self._replay.values():
            (val if self._is_validation(row["question_key"]) else train).append(row)
        return train, val

    def _enough(self, train, val):
        return (len(train) >= int(self.config["min_train_trajectories"])
                and len({r["question_key"] for r in train}) >= int(self.config["min_train_questions"])
                and len(val) >= int(self.config["min_val_trajectories"])
                and len({r["question_key"] for r in val}) >= int(self.config["min_val_questions"])
                and min(sum(r["label"] == 1 for r in val), sum(r["label"] == 0 for r in val)) >= int(self.config["min_val_per_class"]))

    def _validate_deploy(self, train, val, metrics):
        if self.prediction_mode == "semantic_only":
            return self._validate_semantic_deploy(train, val, metrics)
        if self.deployment_mode == "absolute_bootstrap":
            return self._validate_bootstrap_deploy(train, val, metrics)
        flattened = self._flatten(train)
        prior = float(flattened["labels"].mean())
        val_data = self._flatten(val)
        control = self._predict(self.control_head, val_data, no_entropy=True)
        temporal_control = self._predict(self.temporal_control_head, val_data, no_temporal=True)
        candidate = self._evaluate(self.candidate_head, val, prior, control, temporal_control)
        metrics.update({"candidate_" + key: value for key, value in candidate.items()})
        current = self._evaluate(self.deployed_head, val, prior, control, temporal_control) if self.version else None
        enough_entropy = (candidate["entropy_prefixes"] >= int(self.config["min_entropy_val_prefixes"])
                          and candidate["temporal_prefixes"] >= int(self.config["min_entropy_val_prefixes"]))
        fingerprint = hashlib.sha256("\0".join(r["traj_uid"] for r in val).encode()).hexdigest()
        new_validation = fingerprint != self.last_validation_fingerprint
        if current is not None:
            metrics.update({"deployed_" + key: value for key, value in current.items()})
            if enough_entropy and new_validation:
                self.bad_windows = self.bad_windows + 1 if not self._quality_passes(current) else 0
            if (self.ready and bool(self.config["disable_miscalibrated"])
                    and self.bad_windows >= int(self.config["miscalibration_patience"])):
                self.ready = False
                self.reliability = {"solver": 0.0, "verifier": 0.0}
                self.temporal_reliability = {"solver": 0.0, "verifier": 0.0}
                metrics["disabled"] = 1
            # A good Solver aggregate must not indefinitely preserve stale
            # Verifier permissions after the latter has drifted. Only distinct,
            # sufficiently populated windows count; sparse batches retain the
            # previously qualified deployment.
            if self.ready and new_validation and bool(self.config["disable_miscalibrated"]):
                for role in self.reliability:
                    if current[f"{role}_prefixes"] >= int(self.config["min_entropy_val_prefixes"]):
                        self.role_bad_windows[role] = 0 if self._role_quality(current, role) else self.role_bad_windows[role] + 1
                        if self.role_bad_windows[role] >= int(self.config["miscalibration_patience"]):
                            self.reliability[role] = 0.0
                    if current[f"{role}_temporal_prefixes"] >= int(self.config["min_entropy_val_prefixes"]):
                        self.temporal_bad_windows[role] = 0 if self._role_temporal_quality(current, role) else self.temporal_bad_windows[role] + 1
                        if self.temporal_bad_windows[role] >= int(self.config["miscalibration_patience"]):
                            self.temporal_reliability[role] = 0.0
        if enough_entropy:
            self.last_validation_fingerprint = fingerprint
        passes = self._quality_passes(candidate) and any(self._role_quality(candidate, role) for role in self.reliability)
        improves = current is None or candidate["brier"] <= current["brier"] + float(self.config["candidate_brier_tolerance"])
        if passes and improves:
            self.deployed_head.load_state_dict(self.candidate_head.state_dict())
            self.deployed_head.eval().requires_grad_(False)
            self.ready, self.version, self.bad_windows = True, self.version + 1, 0
            for role in self.reliability:
                self.reliability[role] = float(self._role_quality(candidate, role))
                self.temporal_reliability[role] = float(self._role_temporal_quality(candidate, role))
                self.role_bad_windows[role] = self.temporal_bad_windows[role] = 0
            metrics["status"], metrics["deployed"] = "deployed", 1
        else:
            metrics["status"] = "candidate_rejected" if not passes else "candidate_worse"
        metrics.update({"reliability_" + role: score for role, score in self.reliability.items()})
        metrics.update({"temporal_reliability_" + role: score for role, score in self.temporal_reliability.items()})

    def _validate_bootstrap_deploy(self, train, val, metrics):
        """Compare the actual deployed predictor, then grant temporal roles separately.

        Historical two-Solver runs have no nonterminal same-role temporal
        observations. Their terminal S2 features may train semantic/absolute
        prediction but cannot qualify a temporal policy-control permission.
        """
        prior = float(self._flatten(train)["labels"].mean())
        val_data = self._flatten(val)
        control = self._predict(self.control_head, val_data, no_entropy=True)
        temporal_control = self._predict(self.temporal_control_head, val_data, no_temporal=True)
        raw_candidate = self._evaluate(self.candidate_head, val, prior, control, temporal_control)
        proposed_temporal = {
            role: float(not self._absolute_pretrain_active and self._bootstrap_temporal_quality(raw_candidate, role))
            for role in self.reliability
        }
        candidate = self._evaluate(self.candidate_head, val, prior, control, temporal_control,
                                   temporal_permissions=proposed_temporal)
        metrics.update({"candidate_" + key: value for key, value in candidate.items()})
        metrics.update({"candidate_temporal_permission_" + role: value for role, value in proposed_temporal.items()})
        enough_entropy = candidate["entropy_prefixes"] >= int(self.config["min_entropy_val_prefixes"])
        fingerprint = hashlib.sha256("\0".join(r["traj_uid"] for r in val).encode()).hexdigest()
        new_validation = fingerprint != self.last_validation_fingerprint
        current = None
        if self.version:
            current = self._evaluate(self.deployed_head, val, prior, control, temporal_control,
                                     temporal_permissions=self.temporal_reliability)
            old_permissions = self.temporal_reliability.copy()
            if enough_entropy and new_validation:
                self.bad_windows = self.bad_windows + 1 if not self._bootstrap_quality_passes(current) else 0
            if (self.ready and bool(self.config["disable_miscalibrated"])
                    and self.bad_windows >= int(self.config["miscalibration_patience"])):
                self.ready = False
                self.reliability = {role: 0.0 for role in self.reliability}
                self.temporal_reliability = {role: 0.0 for role in self.temporal_reliability}
                metrics["disabled"] = 1
            if self.ready and new_validation and bool(self.config["disable_miscalibrated"]):
                for role in self.reliability:
                    if current[f"{role}_prefixes"] >= int(self.config["min_entropy_val_prefixes"]):
                        self.role_bad_windows[role] = (0 if self._bootstrap_role_quality(current, role)
                                                      else self.role_bad_windows[role] + 1)
                        if self.role_bad_windows[role] >= int(self.config["miscalibration_patience"]):
                            self.reliability[role] = 0.0
                            self.temporal_reliability[role] = 0.0
                    if (old_permissions[role] > 0
                            and current[f"{role}_temporal_prefixes"] >= int(self.config["min_entropy_val_prefixes"])):
                        self.temporal_bad_windows[role] = (0 if self._bootstrap_temporal_quality(current, role)
                                                          else self.temporal_bad_windows[role] + 1)
                        if self.temporal_bad_windows[role] >= int(self.config["miscalibration_patience"]):
                            self.temporal_reliability[role] = 0.0
            if old_permissions != self.temporal_reliability:
                # A revoked residual changes the comparator too: compare the
                # candidate against the fallback that will actually serve.
                current = self._evaluate(self.deployed_head, val, prior, control, temporal_control,
                                         temporal_permissions=self.temporal_reliability)
            metrics.update({"deployed_" + key: value for key, value in current.items()})
        if enough_entropy:
            self.last_validation_fingerprint = fingerprint
        passes = (self._bootstrap_quality_passes(candidate)
                  and any(self._bootstrap_role_quality(candidate, role) for role in self.reliability))
        improves = current is None or candidate["brier"] <= current["brier"] + float(self.config["candidate_brier_tolerance"])
        if passes and improves:
            self.deployed_head.load_state_dict(self.candidate_head.state_dict())
            self.deployed_head.eval().requires_grad_(False)
            self.ready, self.version, self.bad_windows = True, self.version + 1, 0
            for role in self.reliability:
                self.reliability[role] = float(self._bootstrap_role_quality(candidate, role))
                self.temporal_reliability[role] = proposed_temporal[role] * self.reliability[role]
                self.role_bad_windows[role] = self.temporal_bad_windows[role] = 0
            metrics["status"], metrics["deployed"] = "deployed", 1
        else:
            metrics["status"] = "candidate_rejected" if not passes else "candidate_worse"
        metrics.update({"reliability_" + role: score for role, score in self.reliability.items()})
        metrics.update({"temporal_reliability_" + role: score for role, score in self.temporal_reliability.items()})
        metrics.update(self._deployment_metrics())

    def pretrain(self, semantic_epochs=5, entropy_epochs=10, rounds=1):
        """Semantic initialization then frozen-semantic entropy learning.

        Two equally sized controls mask all entropy or only temporal entropy.
        Both receive identical records, stages, losses and epoch budgets.
        Held-out entropy gains are mandatory, not inferred from nonzero grads.
        """
        train, val = self._ingest()
        if not self._enough(train, val):
            raise ValueError("insufficient distinct training/validation questions or outcome classes for pretraining")
        if self.prediction_mode == "semantic_only":
            metrics = {"deployed": 0, "disabled": 0, "train_trajectories": len(train), "val_trajectories": len(val),
                       "semantic_train_loss": self._train(train, semantic_epochs, "semantic"),
                       "semantic_train_prefixes": sum(len(r["features"]) for r in train)}
            self._validate_semantic_deploy(train, val, metrics)
            metrics.update(ready=float(self.ready), version=self.version)
            self.last_metrics = metrics
            return metrics
        self._fit_scaler(train)
        metrics = {"deployed": 0, "disabled": 0, "train_trajectories": len(train), "val_trajectories": len(val)}
        for control in (False, True, "temporal"):
            self._train(train, semantic_epochs, "semantic", control)
        stage = "absolute" if self.config["allow_absolute_only_pretrain"] else "entropy"
        self._absolute_pretrain_active = stage == "absolute"
        try:
            for _ in range(int(rounds)):
                metrics["entropy_train_loss"] = self._train(train, entropy_epochs, stage)
                metrics["control_train_loss"] = self._train(train, entropy_epochs, stage, True)
                metrics["temporal_control_train_loss"] = self._train(train, entropy_epochs, stage, "temporal")
                self._validate_deploy(train, val, metrics)
        finally:
            self._absolute_pretrain_active = False
        self.pretrained = self.ready
        metrics.update(ready=float(self.ready), version=self.version)
        metrics.update(self._deployment_metrics())
        self.last_metrics = metrics
        return metrics

    def _train_paired_online(self, rows, epochs):
        """Share semantic fitting and pair all nuisance randomness in ablations.

        Entropy residuals retain their respective history across online updates;
        their semantic base is always the same newly fitted, frozen network.
        """
        metrics = {"semantic_train_loss": self._train(rows, epochs, "semantic")}
        for head in (self.control_head, self.temporal_control_head):
            head.semantic.load_state_dict(self.candidate_head.semantic.state_dict())
        paired_rng = self._rng.get_state().clone()
        next_rng = None
        for control, name in ((False, "train_loss"), (True, "control_train_loss"),
                              ("temporal", "temporal_control_train_loss")):
            self._rng.set_state(paired_rng)
            metrics[name] = self._train(rows, epochs, "entropy", control)
            if next_rng is None:
                next_rng = self._rng.get_state().clone()
        # All three arms receive identical seeds/permutations. Advance the
        # serialized stream once, independently of the number of controls.
        self._rng.set_state(next_rng)
        return metrics

    def update(self):
        """Train only after the actor batch; insufficient data preserves deployment."""
        started = time.perf_counter()
        self.step += 1
        added = len(self._pending)
        train, val = self._ingest()
        metrics = {"step": self.step, "added_trajectories": added, "train_trajectories": len(train),
                   "val_trajectories": len(val), "deployed": 0, "disabled": 0}
        if self.prediction_mode == "semantic_only":
            enough_train = (len(train) >= int(self.config["min_train_trajectories"])
                            and len({r["question_key"] for r in train}) >= int(self.config["min_train_questions"]))
            if enough_train:
                metrics["semantic_train_loss"] = self._train(train, self.config["train_epochs"], "semantic")
                metrics["semantic_train_prefixes"] = sum(len(r["features"]) for r in train)
                self._validate_semantic_deploy(train, val, metrics)
            else:
                metrics["status"] = "insufficient_data_retained_deployment"
        elif self._enough(train, val):
            if self.scaler is None:
                self._fit_scaler(train)
            if self.training_protocol == _PAIRED_TRAINING_PROTOCOL:
                metrics.update(self._train_paired_online(train, self.config["train_epochs"]))
            else:
                metrics["train_loss"] = self._train(train, self.config["train_epochs"])
                metrics["control_train_loss"] = self._train(train, self.config["train_epochs"], control=True)
                metrics["temporal_control_train_loss"] = self._train(train, self.config["train_epochs"], control="temporal")
            self._validate_deploy(train, val, metrics)
        else:
            metrics["status"] = "insufficient_data_retained_deployment"
        metrics.update(ready=float(self.ready), version=self.version, bad_windows=self.bad_windows,
                       update_seconds=time.perf_counter() - started)
        metrics.update(self._deployment_metrics())
        self.last_metrics = metrics
        return metrics

    def save(self, path):
        target = Path(path)
        if target.is_dir():
            raise ValueError("checkpoint path must be a file")
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {"checkpoint_version": _CHECKPOINT_VERSION, "config": copy.deepcopy(self.config),
                 "prediction_mode": self.prediction_mode, "replay_feature_storage": "raw",
                 "semantic_reliability": self.semantic_reliability,
                 "semantic_validation_fingerprints": self.semantic_validation_fingerprints,
                 "startup_semantic_report": self.startup_semantic_report,
                 "encoder_identity": self.encoder_identity, "schema": FEATURE_SCHEMA,
                 "scaler": copy.deepcopy(self.scaler), "ready": self.ready, "pretrained": self.pretrained,
                  "deployment_mode": self.deployment_mode, "pretraining_mode": self.pretraining_mode,
                  "training_protocol": self.training_protocol, "warm_started": self.warm_started,
                 "deployment_stage": self._deployment_stage(),
                 "version": self.version, "step": self.step, "bad_windows": self.bad_windows,
                 "role_bad_windows": self.role_bad_windows, "temporal_bad_windows": self.temporal_bad_windows,
                 "last_validation_fingerprint": self.last_validation_fingerprint,
                 "reliability": self.reliability, "temporal_reliability": self.temporal_reliability, "last_metrics": self.last_metrics,
                 "rng_state": self._rng.get_state(), "replay": list(self._replay.items()),
                 "pending": list(self._pending.items())}
        for name in ("candidate_head", "deployed_head", "control_head", "temporal_control_head",
                     "optimizer", "control_optimizer", "temporal_control_optimizer"):
            state[name] = _cpu_copy(getattr(self, name).state_dict())
        temporary = target.with_name(target.name + ".tmp")
        torch.save(_cpu_copy(state), temporary)
        os.replace(temporary, target)

    def _migrate_semantic_cache(self, state):
        """Read only raw action metadata from the known legacy cache schema."""
        rows, missing, invalid = [], 0, 0
        migrated = zero_tokens = invalid_actions = 0
        seen = set()
        raw = state.get("replay_feature_storage", "raw") == "raw" and state.get("schema") == FEATURE_SCHEMA
        for uid, source in state.get("replay", []):
            row = dict(source)
            features = row.get("features")
            roles, terminal = row.get("prefix_roles", []), row.get("prefix_terminal")
            if (not torch.is_tensor(features) or features.ndim != 2 or features.shape[1] != self.feature_dim
                    or not len(features) or not bool(torch.isfinite(features).all())
                    or not torch.is_tensor(terminal) or terminal.dtype != torch.bool or terminal.shape != (len(features),)
                    or len(roles) != len(features) or roles[0] != "initial"
                    or any(r not in ("solver", "verifier") for r in roles[1:])
                    or row.get("label") not in (0.0, 1.0) or not row.get("question_key")
                    or str(row.get("traj_uid", uid)) != str(uid) or str(uid) in seen):
                invalid += 1
                continue
            seen.add(str(uid))
            row["traj_uid"] = str(uid)
            own = row.get("semantic_available")
            explicit = (row.get("semantic_metadata_schema") == _SEMANTIC_METADATA_SCHEMA
                        and torch.is_tensor(own) and own.dtype == torch.bool and own.shape == (len(features),))
            if not explicit:
                own = torch.zeros(len(features), dtype=torch.bool)
                absolute = row.get("absolute")
                if raw and torch.is_tensor(absolute) and absolute.shape == (len(features), len(ABS_FEATURE_NAMES)):
                    own[0] = True
                    for i, role in enumerate(roles[1:], 1):
                        offset = 0 if role == "solver" else 6
                        # Do not inspect mean, coverage or entropy availability.
                        log_tokens, valid, truncated = absolute[i, offset + 2:offset + 5].tolist()
                        if (math.isfinite(log_tokens) and log_tokens >= 0 and valid in (0.0, 1.0)
                                and truncated in (0.0, 1.0)):
                            own[i] = log_tokens > 0 and valid == 1.0 and truncated == 0.0
                            migrated += 1
                            zero_tokens += int(log_tokens == 0)
                            invalid_actions += int(valid == 0.0 or truncated == 1.0)
                        else:
                            missing += 1
                else:
                    missing += max(0, len(features) - 1)
                row["semantic_available"] = own
                row["semantic_metadata_schema"] = _SEMANTIC_METADATA_SCHEMA
            rows.append(row)
        return rows, {"cache_trajectories": len(rows), "cache_invalid_trajectories": invalid,
                      "cache_metadata_unrecoverable_prefixes": missing,
                      "cache_raw_metadata_migrated_prefixes": migrated,
                      "cache_zero_or_missing_response_tokens_prefixes": zero_tokens,
                      "cache_invalid_or_truncated_prefixes": invalid_actions}

    def _semantic_load_result(self):
        return {"ready": self.ready, "semantic_ready": self.ready, "version": self.version,
                "pretrained": self.pretrained, "warm_started": self.warm_started,
                "prediction_mode": self.prediction_mode, "deployment_stage": self._deployment_stage(),
                "pretraining_mode": self.pretraining_mode,
                "reliability": self.reliability.copy(), "temporal_reliability": self.temporal_reliability.copy(),
                "semantic_reliability": self.semantic_reliability.copy(),
                "metrics": copy.deepcopy(self.last_metrics),
                "startup_semantic_report": copy.deepcopy(self.startup_semantic_report)}

    def _load_semantic(self, state, *, resume, warm_start):
        if state["config"]["head_hidden_dim"] != self.config["head_hidden_dim"]:
            raise ValueError("checkpoint semantic head_hidden_dim does not match")
        if resume:
            for key in _DEFAULTS.keys() - {"device", "cpu_num_threads", "allow_absolute_only_pretrain"}:
                if state["config"].get(key, _DEFAULTS[key]) != self.config[key]:
                    raise ValueError(f"resume configuration mismatch: {key}")
        if warm_start:
            # Import only the trained candidate semantic branch. In particular,
            # a failed offline deployed_head may be entirely untrained.
            weights = {key.removeprefix("semantic."): value for key, value in state["candidate_head"].items()
                       if key.startswith("semantic.")}
            self.candidate_head.semantic.load_state_dict(weights, strict=True)
            self._configure_semantic_training()
            self.scaler = None
            self.ready = self.pretrained = False
            self.warm_started = True
            self.version = self.step = self.bad_windows = 0
            self.reliability = {"solver": 0.0, "verifier": 0.0}
            self.temporal_reliability = {"solver": 0.0, "verifier": 0.0}
            self.semantic_reliability = {"solver": 0.0, "verifier": 0.0}
            self.role_bad_windows = {"solver": 0, "verifier": 0}
            self.temporal_bad_windows = {"solver": 0, "verifier": 0}
            self.semantic_validation_fingerprints = {"solver": None, "verifier": None}
            self.last_validation_fingerprint = None
            self._pending, self._replay = OrderedDict(), OrderedDict()
            self._rng = torch.Generator(device="cpu").manual_seed(int(self.config["seed"]))
            self.pretraining_mode = state.get("pretraining_mode", "full")
            rows, report = self._migrate_semantic_cache(state)
            source_salt = state["config"].get("holdout_salt", _DEFAULTS["holdout_salt"])
            source_fraction = float(state["config"].get("validation_fraction", _DEFAULTS["validation_fraction"]))
            if not 0 < source_fraction < 1:
                raise ValueError("source checkpoint validation_fraction is invalid")
            train, val = [], []
            for row in rows:
                digest = hashlib.sha256((str(source_salt) + "\0" + str(row["question_key"])).encode()).digest()
                (val if int.from_bytes(digest[:8], "big") / (1 << 64) < source_fraction else train).append(row)
            report.update(train_trajectories=len(train), val_trajectories=len(val),
                          deployed=0, disabled=0, qualification_source="source_checkpoint_replay",
                          source_holdout_salt=str(source_salt), source_validation_fraction=source_fraction)
            # No optimizer step and no encoder call on any source trajectory.
            self._validate_semantic_deploy(train, val, report)
            report.update(ready=float(self.ready), version=self.version)
            self.startup_semantic_report = copy.deepcopy(report)
            self.last_metrics = copy.deepcopy(report)
            # Local references to the historical replay die on return. Only new
            # representative rollouts will populate the online run's cache.
            return self._semantic_load_result()

        if state.get("training_protocol") != _SEMANTIC_TRAINING_PROTOCOL or state.get("deployment_mode") != "semantic_only":
            raise ValueError("checkpoint lacks the semantic training/deployment protocol")
        if state.get("deployment_stage") != "semantic_only":
            raise ValueError("checkpoint semantic deployment stage mismatch")
        permissions = state.get("semantic_reliability", {})
        if set(permissions) != {"solver", "verifier"} or any(
                not math.isfinite(float(v)) or float(v) not in (0.0, 1.0) for v in permissions.values()):
            raise ValueError("checkpoint semantic role permissions are invalid")
        if bool(state["ready"]) != any(permissions.values()) or (state["ready"] and int(state["version"]) < 1):
            raise ValueError("checkpoint ready/version disagree with semantic permissions")
        if any(state.get(name, {}).get(role, 0.0) != 0 for name in ("reliability", "temporal_reliability")
               for role in permissions):
            raise ValueError("semantic checkpoint cannot claim entropy permissions")
        if not resume and not state.get("ready"):
            raise ValueError("initialization requires a semantic-qualified checkpoint, or explicit candidate mode")
        self._configure_semantic_training()
        for name in ("candidate_head", "deployed_head", "control_head", "temporal_control_head"):
            getattr(self, name).load_state_dict(state[name])
        if not resume:
            self.candidate_head.semantic.load_state_dict(self.deployed_head.semantic.state_dict())
        self.scaler = None
        self.ready, self.version = bool(state["ready"]), int(state["version"])
        self.pretrained = False  # semantic qualification is never entropy pretraining qualification
        self.warm_started = bool(state.get("warm_started", False))
        self.pretraining_mode = state.get("pretraining_mode", "full")
        self.semantic_reliability = dict(permissions)
        self.reliability = {r: 0.0 for r in permissions}
        self.temporal_reliability = {r: 0.0 for r in permissions}
        self.startup_semantic_report = copy.deepcopy(state.get("startup_semantic_report", {}))
        self.last_metrics = copy.deepcopy(state["last_metrics"])
        if resume:
            for name in ("optimizer", "control_optimizer", "temporal_control_optimizer"):
                optimizer = getattr(self, name)
                optimizer.load_state_dict(state[name])
                for value in optimizer.state.values():
                    for key, item in value.items():
                        if torch.is_tensor(item):
                            value[key] = item.to(self.device)
            self._replay, self._pending = OrderedDict(state["replay"]), OrderedDict(state["pending"])
            self._rng.set_state(state["rng_state"])
            self.step, self.bad_windows = int(state["step"]), int(state["bad_windows"])
            self.role_bad_windows = dict(state["role_bad_windows"])
            self.temporal_bad_windows = dict(state["temporal_bad_windows"])
            self.semantic_validation_fingerprints = dict(state["semantic_validation_fingerprints"])
            self.last_validation_fingerprint = state.get("last_validation_fingerprint")
        return self._semantic_load_result()

    def load(self, path, *, resume=False, warm_start=False):
        """Load a trusted local checkpoint; initialization keeps runtime policy.

        Old sup single-head checkpoints are intentionally incompatible. Fresh
        main training loads validated weights/scaler, without offline replay or
        optimizer momentum. Exact run resume restores their state. Explicit
        warm-start imports a candidate as a new, unqualified online learner;
        it never presents that candidate as a validated deployed predictor.
        """
        if not isinstance(resume, bool) or not isinstance(warm_start, bool):
            raise ValueError("resume and warm_start must be boolean")
        if resume and warm_start:
            raise ValueError("warm_start initializes a new run and cannot be combined with resume")
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        if state.get("checkpoint_version") != _CHECKPOINT_VERSION or state.get("schema") != FEATURE_SCHEMA:
            raise ValueError("checkpoint is not an entropy-aware v2 prefix-value checkpoint")
        if state.get("encoder_identity") != self.encoder_identity:
            raise ValueError("checkpoint fixed encoder identity/serialization does not match")
        source_mode = state.get("prediction_mode", "entropy_aware")
        if source_mode not in ("entropy_aware", "semantic_only"):
            raise ValueError("checkpoint prediction_mode is invalid")
        if state["config"].get("prediction_mode", "entropy_aware") != source_mode:
            raise ValueError("checkpoint prediction_mode disagrees with configuration")
        if source_mode != self.prediction_mode and not warm_start:
            raise ValueError("prediction_mode mismatch: cross-mode resume/qualified load is forbidden; initialize a new candidate run")
        if self.prediction_mode == "semantic_only":
            return self._load_semantic(state, resume=resume, warm_start=warm_start)
        if source_mode == "semantic_only":
            raise ValueError("semantic-only weights have no entropy qualification; use an entropy-aware checkpoint")
        for key in ("head_hidden_dim", "entropy_hidden_dim", "holdout_salt", "validation_fraction", "min_entropy_coverage"):
            if state["config"][key] != self.config[key]:
                raise ValueError(f"checkpoint {key} does not match runtime configuration")
        for key in ("scaler_clip", "scaler_floor"):
            if state["config"][key] != self.config[key]:
                raise ValueError(f"checkpoint {key} does not match fixed entropy scaling")
        if resume:
            # Moving the scorer between devices/CPU thread counts is allowed;
            # a continuation must preserve every numerical/training setting.
            for key in _DEFAULTS.keys() - {"device", "cpu_num_threads", "allow_absolute_only_pretrain"}:
                if state["config"].get(key, _DEFAULTS[key]) != self.config[key]:
                    raise ValueError(f"resume configuration mismatch: {key}")
        deployment_mode = state.get("deployment_mode", "full")
        pretraining_mode = state.get("pretraining_mode", "full")
        training_protocol = state.get("training_protocol", "legacy")
        if training_protocol not in ("legacy", _PAIRED_TRAINING_PROTOCOL):
            raise ValueError("checkpoint contains an unsupported value training protocol")
        if not isinstance(state.get("warm_started", False), bool):
            raise ValueError("checkpoint warm_started must be boolean")
        if deployment_mode not in ("full", "absolute_bootstrap") or pretraining_mode not in ("full", "absolute"):
            raise ValueError("checkpoint contains an invalid value deployment mode")
        if ((state["config"].get("allow_absolute_only_pretrain", False) or pretraining_mode == "absolute")
                and "deployment_mode" not in state):
            raise ValueError("checkpoint absolute bootstrap deployment mode is missing")
        for name in ("reliability", "temporal_reliability"):
            permissions = state.get(name, {})
            if set(permissions) != {"solver", "verifier"} or any(
                    not math.isfinite(float(v)) or float(v) not in (0.0, 1.0) for v in permissions.values()):
                raise ValueError("checkpoint role permissions are invalid")
        if deployment_mode == "absolute_bootstrap":
            if any(float(state["temporal_reliability"][role]) > float(state["reliability"][role])
                   for role in ("solver", "verifier")):
                raise ValueError("checkpoint temporal permission requires its role's absolute permission")
            enabled = sum(float(v) > 0 for v in state["temporal_reliability"].values())
            expected_stage = "full" if enabled == 2 else "hybrid" if enabled else "absolute"
            if state.get("deployment_stage") != expected_stage:
                raise ValueError("checkpoint deployment stage disagrees with temporal permissions")
        scaler = state.get("scaler")
        if not isinstance(scaler, dict) or scaler.get("schema") != FEATURE_SCHEMA:
            raise ValueError("checkpoint entropy scaler/schema missing")
        if not math.isfinite(float(scaler.get("clip", float("nan")))) or float(scaler["clip"]) != float(self.config["scaler_clip"]):
            raise ValueError("checkpoint scaler clipping does not match fixed entropy scaling")
        for key, names in (("absolute", ABS_FEATURE_NAMES), ("temporal", TEMP_FEATURE_NAMES)):
            spec = scaler.get(key, {})
            if spec.get("names") != list(names) or len(spec.get("center", [])) != len(names) or len(spec.get("scale", [])) != len(names):
                raise ValueError("checkpoint scaler feature order/dimension mismatch")
            if not all(math.isfinite(float(x)) for x in spec["center"] + spec["scale"]) or min(spec["scale"]) <= 0:
                raise ValueError("checkpoint scaler contains invalid values")
        if not resume and not warm_start and (not state.get("ready") or not state.get("pretrained")):
            raise ValueError("initialization requires an entropy-qualified pretrained checkpoint")
        self.scaler = copy.deepcopy(scaler)
        if warm_start:
            # A version-zero failed offline checkpoint has a trained candidate
            # but an untrained deployed head. Never initialize from the latter.
            self.candidate_head.load_state_dict(state["candidate_head"])
            self.control_head.load_state_dict(state["candidate_head"])
            self.temporal_control_head.load_state_dict(state["candidate_head"])
            self.candidate_head.eval().requires_grad_(True)
            self.control_head.eval().requires_grad_(True)
            self.temporal_control_head.eval().requires_grad_(True)
            self.deployed_head.eval().requires_grad_(False)
            self.encoder.eval().requires_grad_(False)
            self.training_protocol = _PAIRED_TRAINING_PROTOCOL
            self.warm_started = True
            self.deployment_mode = "absolute_bootstrap"
            self.pretraining_mode = pretraining_mode
            self.ready = self.pretrained = False
            self.version = self.step = self.bad_windows = 0
            self.reliability = {"solver": 0.0, "verifier": 0.0}
            self.temporal_reliability = {"solver": 0.0, "verifier": 0.0}
            self.role_bad_windows = {"solver": 0, "verifier": 0}
            self.temporal_bad_windows = {"solver": 0, "verifier": 0}
            self.last_validation_fingerprint = None
            self._absolute_pretrain_active = False
            self._pending, self._replay = OrderedDict(), OrderedDict()
            self._rng = torch.Generator(device="cpu").manual_seed(int(self.config["seed"]))
            for name, head in (("optimizer", self.candidate_head), ("control_optimizer", self.control_head),
                               ("temporal_control_optimizer", self.temporal_control_head)):
                setattr(self, name, torch.optim.AdamW(head.parameters(), lr=float(self.config["learning_rate"]),
                                                     weight_decay=float(self.config["weight_decay"])))
            self.last_metrics = {**self._deployment_metrics(), "ready": 0.0, "version": 0}
            return {"ready": False, "pretrained": False, "version": 0, "warm_started": True,
                    "reliability": self.reliability.copy(), "temporal_reliability": self.temporal_reliability.copy(),
                    "deployment_stage": self._deployment_stage(), "pretraining_mode": self.pretraining_mode}
        self.deployed_head.load_state_dict(state["deployed_head"])
        self.candidate_head.load_state_dict(state["candidate_head"] if resume else state["deployed_head"])
        self.control_head.load_state_dict(state["control_head"])
        self.temporal_control_head.load_state_dict(state["temporal_control_head"])
        self.candidate_head.eval().requires_grad_(True)
        self.control_head.eval().requires_grad_(True)
        self.temporal_control_head.eval().requires_grad_(True)
        self.deployed_head.eval().requires_grad_(False)
        self.encoder.eval().requires_grad_(False)
        self.ready, self.pretrained = bool(state["ready"]), bool(state.get("pretrained"))
        self.deployment_mode, self.pretraining_mode = deployment_mode, pretraining_mode
        self.training_protocol, self.warm_started = training_protocol, state.get("warm_started", False)
        self.version = int(state["version"])
        self.reliability = dict(state["reliability"])
        self.temporal_reliability = dict(state["temporal_reliability"])
        self.last_metrics = copy.deepcopy(state["last_metrics"])
        if resume:
            for name in ("optimizer", "control_optimizer", "temporal_control_optimizer"):
                optimizer = getattr(self, name)
                optimizer.load_state_dict(state[name])
                for value in optimizer.state.values():
                    for key, item in value.items():
                        if torch.is_tensor(item):
                            value[key] = item.to(self.device)
            self._replay, self._pending = OrderedDict(state["replay"]), OrderedDict(state["pending"])
            self._rng.set_state(state["rng_state"])
            self.step, self.bad_windows = int(state["step"]), int(state["bad_windows"])
            self.role_bad_windows = dict(state.get("role_bad_windows", {"solver": 0, "verifier": 0}))
            self.temporal_bad_windows = dict(state.get("temporal_bad_windows", {"solver": 0, "verifier": 0}))
            self.last_validation_fingerprint = state.get("last_validation_fingerprint")
        return {"ready": self.ready, "version": self.version, "warm_started": self.warm_started,
                "reliability": self.reliability.copy(),
                "temporal_reliability": self.temporal_reliability.copy(),
                "deployment_stage": self._deployment_stage(), "pretraining_mode": self.pretraining_mode}

