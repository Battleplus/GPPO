from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .gppo_v2 import implementation_hash
from .paper_models import PaperHeteroActorCritic
from .pcrl_v0 import (
    N_OBJECTIVES,
    map_task_preference_to_full,
    task_preference_profile,
)


LEGACY_PREFERENCE_INPUT_MODE = "mapped_shared_v1"
SPLIT_PREFERENCE_INPUT_MODE = "split_task_objective_v2"
MASS_CONTROL_PREFERENCE_INPUT_MODE = "split_task_mass_v3"
PREFERENCE_INPUT_MODES = (
    LEGACY_PREFERENCE_INPUT_MODE,
    SPLIT_PREFERENCE_INPUT_MODE,
    MASS_CONTROL_PREFERENCE_INPUT_MODE,
)


FROZEN_GPPO_IMPLEMENTATION_HASH = (
    "a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6"
)


def verify_frozen_checkpoint_file(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).resolve()
    workspace = Path(__file__).resolve().parents[2]
    manifest_path = workspace / "configs" / "pcrl_v0_frozen_checkpoints.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        relative = path.relative_to(workspace).as_posix()
    except ValueError as error:
        raise ValueError("formal GPPO source checkpoint must be inside the workspace") from error
    entries = {
        str(entry["path"]): entry for entry in manifest["checkpoints"]
    }
    if relative not in entries:
        raise ValueError(f"checkpoint is not present in the frozen PCRL manifest: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = entries[relative]
    if digest != entry["sha256"]:
        raise ValueError("frozen GPPO checkpoint SHA-256 mismatch")
    return dict(entry)


def _zero_output_layer(module: nn.Sequential) -> None:
    output = module[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("residual head must end in nn.Linear")
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)


class PreferenceConditionedPaperActorCritic(nn.Module):
    """Preference-conditioned residual policy initialized from frozen GPPO-v2.

    The original GPPO modules remain under ``backbone``.  Zero-initialized
    residual actor heads make the initial action distribution exactly equal to
    the loaded GPPO checkpoint, while the critic exposes seven objective values.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        max_uavs: int,
        max_tasks: int,
        *,
        hidden_dim: int = 64,
        graph_mode: str = "adaptive",
        n_objectives: int = N_OBJECTIVES,
        preference_conditioning: bool = True,
        preference_input_mode: str = LEGACY_PREFERENCE_INPUT_MODE,
    ):
        super().__init__()
        if n_objectives != N_OBJECTIVES:
            raise ValueError(f"PCRL-v0 requires {N_OBJECTIVES} objectives")
        self.max_uavs = max_uavs
        self.max_tasks = max_tasks
        self.hidden_dim = hidden_dim
        self.graph_mode = "single_head" if graph_mode == "staged" else graph_mode
        self.n_objectives = n_objectives
        self.preference_conditioning = preference_conditioning
        if preference_input_mode not in PREFERENCE_INPUT_MODES:
            raise ValueError(
                "preference_input_mode must be one of "
                f"{PREFERENCE_INPUT_MODES!r}"
            )
        self.preference_input_mode = preference_input_mode
        self.backbone = PaperHeteroActorCritic(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            max_uavs=max_uavs,
            max_tasks=max_tasks,
            hidden_dim=hidden_dim,
            graph_mode=graph_mode,
        )
        if preference_input_mode == LEGACY_PREFERENCE_INPUT_MODE:
            self.preference_encoder = nn.Sequential(
                nn.Linear(n_objectives + 4, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
        else:
            self.actor_preference_encoder = nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.critic_preference_encoder = nn.Sequential(
                nn.Linear(n_objectives, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            if preference_input_mode == MASS_CONTROL_PREFERENCE_INPUT_MODE:
                self.task_mass_calibration = nn.Sequential(
                    nn.Linear(12, hidden_dim // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                _zero_output_layer(self.task_mass_calibration)
        self.pair_preference_residual = nn.Sequential(
            nn.Linear(3 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.noop_preference_residual = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.vector_critic_residual = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objectives),
        )
        # A direct, zero-initialized task-type control path makes the intended
        # semantics identifiable even before the residual MLP has specialized.
        # In v1/v2 this is an unconstrained additive gain. In v3 it is the
        # unconstrained parameter of a positive softplus task-mass gain.
        self.task_preference_gain = nn.Parameter(torch.tensor(0.0))
        _zero_output_layer(self.pair_preference_residual)
        _zero_output_layer(self.noop_preference_residual)
        _zero_output_layer(self.vector_critic_residual)
        balanced = torch.as_tensor(
            map_task_preference_to_full(task_preference_profile("balanced")),
            dtype=torch.float32,
        )
        self.register_buffer("default_preference", balanced)
        if preference_input_mode != LEGACY_PREFERENCE_INPUT_MODE:
            self.register_buffer(
                "default_task_preference",
                torch.full((4,), 0.25, dtype=torch.float32),
            )
        self.source_checkpoint_metadata: dict[str, Any] | None = None

    @staticmethod
    def task_mass_target(
        task_preference: torch.Tensor,
        preference_deficit: torch.Tensor,
        legal_task_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Return the desired next-action mass over currently legal task types.

        The cumulative deficit closes the loop around the episode-level
        assignment mix. Illegal types receive zero target mass and the result
        is normalized only over task types that can be executed now.
        """

        if not (
            task_preference.shape
            == preference_deficit.shape
            == legal_task_counts.shape
        ) or task_preference.shape[-1] != 4:
            raise ValueError("task-mass inputs must have matching [batch, 4] shapes")
        legal_types = legal_task_counts > 0
        raw_target = 1e-3 * task_preference + torch.relu(preference_deficit)
        target = raw_target * legal_types.to(raw_target.dtype)
        target_sum = target.sum(dim=-1, keepdim=True)
        legal_uniform = legal_types.to(raw_target.dtype)
        legal_uniform = legal_uniform / legal_uniform.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        return torch.where(
            target_sum > 0,
            target / target_sum.clamp_min(1e-8),
            legal_uniform,
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)

    def initialize_from_gppo_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        require_event_sync: bool = True,
        require_frozen_hash: bool = True,
        require_frozen_file: bool = True,
    ) -> dict[str, Any]:
        path = Path(checkpoint_path)
        frozen_entry = (
            verify_frozen_checkpoint_file(path) if require_frozen_file else None
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("version") != "paper-aligned-gppo-v2":
            raise ValueError("PCRL initialization requires a GPPO-v2 checkpoint")
        checkpoint_graph_mode = str(payload.get("graph_mode"))
        if checkpoint_graph_mode == "staged":
            checkpoint_graph_mode = "single_head"
        if checkpoint_graph_mode != self.graph_mode:
            raise ValueError(
                f"checkpoint graph_mode={checkpoint_graph_mode!r} does not match "
                f"model graph_mode={self.graph_mode!r}"
            )
        if require_event_sync and payload.get("sync_mode") != "event":
            raise ValueError("PCRL-v0 must initialize from an event-sync GPPO baseline")
        checkpoint_hash = str(payload.get("implementation_hash", ""))
        if require_frozen_hash:
            current_hash = implementation_hash()
            if checkpoint_hash != FROZEN_GPPO_IMPLEMENTATION_HASH:
                raise ValueError("checkpoint does not use the frozen accepted GPPO hash")
            if current_hash != FROZEN_GPPO_IMPLEMENTATION_HASH:
                raise ValueError("frozen GPPO implementation files have changed")
        if frozen_entry is not None:
            if payload.get("method_id") != frozen_entry["method"]:
                raise ValueError("checkpoint method does not match the frozen manifest")
            if int(payload.get("training", {}).get("seed", -1)) != int(
                frozen_entry["seed"]
            ):
                raise ValueError("checkpoint seed does not match the frozen manifest")
        self.backbone.load_state_dict(payload["model_state"], strict=True)
        metadata = {
            "path": str(path.resolve()),
            "method_id": str(payload.get("method_id", "")),
            "graph_mode": checkpoint_graph_mode,
            "sync_mode": str(payload.get("sync_mode", "")),
            "training_seed": int(payload.get("training", {}).get("seed", -1)),
            "best_update": int(payload.get("best_update", -1)),
            "implementation_hash": checkpoint_hash,
            "scenario_hash": str(payload.get("scenario_hash", "")),
            "scenario": str(payload.get("scenario", "")),
            "checkpoint_sha256": (
                str(frozen_entry["sha256"]) if frozen_entry is not None else "unverified"
            ),
        }
        self.source_checkpoint_metadata = metadata
        return metadata

    def _prepare_preference(
        self, preference: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        preference = preference.to(device=device, dtype=torch.float32)
        if preference.ndim == 1:
            preference = preference.unsqueeze(0)
        if preference.shape[-1] != self.n_objectives:
            raise ValueError(
                f"expected {self.n_objectives} preference weights, got "
                f"{preference.shape[-1]}"
            )
        if preference.shape[0] == 1 and batch_size > 1:
            preference = preference.expand(batch_size, -1)
        if preference.shape[0] != batch_size:
            raise ValueError("preference batch does not match observation batch")
        preference = preference.clamp_min(0.0)
        preference = preference / preference.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if not self.preference_conditioning:
            preference = self.default_preference.to(device).expand(batch_size, -1)
        return preference

    def _prepare_task_preference(
        self,
        task_preference: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if task_preference is None:
            if self.preference_input_mode != LEGACY_PREFERENCE_INPUT_MODE:
                raise ValueError(
                    "split preference input requires a raw four-dimensional "
                    "task_preference"
                )
            task_preference = torch.full(
                (batch_size, 4), 0.25, dtype=torch.float32, device=device
            )
        else:
            task_preference = task_preference.to(
                device=device, dtype=torch.float32
            )
            if task_preference.ndim == 1:
                task_preference = task_preference.unsqueeze(0)
            if task_preference.shape[-1] != 4:
                raise ValueError(
                    "task_preference must have four task weights"
                )
            if task_preference.shape[0] == 1 and batch_size > 1:
                task_preference = task_preference.expand(batch_size, -1)
            if task_preference.shape[0] != batch_size:
                raise ValueError(
                    "task_preference batch does not match observation batch"
                )
            task_preference = task_preference.clamp_min(0.0)
            task_preference = task_preference / task_preference.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        if not self.preference_conditioning:
            task_preference = torch.full_like(task_preference, 0.25)
        return task_preference

    def forward(
        self,
        nodes: torch.Tensor,
        edge_types: torch.Tensor,
        edge_features: torch.Tensor,
        action_mask: torch.Tensor,
        preference: torch.Tensor,
        preference_deficit: torch.Tensor | None = None,
        task_preference: torch.Tensor | None = None,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        if nodes.ndim == 2:
            nodes = nodes.unsqueeze(0)
            edge_types = edge_types.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        preference = self._prepare_preference(
            preference, nodes.shape[0], nodes.device
        )
        task_preference = self._prepare_task_preference(
            task_preference, nodes.shape[0], nodes.device
        )
        if preference_deficit is None:
            preference_deficit = torch.zeros(
                nodes.shape[0], 4, dtype=nodes.dtype, device=nodes.device
            )
        else:
            preference_deficit = preference_deficit.to(
                device=nodes.device, dtype=nodes.dtype
            )
            if preference_deficit.ndim == 1:
                preference_deficit = preference_deficit.unsqueeze(0)
            if preference_deficit.shape[0] == 1 and nodes.shape[0] > 1:
                preference_deficit = preference_deficit.expand(nodes.shape[0], -1)
            if preference_deficit.shape != (nodes.shape[0], 4):
                raise ValueError("preference_deficit must have shape [batch, 4]")
        if not self.preference_conditioning:
            preference_deficit = torch.zeros_like(preference_deficit)
        encoded = self.backbone._encode(nodes, edge_types, edge_features)
        uav = encoded[:, : self.max_uavs]
        task = encoded[:, self.max_uavs : self.max_uavs + self.max_tasks]
        pair_uav = uav.unsqueeze(2).expand(-1, -1, self.max_tasks, -1)
        pair_task = task.unsqueeze(1).expand(-1, self.max_uavs, -1, -1)
        pair_edges = edge_features[
            :, : self.max_uavs, self.max_uavs : self.max_uavs + self.max_tasks
        ]
        base_pair = torch.cat((pair_uav, pair_task, pair_edges), dim=-1)
        base_pair_logits = self.backbone.pair_actor(base_pair)

        active_nodes = nodes[..., 2].unsqueeze(-1)
        pooled = torch.sum(encoded * active_nodes, dim=1) / active_nodes.sum(
            dim=1
        ).clamp_min(1.0)
        if self.preference_input_mode == LEGACY_PREFERENCE_INPUT_MODE:
            actor_preference_embedding = self.preference_encoder(
                torch.cat((preference, preference_deficit), dim=-1)
            )
            critic_preference_embedding = actor_preference_embedding
            task_control_preference = preference[:, :4]
            balanced_task = self.default_preference[:4].to(nodes.device)
        else:
            actor_preference_embedding = self.actor_preference_encoder(
                torch.cat((task_preference, preference_deficit), dim=-1)
            )
            critic_preference_embedding = self.critic_preference_encoder(
                preference
            )
            task_control_preference = task_preference
            balanced_task = self.default_task_preference.to(nodes.device)
        pair_preference = actor_preference_embedding[:, None, None, :].expand(
            -1, self.max_uavs, self.max_tasks, -1
        )
        residual_pair_input = torch.cat(
            (pair_uav, pair_task, pair_edges, pair_preference), dim=-1
        )
        pair_logits = (
            base_pair_logits + self.pair_preference_residual(residual_pair_input)
        ).reshape(nodes.shape[0], -1)
        task_type_one_hot = nodes[
            :, self.max_uavs : self.max_uavs + self.max_tasks, 16:20
        ]
        base_noop = self.backbone.noop_actor(pooled)
        noop_logits = base_noop + self.noop_preference_residual(
            torch.cat((pooled, actor_preference_embedding), dim=-1)
        )
        if (
            self.preference_input_mode == MASS_CONTROL_PREFERENCE_INPUT_MODE
            and self.preference_conditioning
        ):
            pair_mask = action_mask[:, :-1].reshape(
                -1, self.max_uavs, self.max_tasks
            )
            legal_task_counts = torch.einsum(
                "but,bti->bi", pair_mask.to(nodes.dtype), task_type_one_hot
            )
            desired_mass = self.task_mass_target(
                task_control_preference,
                preference_deficit,
                legal_task_counts,
            )
            pre_calibration_logits = torch.cat((pair_logits, noop_logits), dim=-1)
            pre_calibration_logits = pre_calibration_logits.masked_fill(
                ~action_mask.bool(), -1e9
            )
            pre_calibration_probs = torch.softmax(pre_calibration_logits, dim=-1)
            current_task_mass = torch.einsum(
                "but,bti->bi",
                pre_calibration_probs[:, :-1].reshape(
                    -1, self.max_uavs, self.max_tasks
                ),
                task_type_one_hot,
            )
            current_group_mass = torch.cat(
                (current_task_mass, pre_calibration_probs[:, -1:]), dim=-1
            )

            # Preserve the frozen GPPO assignment/noop rhythm. Preference
            # control only redistributes the assignment mass across task types.
            frozen_logits = torch.cat(
                (base_pair_logits.reshape(nodes.shape[0], -1), base_noop), dim=-1
            ).masked_fill(~action_mask.bool(), -1e9)
            frozen_pair_mass = torch.softmax(frozen_logits, dim=-1)[
                :, :-1
            ].sum(dim=-1, keepdim=True).detach()
            target_group_mass = torch.cat(
                (
                    frozen_pair_mass * desired_mass,
                    1.0 - frozen_pair_mass,
                ),
                dim=-1,
            )
            normalized_legal_counts = legal_task_counts / legal_task_counts.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            alpha = torch.sigmoid(
                self.task_preference_gain
                + self.task_mass_calibration(
                    torch.cat(
                        (
                            task_control_preference,
                            preference_deficit,
                            normalized_legal_counts,
                        ),
                        dim=-1,
                    )
                )
            )
            mixed_group_mass = (
                (1.0 - alpha) * current_group_mass
                + alpha * target_group_mass
            )
            group_log_bias = torch.log(mixed_group_mass.clamp_min(1e-8)) - torch.log(
                current_group_mass.clamp_min(1e-8)
            )
            pair_group_bias = torch.einsum(
                "bti,bi->bt", task_type_one_hot, group_log_bias[:, :4]
            )
            pair_logits = pair_logits + pair_group_bias.repeat(1, self.max_uavs)
            noop_logits = noop_logits + group_log_bias[:, 4:5]
        else:
            task_preference_bias = torch.einsum(
                "bti,bi->bt",
                task_type_one_hot,
                preference_deficit
                + task_control_preference
                - balanced_task,
            )
            pair_logits = pair_logits + self.task_preference_gain * task_preference_bias.repeat(
                1, self.max_uavs
            )
        logits = torch.cat((pair_logits, noop_logits), dim=-1)
        logits = logits.masked_fill(~action_mask.bool(), -1e9)

        values = self.vector_critic_residual(
            torch.cat((pooled, critic_preference_embedding), dim=-1)
        )
        return torch.distributions.Categorical(logits=logits), values

    @torch.no_grad()
    def act(
        self,
        observation: dict[str, torch.Tensor],
        preference: torch.Tensor,
        *,
        task_preference: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self(
            observation["nodes"],
            observation["edge_types"],
            observation["edge_features"],
            observation["action_mask"],
            preference,
            observation.get("preference_deficit"),
            task_preference,
        )
        action = (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic
            else distribution.sample()
        )
        return action, distribution.log_prob(action), value
