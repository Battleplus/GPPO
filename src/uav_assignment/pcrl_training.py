from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn

from .paper_env import PaperEnvConfig
from .pcrl_models import (
    MASS_CONTROL_PREFERENCE_INPUT_MODE,
    PreferenceConditionedPaperActorCritic,
)
from .pcrl_v0 import (
    LEGACY_CAPABILITY_COVERAGE_MODE,
    DEFAULT_ASSIGNMENT_PRIORITY_DECAY,
    DEFAULT_PREFERENCE_TARGET_MODE,
    LEGACY_PREFERENCE_SAMPLER_MODE,
    N_OBJECTIVES,
    PreferencePaperEnv,
    map_task_preference_to_full,
    sample_training_task_preference,
)
from .preco import min_norm_weights


PCRLAlgorithm = Literal["preco", "ls", "sdmgrad"]


@dataclass(slots=True)
class PCRLTrainingConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    update_epochs: int = 4
    minibatch_size: int = 256
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 0.5
    preco_lambda: float = 0.25
    preference_aux_coefficient: float = 0.10
    preference_group_episodes: int = 1
    advantage_normalization_mode: str = "per_objective"
    preco_value_transform: str = "legacy_min_shift"
    seed: int = 1


@dataclass(slots=True)
class PCRLRolloutBatch:
    nodes: torch.Tensor
    edge_types: torch.Tensor
    edge_features: torch.Tensor
    action_mask: torch.Tensor
    preference_deficit: torch.Tensor
    task_preferences: torch.Tensor
    objective_preferences: torch.Tensor
    preference_ids: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    episode_metrics: list[dict[str, object]]
    trajectory_ids: torch.Tensor | None = None

    @property
    def preferences(self) -> torch.Tensor:
        """Backward-compatible alias for the seven-dimensional objective tape."""

        return self.objective_preferences


def tensor_observation(
    observation: dict[str, np.ndarray],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        "nodes": torch.as_tensor(
            observation["nodes"], dtype=torch.float32, device=device
        ),
        "edge_types": torch.as_tensor(
            observation["edge_types"], dtype=torch.long, device=device
        ),
        "edge_features": torch.as_tensor(
            observation["edge_features"], dtype=torch.float32, device=device
        ),
        "action_mask": torch.as_tensor(
            observation["action_mask"], dtype=torch.bool, device=device
        ),
        **(
            {
                "preference_deficit": torch.as_tensor(
                    observation["preference_deficit"],
                    dtype=torch.float32,
                    device=device,
                )
            }
            if "preference_deficit" in observation
            else {}
        ),
    }


def model_conditioning_inputs(
    model: PreferenceConditionedPaperActorCritic,
    observation: dict[str, torch.Tensor],
    task_preference: torch.Tensor,
    objective_preference: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor] | tuple[
    dict[str, torch.Tensor], torch.Tensor, torch.Tensor
]:
    """Remove conditional inputs at the no-conditioning model boundary."""

    legacy_call = objective_preference is None
    if legacy_call:
        objective_preference = task_preference
    if model.preference_conditioning:
        if legacy_call:
            return observation, objective_preference
        return observation, task_preference, objective_preference
    sanitized = dict(observation)
    sanitized.pop("preference_deficit", None)
    batch_size = (
        int(sanitized["nodes"].shape[0])
        if sanitized["nodes"].ndim == 3
        else 1
    )
    neutral_objective = model.default_preference.to(
        device=sanitized["nodes"].device, dtype=torch.float32
    ).expand(batch_size, -1)
    neutral_task = torch.full(
        (batch_size, 4),
        0.25,
        device=sanitized["nodes"].device,
        dtype=torch.float32,
    )
    if legacy_call:
        return (
            sanitized,
            neutral_objective.squeeze(0)
            if batch_size == 1
            else neutral_objective,
        )
    if batch_size == 1:
        return sanitized, neutral_task.squeeze(0), neutral_objective.squeeze(0)
    return sanitized, neutral_task, neutral_objective


def vector_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
    bootstrap_value: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rewards.ndim != 2 or rewards.shape[1] != N_OBJECTIVES:
        raise ValueError(f"rewards must have shape [steps, {N_OBJECTIVES}]")
    if values.shape != rewards.shape:
        raise ValueError("values must have the same shape as rewards")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    accumulator = np.zeros(N_OBJECTIVES, dtype=np.float32)
    next_value = (
        np.zeros(N_OBJECTIVES, dtype=np.float32)
        if bootstrap_value is None
        else np.asarray(bootstrap_value, dtype=np.float32)
    )
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        accumulator = delta + gamma * gae_lambda * accumulator
        advantages[index] = accumulator
        next_value = values[index]
    return advantages, advantages + values


def collect_pcrl_rollouts(
    model: PreferenceConditionedPaperActorCritic,
    config: PaperEnvConfig,
    *,
    episodes: int,
    seed_offset: int,
    sync_mode: str,
    gamma: float,
    gae_lambda: float,
    scale_choices: tuple[tuple[int, int], ...] = (),
    scale_deadlines: dict[tuple[int, int], float] | None = None,
    fixed_task_preferences: Sequence[np.ndarray] | None = None,
    task_release_modes: tuple[str, ...] = ("chain",),
    phase_staggered_deadline_scale: float = 1.0,
    assignment_priority_decay: float = DEFAULT_ASSIGNMENT_PRIORITY_DECAY,
    preference_target_mode: str = DEFAULT_PREFERENCE_TARGET_MODE,
    preference_sampler_mode: str = LEGACY_PREFERENCE_SAMPLER_MODE,
    calibration_priority_share: float | None = None,
    preference_sampler_concentration: float = 0.7,
    preference_sampler_anchor_probability: float = 0.30,
    preference_sampler_anchor_jitter: float = 0.10,
    capability_coverage_mode: str = LEGACY_CAPABILITY_COVERAGE_MODE,
    episodes_per_preference: int = 1,
    device: torch.device | str = "cpu",
) -> PCRLRolloutBatch:
    if episodes_per_preference <= 0:
        raise ValueError("episodes_per_preference must be positive")
    scale_deadlines = scale_deadlines or {}
    storage: dict[str, list[np.ndarray | int | float]] = {
        "nodes": [],
        "edge_types": [],
        "edge_features": [],
        "action_mask": [],
        "preference_deficit": [],
        "task_preferences": [],
        "objective_preferences": [],
        "preference_ids": [],
        "trajectory_ids": [],
        "actions": [],
        "old_log_probs": [],
        "returns": [],
        "advantages": [],
    }
    all_metrics: list[dict[str, object]] = []
    model.eval()
    preference_rng = np.random.default_rng(
        7_919 * config.seed + 104_729 * seed_offset + 17
    )
    task_preference: np.ndarray | None = None
    for episode in range(episodes):
        episode_config = config
        if scale_choices:
            active_uavs, initial_tasks = scale_choices[
                (seed_offset + episode) % len(scale_choices)
            ]
            episode_config = replace(
                config,
                active_uavs=active_uavs,
                initial_tasks=initial_tasks,
                mission_deadline=scale_deadlines.get(
                    (active_uavs, initial_tasks), config.mission_deadline
                ),
            )
        preference_group = episode // episodes_per_preference
        if episode % episodes_per_preference == 0:
            if fixed_task_preferences:
                task_preference = np.asarray(
                    fixed_task_preferences[
                        preference_group % len(fixed_task_preferences)
                    ],
                    dtype=np.float32,
                )
            else:
                task_preference = sample_training_task_preference(
                    preference_rng,
                    sampler_mode=preference_sampler_mode,
                    calibration_priority_share=calibration_priority_share,
                    concentration=preference_sampler_concentration,
                    anchor_probability=preference_sampler_anchor_probability,
                    anchor_jitter=preference_sampler_anchor_jitter,
                )
        if task_preference is None:
            raise AssertionError("preference group was not initialized")
        preference = map_task_preference_to_full(
            task_preference,
            preference_target_mode=preference_target_mode,
        )
        task_release_mode = task_release_modes[
            (seed_offset + episode) % len(task_release_modes)
        ]
        if task_release_mode == "phase_staggered":
            episode_config = replace(
                episode_config,
                mission_deadline=(
                    episode_config.mission_deadline
                    * phase_staggered_deadline_scale
                ),
            )
        env = PreferencePaperEnv(
            episode_config,
            preference=task_preference,
            task_release_mode=task_release_mode,
            assignment_priority_decay=assignment_priority_decay,
            preference_target_mode=preference_target_mode,
            capability_coverage_mode=capability_coverage_mode,
        )
        episode_seed = 100_000 * config.seed + seed_offset + episode
        observation = env.reset(seed=episode_seed)
        states: list[dict[str, np.ndarray]] = []
        actions: list[int] = []
        old_log_probs: list[float] = []
        rewards: list[np.ndarray] = []
        values: list[np.ndarray] = []
        done = False
        final_info: dict[str, object] = {}
        task_preference_tensor = torch.as_tensor(
            env.user_task_preference, dtype=torch.float32, device=device
        )
        objective_preference_tensor = torch.as_tensor(
            preference, dtype=torch.float32, device=device
        )
        while not done:
            state = {
                key: observation[key]
                for key in (
                    "nodes",
                    "edge_types",
                    "edge_features",
                    "action_mask",
                    "preference_deficit",
                )
            }
            torch_observation = tensor_observation(observation, device)
            (
                policy_observation,
                policy_task_preference,
                policy_objective_preference,
            ) = model_conditioning_inputs(
                model,
                torch_observation,
                task_preference_tensor,
                objective_preference_tensor,
            )
            with torch.no_grad():
                action, log_prob, value = model.act(
                    policy_observation,
                    policy_objective_preference,
                    task_preference=policy_task_preference,
                )
            next_observation, _, done, info = env.step(
                int(action.item()), sync_mode=sync_mode
            )
            states.append(state)
            actions.append(int(action.item()))
            old_log_probs.append(float(log_prob.item()))
            rewards.append(np.asarray(info["vector_reward"], dtype=np.float32))
            values.append(value.squeeze(0).detach().cpu().numpy())
            final_info = info
            observation = next_observation

        bootstrap = np.zeros(N_OBJECTIVES, dtype=np.float32)
        if final_info.get("termination_reason") == "max_decisions":
            (
                bootstrap_observation,
                bootstrap_task_preference,
                bootstrap_objective_preference,
            ) = model_conditioning_inputs(
                model,
                tensor_observation(observation, device),
                task_preference_tensor,
                objective_preference_tensor,
            )
            with torch.no_grad():
                _, next_value = model(
                    **bootstrap_observation,
                    preference=bootstrap_objective_preference,
                    task_preference=bootstrap_task_preference,
                )
            bootstrap = next_value.squeeze(0).detach().cpu().numpy()
        advantages, returns = vector_gae(
            np.asarray(rewards, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            gamma,
            gae_lambda,
            bootstrap,
        )
        preference_id = seed_offset + preference_group
        for index, state in enumerate(states):
            for key in (
                "nodes",
                "edge_types",
                "edge_features",
                "action_mask",
                "preference_deficit",
            ):
                storage[key].append(state[key])
            storage["task_preferences"].append(env.user_task_preference)
            storage["objective_preferences"].append(preference)
            storage["preference_ids"].append(preference_id)
            storage["trajectory_ids"].append(seed_offset + episode)
            storage["actions"].append(actions[index])
            storage["old_log_probs"].append(old_log_probs[index])
            storage["advantages"].append(advantages[index])
            storage["returns"].append(returns[index])
        metrics = env.metrics()
        metrics["preference_id"] = int(preference_id)
        metrics["episode_seed"] = int(episode_seed)
        all_metrics.append(metrics)

    return PCRLRolloutBatch(
        nodes=torch.as_tensor(np.asarray(storage["nodes"]), dtype=torch.float32),
        edge_types=torch.as_tensor(
            np.asarray(storage["edge_types"]), dtype=torch.long
        ),
        edge_features=torch.as_tensor(
            np.asarray(storage["edge_features"]), dtype=torch.float32
        ),
        action_mask=torch.as_tensor(
            np.asarray(storage["action_mask"]), dtype=torch.bool
        ),
        preference_deficit=torch.as_tensor(
            np.asarray(storage["preference_deficit"]), dtype=torch.float32
        ),
        task_preferences=torch.as_tensor(
            np.asarray(storage["task_preferences"]), dtype=torch.float32
        ),
        objective_preferences=torch.as_tensor(
            np.asarray(storage["objective_preferences"]), dtype=torch.float32
        ),
        preference_ids=torch.as_tensor(
            np.asarray(storage["preference_ids"]), dtype=torch.long
        ),
        actions=torch.as_tensor(np.asarray(storage["actions"]), dtype=torch.long),
        old_log_probs=torch.as_tensor(
            np.asarray(storage["old_log_probs"]), dtype=torch.float32
        ),
        returns=torch.as_tensor(np.asarray(storage["returns"]), dtype=torch.float32),
        advantages=torch.as_tensor(
            np.asarray(storage["advantages"]), dtype=torch.float32
        ),
        episode_metrics=all_metrics,
        trajectory_ids=torch.as_tensor(
            np.asarray(storage["trajectory_ids"]), dtype=torch.long
        ),
    )


def ray_similarity_gradient(
    preference: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Subgradient of the PCRL ray-similarity objective in value space.

    Psi(p, v) = -0.5 ||max_i(v_i / p_i) p - v||^2.
    Values are shifted to the positive orthant by the caller because the UAV
    reward vector contains negative cost objectives.
    """

    eps = torch.finfo(value.dtype).eps
    p = preference.clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    ratios = value / p
    active = int(torch.argmax(ratios).item())
    scale = ratios[active]
    residual = scale * p - value
    gradient = residual.clone()
    gradient[active] = residual[active] - torch.dot(residual, p) / p[active]
    return gradient


def _preco_group_direction(
    advantages: torch.Tensor,
    returns: torch.Tensor,
    preference: torch.Tensor,
    coefficient: float,
    value_transform: str = "legacy_min_shift",
    estimated_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    objective_directions = advantages.transpose(0, 1)
    if estimated_value is None:
        estimated_value = returns.mean(dim=0)
    if value_transform == "legacy_min_shift":
        estimated_value = estimated_value - estimated_value.min() + 1e-4
    elif value_transform == "softplus":
        estimated_value = torch.nn.functional.softplus(estimated_value)
    else:
        raise ValueError(f"unsupported PreCo value transform: {value_transform}")
    similarity_gradient = ray_similarity_gradient(preference, estimated_value)
    similarity_gradient = similarity_gradient / torch.linalg.vector_norm(
        similarity_gradient
    ).clamp_min(1e-8)
    similarity_direction = similarity_gradient @ objective_directions
    weights = min_norm_weights(objective_directions, bias=similarity_direction)
    moo_direction = weights @ objective_directions + similarity_direction
    preference_direction = advantages @ preference
    direction = (
        (1.0 - coefficient) * preference_direction
        + coefficient * moo_direction
    )
    return direction.detach(), weights.detach()


def preference_action_alignment_loss(
    action_probabilities: torch.Tensor,
    nodes: torch.Tensor,
    action_mask: torch.Tensor,
    task_preferences: torch.Tensor,
    *,
    max_uavs: int,
    max_tasks: int,
    preference_deficit: torch.Tensor | None = None,
    include_noop_mass: bool = False,
) -> torch.Tensor:
    """Cross-entropy between legal action-type mass and task preference.

    The loss is active only when at least one pair action is legal. It cannot
    bypass dependencies or action masks; it only resolves choices among task
    types that the frozen environment already declares executable.
    """

    pair_probabilities = action_probabilities[:, :-1].reshape(
        -1, max_uavs, max_tasks
    )
    pair_mask = action_mask[:, :-1].reshape(-1, max_uavs, max_tasks)
    task_types = nodes[:, max_uavs : max_uavs + max_tasks, 16:20]
    predicted_mass = torch.einsum(
        "but,bti->bi", pair_probabilities, task_types
    )
    legal_task_mass = torch.einsum(
        "but,bti->bi", pair_mask.to(nodes.dtype), task_types
    )
    legal_types = legal_task_mass > 0
    if task_preferences.ndim != 2 or task_preferences.shape[-1] not in {4, 7}:
        raise ValueError("task_preferences must have shape [batch, 4]")
    if task_preferences.shape[-1] == 7:
        task_preferences = task_preferences[:, :4]
    if include_noop_mass:
        if preference_deficit is None:
            raise ValueError("task-mass alignment requires preference_deficit")
        target_task_mass = PreferenceConditionedPaperActorCritic.task_mass_target(
            task_preferences,
            preference_deficit,
            legal_task_mass,
        )
        valid_rows = legal_types.sum(dim=-1) >= 2
        if not bool(valid_rows.any()):
            return action_probabilities.sum() * 0.0
        pair_mass = pair_probabilities.sum(dim=(1, 2)).detach()
        target_group_mass = torch.cat(
            (
                pair_mass[:, None] * target_task_mass,
                (1.0 - pair_mass)[:, None],
            ),
            dim=-1,
        )
        predicted_group_mass = torch.cat(
            (predicted_mass, action_probabilities[:, -1:]), dim=-1
        )
        cross_entropy = -torch.sum(
            target_group_mass
            * torch.log(predicted_group_mass.clamp_min(1e-8)),
            dim=-1,
        )
        return cross_entropy[valid_rows].mean()
    target = task_preferences * legal_types.to(task_preferences.dtype)
    valid_rows = target.sum(dim=-1) > 0
    if not bool(valid_rows.any()):
        return action_probabilities.sum() * 0.0
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    predicted_mass = predicted_mass * legal_types.to(predicted_mass.dtype)
    predicted_mass = predicted_mass / predicted_mass.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    cross_entropy = -torch.sum(
        target * torch.log(predicted_mass.clamp_min(1e-8)), dim=-1
    )
    return cross_entropy[valid_rows].mean()


def grouped_policy_direction(
    advantages: torch.Tensor,
    returns: torch.Tensor,
    preferences: torch.Tensor,
    preference_ids: torch.Tensor,
    *,
    algorithm: PCRLAlgorithm,
    preco_lambda: float,
    preco_value_transform: str = "legacy_min_shift",
    trajectory_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (
        advantages.shape == returns.shape == preferences.shape
        and advantages.shape[0] == preference_ids.shape[0]
    ):
        raise ValueError("grouped policy tensors have incompatible shapes")
    direction = torch.zeros(
        advantages.shape[0], dtype=advantages.dtype, device=advantages.device
    )
    weights: list[torch.Tensor] = []
    for preference_id in torch.unique(preference_ids):
        mask = preference_ids == preference_id
        group_advantages = advantages[mask]
        group_returns = returns[mask]
        group_preference = preferences[mask][0]
        if not torch.allclose(
            preferences[mask],
            group_preference.expand_as(preferences[mask]),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("one episode/preference_id contains multiple preferences")
        if algorithm == "ls":
            group_direction = group_advantages @ group_preference
            group_weights = group_preference
        elif algorithm == "sdmgrad":
            objective_directions = group_advantages.transpose(0, 1)
            group_weights = min_norm_weights(objective_directions)
            group_direction = group_weights @ objective_directions
        elif algorithm == "preco":
            group_estimated_value = None
            if trajectory_ids is not None:
                group_trajectory_ids = trajectory_ids[mask]
                initial_returns = []
                for trajectory_id in torch.unique(group_trajectory_ids):
                    first = torch.nonzero(
                        group_trajectory_ids == trajectory_id, as_tuple=False
                    )[0, 0]
                    initial_returns.append(group_returns[first])
                group_estimated_value = torch.stack(initial_returns).mean(dim=0)
            group_direction, group_weights = _preco_group_direction(
                group_advantages,
                group_returns,
                group_preference,
                preco_lambda,
                preco_value_transform,
                group_estimated_value,
            )
        else:
            raise ValueError(f"unsupported PCRL algorithm: {algorithm}")
        direction[mask] = group_direction
        weights.append(group_weights)
    if direction.numel() > 1:
        direction = (direction - direction.mean()) / direction.std().clamp_min(1e-6)
    return direction.detach(), torch.stack(weights).mean(dim=0).detach()


def pcrl_ppo_update(
    model: PreferenceConditionedPaperActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PCRLRolloutBatch,
    config: PCRLTrainingConfig,
    *,
    algorithm: PCRLAlgorithm,
    device: torch.device | str = "cpu",
    update_index: int = 0,
) -> dict[str, float]:
    model.train()
    tensors = {
        "nodes": batch.nodes.to(device),
        "edge_types": batch.edge_types.to(device),
        "edge_features": batch.edge_features.to(device),
        "action_mask": batch.action_mask.to(device),
        "preference_deficit": batch.preference_deficit.to(device),
        "task_preferences": batch.task_preferences.to(device),
        "objective_preferences": batch.objective_preferences.to(device),
        "preference_ids": batch.preference_ids.to(device),
        "trajectory_ids": (
            batch.trajectory_ids.to(device)
            if batch.trajectory_ids is not None
            else None
        ),
        "actions": batch.actions.to(device),
        "old_log_probs": batch.old_log_probs.to(device),
        "returns": batch.returns.to(device),
        "advantages": batch.advantages.to(device),
    }
    advantages = tensors["advantages"]
    if config.advantage_normalization_mode == "per_objective":
        advantages = (advantages - advantages.mean(dim=0)) / advantages.std(
            dim=0
        ).clamp_min(1e-6)
    elif config.advantage_normalization_mode == "shared_scale":
        advantages = advantages - advantages.mean(dim=0)
        advantages = advantages / advantages.std().clamp_min(1e-6)
    elif config.advantage_normalization_mode != "none":
        raise ValueError(
            "unsupported advantage normalization mode: "
            f"{config.advantage_normalization_mode}"
        )
    tensors["advantages"] = advantages
    sample_count = advantages.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + 10_007 * update_index
    )
    actor_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []
    clip_fractions: list[float] = []
    preference_aux_losses: list[float] = []
    weight_rows: list[np.ndarray] = []
    policy_directions, objective_weights = grouped_policy_direction(
        tensors["advantages"],
        tensors["returns"],
        tensors["objective_preferences"],
        tensors["preference_ids"],
        algorithm=algorithm,
        preco_lambda=config.preco_lambda,
        preco_value_transform=config.preco_value_transform,
        trajectory_ids=tensors["trajectory_ids"],
    )
    for _ in range(config.update_epochs):
        permutation = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size].to(device)
            policy_direction = policy_directions[indices]
            (
                policy_observation,
                policy_task_preference,
                policy_objective_preference,
            ) = model_conditioning_inputs(
                model,
                {
                    "nodes": tensors["nodes"][indices],
                    "edge_types": tensors["edge_types"][indices],
                    "edge_features": tensors["edge_features"][indices],
                    "action_mask": tensors["action_mask"][indices],
                    "preference_deficit": tensors["preference_deficit"][indices],
                },
                tensors["task_preferences"][indices],
                tensors["objective_preferences"][indices],
            )
            distribution, values = model(
                **policy_observation,
                preference=policy_objective_preference,
                task_preference=policy_task_preference,
            )
            log_probs = distribution.log_prob(tensors["actions"][indices])
            log_ratio = log_probs - tensors["old_log_probs"][indices]
            ratios = torch.exp(log_ratio)
            unclipped = ratios * policy_direction
            clipped = torch.clamp(
                ratios, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
            ) * policy_direction
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.smooth_l1_loss(
                values, tensors["returns"][indices]
            )
            preference_aux_loss = (
                preference_action_alignment_loss(
                    distribution.probs,
                    tensors["nodes"][indices],
                    tensors["action_mask"][indices],
                    tensors["task_preferences"][indices],
                    max_uavs=model.max_uavs,
                    max_tasks=model.max_tasks,
                    preference_deficit=tensors["preference_deficit"][indices],
                    include_noop_mass=(
                        model.preference_input_mode
                        == MASS_CONTROL_PREFERENCE_INPUT_MODE
                    ),
                )
                if model.preference_conditioning
                else distribution.probs.sum() * 0.0
            )
            entropy = distribution.entropy().mean()
            loss = (
                actor_loss
                + config.value_coefficient * value_loss
                + config.preference_aux_coefficient * preference_aux_loss
                - config.entropy_coefficient * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            actor_losses.append(float(actor_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            preference_aux_losses.append(
                float(preference_aux_loss.detach().cpu())
            )
            approximate_kls.append(float(((ratios - 1.0) - log_ratio).mean().detach().cpu()))
            clip_fractions.append(
                float(((ratios - 1.0).abs() > config.clip_ratio).float().mean().detach().cpu())
            )
            weight_rows.append(objective_weights.detach().cpu().numpy())
    mean_weights = np.mean(weight_rows, axis=0)
    result = {
        "actor_loss": float(np.mean(actor_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "preference_aux_loss": float(np.mean(preference_aux_losses)),
        "approximate_kl": float(np.mean(approximate_kls)),
        "clip_fraction": float(np.mean(clip_fractions)),
        "objective_weights_min": float(mean_weights.min()),
        "objective_weights_max": float(mean_weights.max()),
    }
    for index, value in enumerate(mean_weights):
        result[f"objective_weight_{index}"] = float(value)
    return result
