from __future__ import annotations

import torch

from uav_assignment.pcrl_models import PreferenceConditionedPaperActorCritic
from uav_assignment.pcrl_training import (
    grouped_policy_direction,
    model_conditioning_inputs,
)
from uav_assignment.pcrl_v0 import (
    map_task_preference_to_full,
    task_preference_profile,
)


def test_no_conditioning_removes_network_inputs_but_keeps_preco_signal() -> None:
    """Hard-4 isolates network conditioning, not multi-objective training."""

    model = PreferenceConditionedPaperActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=2,
        max_tasks=4,
        hidden_dim=16,
        graph_mode="adaptive",
        preference_conditioning=False,
    )
    observation = {
        "nodes": torch.zeros(6, 24),
        "edge_types": torch.zeros(6, 6, dtype=torch.long),
        "edge_features": torch.zeros(6, 6, 5),
        "action_mask": torch.ones(9, dtype=torch.bool),
        "preference_deficit": torch.tensor((0.4, -0.2, -0.1, -0.1)),
    }
    search = torch.as_tensor(
        map_task_preference_to_full(task_preference_profile("search"))
    )
    strike = torch.as_tensor(
        map_task_preference_to_full(task_preference_profile("strike"))
    )

    search_observation, search_network_preference = model_conditioning_inputs(
        model, observation, search
    )
    strike_observation, strike_network_preference = model_conditioning_inputs(
        model, observation, strike
    )
    assert "preference_deficit" not in search_observation
    assert "preference_deficit" not in strike_observation
    assert torch.equal(search_network_preference, model.default_preference)
    assert torch.equal(strike_network_preference, model.default_preference)

    generator = torch.Generator().manual_seed(19)
    advantages = torch.randn(12, 7, generator=generator)
    returns = torch.randn(12, 7, generator=generator)
    preference_ids = torch.zeros(12, dtype=torch.long)
    search_direction, _ = grouped_policy_direction(
        advantages,
        returns,
        search.expand(12, -1),
        preference_ids,
        algorithm="preco",
        preco_lambda=0.25,
    )
    strike_direction, _ = grouped_policy_direction(
        advantages,
        returns,
        strike.expand(12, -1),
        preference_ids,
        algorithm="preco",
        preco_lambda=0.25,
    )
    assert not torch.allclose(search_direction, strike_direction)
