import numpy as np
import torch
from torch import nn

from uav_assignment.paper_env import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    SELF_EDGE,
    UAV_COMM_EDGE,
    UAV_TASK_EDGE,
    PaperAlignedUAVEnv,
    PaperEnvConfig,
)
from uav_assignment.paper_models import (
    AdaptiveUAVAttention,
    PaperHeteroActorCritic,
    _masked_softmax,
)


def _single_uav_task_inputs() -> tuple[torch.Tensor, ...]:
    x = torch.tensor(
        [[[0.2, -0.1, 0.5], [1.0, -0.5, 0.25]]], dtype=torch.float32
    )
    raw_nodes = torch.zeros((1, 2, NODE_FEATURE_DIM), dtype=torch.float32)
    raw_nodes[0, 1, 7] = 0.8
    edge_types = torch.zeros((1, 2, 2), dtype=torch.long)
    edge_types[0, 0, 0] = SELF_EDGE
    edge_types[0, 0, 1] = UAV_TASK_EDGE
    edge_features = torch.zeros((1, 2, 2, 1), dtype=torch.float32)
    return x, raw_nodes, edge_types, edge_features


def test_eq3_self_and_task_value_transforms_are_independent() -> None:
    attention = AdaptiveUAVAttention(hidden_dim=3, edge_dim=1)
    assert attention.self_value is not attention.task_value
    assert attention.self_value.weight.data_ptr() != attention.task_value.weight.data_ptr()

    with torch.no_grad():
        attention.task_value.weight.fill_(2.0)
        task_before = attention.task_value.weight.clone()
        attention.self_value.weight.zero_()
    assert torch.equal(attention.task_value.weight, task_before)


def test_communication_edges_do_not_enter_task_attention_softmax() -> None:
    edge_types = torch.zeros((1, 1, 4), dtype=torch.long)
    edge_types[0, 0, 0] = SELF_EDGE
    edge_types[0, 0, 1] = UAV_COMM_EDGE
    edge_types[0, 0, 2:] = UAV_TASK_EDGE
    scores = torch.tensor([[[0.0, 2.0, 1.0, -1.0]]])
    mask = AdaptiveUAVAttention._task_attention_mask(edge_types, max_uavs=2)
    weights = _masked_softmax(scores, mask)

    changed_communication_score = scores.clone()
    changed_communication_score[0, 0, 1] = 1_000.0
    changed_weights = _masked_softmax(changed_communication_score, mask)

    assert not mask[0, 0, 1]
    assert weights[0, 0, 1] == 0
    assert torch.allclose(weights, changed_weights)
    assert torch.allclose(weights.sum(dim=-1), torch.ones((1, 1)))


def test_adaptive_gate_changes_eq3_task_contribution() -> None:
    attention = AdaptiveUAVAttention(hidden_dim=3, edge_dim=1).eval()
    x, raw_nodes, edge_types, edge_features = _single_uav_task_inputs()
    with torch.no_grad():
        attention.attention.zero_()
        attention.edge_bias.weight.zero_()
        attention.priority_scale.zero_()
        attention.self_value.weight.copy_(torch.eye(3))
        attention.task_value.weight.copy_(torch.eye(3))
        attention.task_edge_value.weight.zero_()
        attention.output.weight.copy_(torch.eye(3))
        attention.output.bias.zero_()
        attention.communication_output.weight.zero_()
        for layer in attention.gate:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        final_linear = attention.gate[2]
        assert isinstance(final_linear, nn.Linear)
        final_linear.bias.fill_(-8.0)

    gated = attention(
        x, raw_nodes, edge_types, edge_features, max_uavs=1, adaptive_gate=True
    )
    no_gate = attention(
        x, raw_nodes, edge_types, edge_features, max_uavs=1, adaptive_gate=False
    )
    assert not torch.allclose(gated[:, :1], no_gate[:, :1])


def test_communication_relation_is_fused_separately() -> None:
    attention = AdaptiveUAVAttention(hidden_dim=3, edge_dim=1).eval()
    x = torch.tensor(
        [[[0.2, -0.1, 0.5], [0.8, 0.3, -0.4], [1.0, -0.5, 0.25]]]
    )
    raw_nodes = torch.zeros((1, 3, NODE_FEATURE_DIM), dtype=torch.float32)
    edge_types = torch.zeros((1, 3, 3), dtype=torch.long)
    edge_types[0, 0, 0] = SELF_EDGE
    edge_types[0, 0, 2] = UAV_TASK_EDGE
    edge_features = torch.zeros((1, 3, 3, 1), dtype=torch.float32)
    with torch.no_grad():
        attention.attention.zero_()
        attention.edge_bias.weight.zero_()
        attention.priority_scale.zero_()
        attention.self_value.weight.copy_(torch.eye(3))
        attention.task_value.weight.copy_(torch.eye(3))
        attention.task_edge_value.weight.zero_()
        attention.communication_value.weight.copy_(torch.eye(3))
        attention.communication_edge_value.weight.zero_()
        attention.output.weight.copy_(torch.eye(3))
        attention.output.bias.zero_()
        attention.communication_output.weight.copy_(torch.eye(3))

    without_communication = attention(
        x, raw_nodes, edge_types, edge_features, max_uavs=1, adaptive_gate=False
    )
    edge_types[0, 0, 1] = UAV_COMM_EDGE
    with_communication = attention(
        x, raw_nodes, edge_types, edge_features, max_uavs=1, adaptive_gate=False
    )
    assert not torch.allclose(
        without_communication[:, :1], with_communication[:, :1]
    )


def test_deterministic_rrelu_approximation_and_policy_mask_shapes() -> None:
    attention = AdaptiveUAVAttention(hidden_dim=3, edge_dim=1).eval()
    assert isinstance(attention.activation, nn.LeakyReLU)
    assert attention.activation.negative_slope == 0.2
    inputs = _single_uav_task_inputs()
    first = attention(*inputs, max_uavs=1)
    second = attention(*inputs, max_uavs=1)
    assert torch.equal(first, second)

    env = PaperAlignedUAVEnv(
        PaperEnvConfig(
            max_uavs=3,
            max_tasks=8,
            active_uavs=2,
            initial_tasks=6,
            weather_probability=0.0,
            failure_probability=0.0,
            task_change_probability=0.0,
            communication_drop_probability=0.0,
        )
    )
    observation = env.reset(seed=91)
    tensors = {
        "nodes": torch.as_tensor(observation["nodes"], dtype=torch.float32),
        "edge_types": torch.as_tensor(observation["edge_types"], dtype=torch.long),
        "edge_features": torch.as_tensor(
            observation["edge_features"], dtype=torch.float32
        ),
        "action_mask": torch.as_tensor(
            observation["action_mask"], dtype=torch.bool
        ),
    }
    invalid = ~tensors["action_mask"]
    for mode in ("single_head", "adaptive_no_gate", "adaptive"):
        model = PaperHeteroActorCritic(
            NODE_FEATURE_DIM,
            EDGE_FEATURE_DIM,
            env.config.max_uavs,
            env.config.max_tasks,
            hidden_dim=24,
            graph_mode=mode,
        ).eval()
        distribution, value = model(**tensors)
        assert distribution.logits.shape == (1, env.n_actions)
        assert value.shape == (1, 1)
        assert torch.all(distribution.probs[0, invalid] == 0)
        assert np.isfinite(distribution.probs.detach().numpy()).all()
