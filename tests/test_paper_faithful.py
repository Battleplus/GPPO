import numpy as np
import torch

from uav_assignment.paper_env import SELF_EDGE, UAV_TASK_EDGE
from uav_assignment.paper_faithful_env import (
    PAPER_SCALES,
    PaperFaithfulConfig,
    PaperFaithfulUAVEnv,
    PaperScale,
    deterministic_instance_seeds,
)
from uav_assignment.paper_faithful_models import (
    DeterministicRReLU,
    LiteralTaskUpdate,
    LiteralUAVAttention,
    PaperFaithfulActorCritic,
)


def small_config(**kwargs) -> PaperFaithfulConfig:
    values = dict(
        scale=PaperScale(3, 2, 7),
        max_uavs=3,
        max_subtasks=7,
        reassignment_steps=(2,),
        max_decisions=200,
    )
    values.update(kwargs)
    return PaperFaithfulConfig(**values)


def test_paper_scales_and_instance_bank_are_exact_and_deterministic() -> None:
    assert [scale.name for scale in PAPER_SCALES] == [
        "T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92"
    ]
    seeds = deterministic_instance_seeds(PAPER_SCALES[0])
    assert len(seeds) == 100
    assert len(set(seeds)) == 100
    assert seeds == deterministic_instance_seeds(PAPER_SCALES[0])
    validation = deterministic_instance_seeds(PAPER_SCALES[0], split="validation")
    test = deterministic_instance_seeds(PAPER_SCALES[0], split="test")
    assert set(seeds).isdisjoint(validation)
    assert set(seeds).isdisjoint(test)
    assert set(validation).isdisjoint(test)


def test_parent_task_hierarchy_has_exact_subtask_partition() -> None:
    env = PaperFaithfulUAVEnv(small_config())
    observation = env.reset(seed=17)
    active_ids = env.parent_task_ids[:7]
    counts = np.bincount(active_ids, minlength=2)
    assert counts.tolist() == [4, 3]
    assert sum(task.active for task in env.tasks) == 7
    assert np.any(observation["edge_types"] == UAV_TASK_EDGE)
    for parent_id in range(2):
        chain = np.flatnonzero(active_ids == parent_id)
        assert env.tasks[int(chain[0])].predecessor == -1
        for previous, current in zip(chain[:-1], chain[1:]):
            assert env.tasks[int(current)].predecessor == int(previous)


def test_only_fixed_tape_event_triggers_event_sync() -> None:
    env = PaperFaithfulUAVEnv(small_config())
    observation = env.reset(seed=21)
    action = int(np.flatnonzero(observation["action_mask"][:-1])[0])
    observation, _, done, info = env.step(action, sync_mode="event")
    assert not done
    assert env.decision_count == 1
    assert not info["synchronized"]
    action = int(np.flatnonzero(observation["action_mask"])[0])
    _, _, _, info = env.step(action, sync_mode="event")
    assert env.decision_count == 2
    assert info["synchronized"]
    assert [row["event_type"] for row in info["events"]] == [
        "task_distribution_changed"
    ]


def test_same_instance_replays_identical_true_event_tape_across_sync_modes() -> None:
    event_env = PaperFaithfulUAVEnv(small_config(reassignment_steps=(1, 2, 3)))
    no_sync_env = PaperFaithfulUAVEnv(small_config(reassignment_steps=(1, 2, 3)))
    event_observation = event_env.reset(seed=23)
    no_sync_observation = no_sync_env.reset(seed=23)
    for _ in range(3):
        event_action = int(np.flatnonzero(event_observation["action_mask"])[0])
        no_sync_action = int(np.flatnonzero(no_sync_observation["action_mask"])[0])
        assert event_action == no_sync_action
        event_observation, _, _, event_info = event_env.step(event_action, "event")
        no_sync_observation, _, _, no_sync_info = no_sync_env.step(no_sync_action, "none")
        event_record = event_info["events"][0]
        no_sync_record = no_sync_info["events"][0]
        assert event_record["target_id"] == no_sync_record["target_id"]
        assert event_record["after"] == no_sync_record["after"]


def test_full_sync_uses_more_bytes_than_event_sync() -> None:
    event_env = PaperFaithfulUAVEnv(small_config(reassignment_steps=(1,)))
    full_env = PaperFaithfulUAVEnv(small_config(reassignment_steps=(1,)))
    event_observation = event_env.reset(seed=29)
    full_observation = full_env.reset(seed=29)
    for _ in range(4):
        event_action = int(np.flatnonzero(event_observation["action_mask"])[0])
        full_action = int(np.flatnonzero(full_observation["action_mask"])[0])
        event_observation, _, _, _ = event_env.step(event_action, "event")
        full_observation, _, _, full_info = full_env.step(full_action, "always")
    assert event_env.communication_bytes < full_env.communication_bytes
    assert event_env.metrics()["communication_bytes"] == event_env.communication_bytes
    assert full_env.communication_events == 4
    assert full_info["synchronized_uavs"] == list(range(full_env.config.max_uavs))


def test_realized_makespan_requires_actual_completions() -> None:
    env = PaperFaithfulUAVEnv(small_config(reassignment_steps=()))
    observation = env.reset(seed=31)
    assert env.realized_makespan == 0.0
    done = False
    for _ in range(env.config.max_decisions):
        valid = np.flatnonzero(observation["action_mask"])
        observation, _, done, _ = env.step(int(valid[0]), sync_mode="none")
        if done:
            break
    metrics = env.metrics()
    assert done
    assert metrics["all_tasks_completed"] == 1.0
    assert metrics["realized_makespan"] == env.current_time
    assert metrics["realized_makespan"] > 0.0


def literal_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # One UAV target, two task sources. The self and both task neighbors share
    # exactly one softmax domain.
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    edge_types = torch.zeros(1, 3, 3, dtype=torch.long)
    edge_types[0, 0, 0] = SELF_EDGE
    edge_types[0, 0, 1:] = UAV_TASK_EDGE
    edge_features = torch.zeros(1, 3, 3, 1)
    return x, edge_types, edge_features


def test_literal_eq_1_to_3_matches_manual_small_graph() -> None:
    attention = LiteralUAVAttention(hidden_dim=2, edge_dim=1)
    with torch.no_grad():
        attention.uav_transform.weight.copy_(torch.eye(2))
        attention.self_neighbor_transform.weight.copy_(
            torch.tensor([[0.0, 2.0], [3.0, 0.0]])
        )
        attention.task_edge_transform.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
        attention.attention.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0]))
    x, edge_types, edge_features = literal_inputs()
    actual = attention(x, edge_types, edge_features, max_uavs=1, use_gate=False)
    # Eq. (2) catches the distinct W^T self transform: self score is
    # [1, 0] dot [0, 3] = 4, not the old accidental score of 1.
    scores = torch.tensor([4.0, 2.0, 2.0])
    alpha = torch.softmax(scores, dim=0)
    expected = alpha[0] * x[0, 0] + alpha[1] * x[0, 1] + alpha[2] * x[0, 2]
    assert torch.allclose(actual[0, 0], expected, atol=1e-6)
    assert torch.allclose(attention.last_attention[0, 0].sum(), torch.tensor(1.0))


def test_gate_is_not_constant_and_receives_gradient() -> None:
    torch.manual_seed(3)
    attention = LiteralUAVAttention(hidden_dim=4, edge_dim=2)
    x = torch.randn(2, 3, 4)
    edge_types = torch.zeros(2, 3, 3, dtype=torch.long)
    edge_types[:, 0, 0] = SELF_EDGE
    edge_types[:, 0, 1:] = UAV_TASK_EDGE
    edge_features = torch.randn(2, 3, 3, 2)
    output = attention(x, edge_types, edge_features, max_uavs=1, use_gate=True)
    task_gates = attention.last_gates[:, 0, 1:]
    assert torch.std(task_gates) > 0
    output[:, 0].sum().backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in attention.gate.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0.0


def test_softplus_gate_is_positive_and_can_amplify_messages() -> None:
    torch.manual_seed(7)
    attention = LiteralUAVAttention(hidden_dim=4, edge_dim=2, gate_activation="softplus")
    x = torch.randn(1, 3, 4)
    edge_types = torch.zeros(1, 3, 3, dtype=torch.long)
    edge_types[0, 0, 0] = SELF_EDGE
    edge_types[0, 0, 1:] = UAV_TASK_EDGE
    edge_features = torch.randn(1, 3, 3, 2)
    attention(x, edge_types, edge_features, max_uavs=1)
    gates = attention.last_gates[0, 0, 1:]
    assert torch.all(gates > 0)
    with torch.no_grad():
        attention.gate[2].bias.fill_(3.0)
    attention(x, edge_types, edge_features, max_uavs=1)
    assert float(attention.last_gates[0, 0, 1:].mean().detach()) > 1.0


def test_gate_scope_variants_are_explicit_and_score_scope_renormalizes() -> None:
    torch.manual_seed(13)
    x, edge_types, edge_features = literal_inputs()
    message = LiteralUAVAttention(hidden_dim=2, edge_dim=1, gate_scope="task_message")
    score = LiteralUAVAttention(hidden_dim=2, edge_dim=1, gate_scope="score")
    aggregate = LiteralUAVAttention(hidden_dim=2, edge_dim=1, gate_scope="aggregate")
    score.load_state_dict(message.state_dict())
    aggregate.load_state_dict(message.state_dict())
    with torch.no_grad():
        # Make the gate visibly non-identity so the scopes cannot coincide by
        # accident; all other parameters remain exactly matched.
        message.gate[2].bias.fill_(-1.0)
        score.gate[2].bias.fill_(-1.0)
        aggregate.gate[2].bias.fill_(-1.0)
    message(x, edge_types, edge_features, max_uavs=1)
    message_alpha = message.last_attention.clone()
    score(x, edge_types, edge_features, max_uavs=1)
    score_alpha = score.last_attention.clone()
    aggregate_output = aggregate(x, edge_types, edge_features, max_uavs=1)
    assert torch.allclose(message_alpha.sum(dim=-1), torch.ones(1, 1))
    assert torch.allclose(score_alpha.sum(dim=-1), torch.ones(1, 1))
    # Message gating leaves the pre-gate softmax unchanged; score gating does
    # not. Aggregate scope is a separate, post-aggregation interpretation.
    assert not torch.allclose(message_alpha, score_alpha)
    assert not torch.allclose(aggregate_output[:, :1], message(x, edge_types, edge_features, max_uavs=1)[:, :1])


def test_task_priority_and_edge_features_change_attention() -> None:
    torch.manual_seed(7)
    attention = LiteralUAVAttention(hidden_dim=4, edge_dim=2)
    x = torch.randn(1, 3, 4)
    edge_types = torch.zeros(1, 3, 3, dtype=torch.long)
    edge_types[0, 0, 0] = SELF_EDGE
    edge_types[0, 0, 1:] = UAV_TASK_EDGE
    edge_features = torch.zeros(1, 3, 3, 2)
    attention(x, edge_types, edge_features, max_uavs=1)
    baseline = attention.last_attention.clone()
    changed_node = x.clone()
    changed_node[0, 1] += 2.0
    attention(changed_node, edge_types, edge_features, max_uavs=1)
    after_node = attention.last_attention.clone()
    changed_edge = edge_features.clone()
    changed_edge[0, 0, 1] = torch.tensor([2.0, -1.0])
    attention(x, edge_types, changed_edge, max_uavs=1)
    after_edge = attention.last_attention.clone()
    assert not torch.allclose(baseline, after_node)
    assert not torch.allclose(baseline, after_edge)


def test_literal_eq_4_and_5_matches_explicit_five_mlp_composition() -> None:
    torch.manual_seed(11)
    update = LiteralTaskUpdate(hidden_dim=3)
    x = torch.randn(1, 4, 3)
    # Nodes 0-1 are UAVs, nodes 2-3 are tasks. Task 0 sees one UAV and
    # task 1 as its successor; task 1 sees task 0 as predecessor.
    edge_types = torch.zeros(1, 4, 4, dtype=torch.long)
    edge_types[0, 2, 0] = UAV_TASK_EDGE
    edge_types[0, 2, 3] = 5
    edge_types[0, 3, 2] = 3
    actual = update(x, edge_types, max_uavs=2, max_tasks=2)
    task_edges = edge_types[:, 2:]
    predecessor = update.predecessor_mlp(update._sum(x, task_edges == 3))
    successor = update.successor_mlp(update._sum(x, task_edges == 5))
    uav = update.uav_mlp(update._sum(x, task_edges == UAV_TASK_EDGE))
    own = update.self_mlp(x[:, 2:])
    expected = update.fusion_mlp(
        torch.nn.functional.elu(torch.cat((predecessor, successor, uav, own), dim=-1))
    )
    assert torch.allclose(actual[:, 2:], expected)


def test_rrelu_uses_expected_randomized_slope_deterministically() -> None:
    activation = DeterministicRReLU(lower=0.125, upper=1.0 / 3.0)
    value = activation(torch.tensor([-2.0, 2.0]))
    expected_slope = (0.125 + 1.0 / 3.0) / 2.0
    assert torch.allclose(value, torch.tensor([-2.0 * expected_slope, 2.0]))
    activation.train()
    assert torch.equal(value, activation(torch.tensor([-2.0, 2.0])))


def test_stochastic_rrelu_is_random_only_in_training_mode() -> None:
    activation = torch.nn.RReLU(lower=0.125, upper=1.0 / 3.0)
    value = torch.full((100,), -2.0)
    activation.eval()
    expected_a = activation(value)
    expected_b = activation(value)
    assert torch.equal(expected_a, expected_b)
    activation.train()
    random_a = activation(value)
    random_b = activation(value)
    assert not torch.equal(random_a, random_b)


def test_four_structures_share_io_but_report_active_parameter_counts() -> None:
    env = PaperFaithfulUAVEnv(small_config())
    observation = env.reset(seed=41)
    tensors = {
        "nodes": torch.as_tensor(observation["nodes"]),
        "edge_types": torch.as_tensor(observation["edge_types"]),
        "edge_features": torch.as_tensor(observation["edge_features"]),
        "action_mask": torch.as_tensor(observation["action_mask"]),
    }
    counts = {}
    for mode in sorted(PaperFaithfulActorCritic.MODES):
        model = PaperFaithfulActorCritic(24, 5, 3, 7, hidden_dim=16, graph_mode=mode)
        distribution, value = model(**tensors)
        assert distribution.logits.shape == (1, env.n_actions)
        assert value.shape == (1, 1)
        counts[mode] = model.active_parameter_count()
    assert counts["literal"] > counts["literal_no_gate"]
    assert counts["ppo_mlp"] < counts["literal"]
