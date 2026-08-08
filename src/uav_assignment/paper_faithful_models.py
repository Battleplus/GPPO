from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .paper_env import (
    SELF_EDGE,
    TASK_PREDECESSOR_EDGE,
    TASK_SUCCESSOR_EDGE,
    UAV_TASK_EDGE,
)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=-1) * mask.to(scores.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class DeterministicRReLU(nn.RReLU):
    """RReLU's paper-defined expected slope, deterministic for PPO ratios."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.rrelu(inputs, self.lower, self.upper, training=False)


class LiteralUAVAttention(nn.Module):
    """Literal Eqs. (1)-(3): one softmax domain and post-softmax gate."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        rrelu_mode: str = "expected",
        gate_bias_init: float = 0.0,
        gate_scope: str = "task_message",
        gate_activation: str = "sigmoid",
    ):
        super().__init__()
        if rrelu_mode not in {"expected", "stochastic"}:
            raise ValueError("rrelu_mode must be expected or stochastic")
        if gate_scope not in {"task_message", "score", "aggregate"}:
            raise ValueError("gate_scope must be task_message, score, or aggregate")
        if gate_activation not in {"sigmoid", "softplus"}:
            raise ValueError("gate_activation must be sigmoid or softplus")
        self.rrelu_mode = rrelu_mode
        self.gate_scope = gate_scope
        self.gate_activation = gate_activation
        self.uav_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Eq. (2) uses W^T v_k for the self term; it is distinct from W^U v_k.
        self.self_neighbor_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.task_edge_transform = nn.Linear(hidden_dim + edge_dim, hidden_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(2 * hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.gate[2].bias, gate_bias_init)
        self.activation = (
            DeterministicRReLU(lower=0.125, upper=1.0 / 3.0)
            if rrelu_mode == "expected"
            else nn.RReLU(lower=0.125, upper=1.0 / 3.0)
        )
        self.output_activation = nn.ELU()
        nn.init.xavier_uniform_(self.attention.view(1, -1))
        self.last_gates: torch.Tensor | None = None
        self.last_attention: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        edge_types: torch.Tensor,
        edge_features: torch.Tensor,
        max_uavs: int,
        use_gate: bool = True,
    ) -> torch.Tensor:
        uav = self.uav_transform(x[:, :max_uavs])
        source_x = x.unsqueeze(1).expand(-1, max_uavs, -1, -1)
        task_edge = self.task_edge_transform(torch.cat((source_x, edge_features[:, :max_uavs]), dim=-1))
        target = uav.unsqueeze(2).expand_as(task_edge)
        source_ids = torch.arange(x.shape[1], device=x.device)
        task_mask = (edge_types[:, :max_uavs] == UAV_TASK_EDGE) & (
            source_ids.view(1, 1, -1) >= max_uavs
        )
        self_mask = edge_types[:, :max_uavs] == SELF_EDGE
        # The self source is W_U v_k; a task source is the extended
        # task-edge representation W_T mu_ijk.
        self_neighbor = self.self_neighbor_transform(x[:, :max_uavs])
        self_neighbor = self_neighbor.unsqueeze(2).expand_as(task_edge)
        neighbor = torch.where(self_mask.unsqueeze(-1), self_neighbor, task_edge)
        pairs = torch.cat((target, neighbor), dim=-1)
        scores = self.activation(torch.sum(pairs * self.attention, dim=-1))
        domain = self_mask | task_mask
        gate_logits = self.gate(pairs).squeeze(-1)
        # The paper only specifies that f_{i,j,k} is a neural network; it does
        # not prescribe its output nonlinearity.  Keep the historical sigmoid
        # as the frozen formal default, while exposing a positive, unbounded
        # softplus sensitivity variant so that the gate can amplify as well as
        # attenuate a task message.  This avoids silently baking an extra
        # modelling assumption into the reproduction.
        learned_gate = (
            torch.sigmoid(gate_logits)
            if self.gate_activation == "sigmoid"
            else F.softplus(gate_logits)
        )
        task_gate = learned_gate if use_gate else torch.ones_like(learned_gate)
        if self.gate_scope == "score":
            # Score gating is the normalized interpretation: the gate changes
            # the competition before one joint self/task softmax.
            score_adjustment = torch.where(
                task_mask, torch.log(task_gate.clamp_min(1e-6)), torch.zeros_like(task_gate)
            )
            alpha = masked_softmax(scores + score_adjustment, domain)
            coefficients = alpha
        else:
            alpha = masked_softmax(scores, domain)
            coefficients = torch.where(task_mask, alpha * task_gate, alpha)
        messages = torch.where(task_mask.unsqueeze(-1), task_edge, target)
        aggregated = torch.sum(coefficients.unsqueeze(-1) * messages, dim=-2)
        if self.gate_scope == "aggregate":
            task_mass = torch.sum(alpha * task_mask.to(alpha.dtype), dim=-1, keepdim=True)
            aggregate_gate = torch.sum(
                alpha * task_gate * task_mask.to(alpha.dtype), dim=-1, keepdim=True
            ) / task_mass.clamp_min(1e-6)
            aggregate_gate = torch.where(task_mass > 0, aggregate_gate, torch.ones_like(aggregate_gate))
            aggregated = aggregated * aggregate_gate
        updated = self.output_activation(aggregated)
        self.last_gates = learned_gate
        self.last_attention = alpha
        result = x.clone()
        result[:, :max_uavs] = updated
        return result


class LiteralSingleHeadAttention(nn.Module):
    """Vanilla single-head GAT control on the exact same neighbor domain."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.output_activation = nn.ELU()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim + edge_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim + edge_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_types: torch.Tensor, edge_features: torch.Tensor, max_uavs: int) -> torch.Tensor:
        source = x.unsqueeze(1).expand(-1, max_uavs, -1, -1)
        extended = torch.cat((source, edge_features[:, :max_uavs]), dim=-1)
        query = self.query(x[:, :max_uavs]).unsqueeze(2)
        scores = torch.sum(query * self.key(extended), dim=-1) / math.sqrt(x.shape[-1])
        source_ids = torch.arange(x.shape[1], device=x.device)
        domain = (edge_types[:, :max_uavs] == SELF_EDGE) | (
            (edge_types[:, :max_uavs] == UAV_TASK_EDGE)
            & (source_ids.view(1, 1, -1) >= max_uavs)
        )
        updated = self.output_activation(
            torch.sum(
                masked_softmax(scores, domain).unsqueeze(-1) * self.value(extended),
                dim=-2,
            )
        )
        result = x.clone()
        result[:, :max_uavs] = updated
        return result


class LiteralTaskUpdate(nn.Module):
    """Eqs. (4)-(5): four relation MLPs followed by the fifth fusion MLP."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.predecessor_mlp = self._mlp(hidden_dim)
        self.successor_mlp = self._mlp(hidden_dim)
        self.uav_mlp = self._mlp(hidden_dim)
        self.self_mlp = self._mlp(hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim)
        )

    @staticmethod
    def _mlp(hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim))

    @staticmethod
    def _sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.matmul(mask.to(x.dtype), x)

    def forward(self, x: torch.Tensor, edge_types: torch.Tensor, max_uavs: int, max_tasks: int) -> torch.Tensor:
        task_slice = slice(max_uavs, max_uavs + max_tasks)
        edges = edge_types[:, task_slice]
        task_x = x[:, task_slice]
        relations = (
            self.predecessor_mlp(self._sum(x, edges == TASK_PREDECESSOR_EDGE)),
            self.successor_mlp(self._sum(x, edges == TASK_SUCCESSOR_EDGE)),
            self.uav_mlp(self._sum(x, edges == UAV_TASK_EDGE)),
            self.self_mlp(task_x),
        )
        result = x.clone()
        # Eq. (5): M_theta1(ELU(M_theta2 || M_theta3 || M_theta4 || M_theta5)).
        result[:, task_slice] = self.fusion_mlp(F.elu(torch.cat(relations, dim=-1)))
        return result


class PaperFaithfulActorCritic(nn.Module):
    MODES = {"ppo_mlp", "literal", "literal_no_gate", "literal_single_head"}

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        max_uavs: int,
        max_tasks: int,
        hidden_dim: int = 64,
        graph_mode: str = "literal",
        rrelu_mode: str = "expected",
        gate_bias_init: float = 0.0,
        gate_scope: str = "task_message",
        gate_activation: str = "sigmoid",
    ):
        super().__init__()
        if graph_mode not in self.MODES:
            raise ValueError(f"unsupported literal graph mode: {graph_mode}")
        self.max_uavs = max_uavs
        self.max_tasks = max_tasks
        self.graph_mode = graph_mode
        self.rrelu_mode = rrelu_mode
        self.gate_bias_init = gate_bias_init
        self.gate_scope = gate_scope
        self.gate_activation = gate_activation
        self.node_encoder = nn.Sequential(nn.Linear(node_feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim))
        self.literal_attention = LiteralUAVAttention(
            hidden_dim,
            edge_feature_dim,
            rrelu_mode=rrelu_mode,
            gate_bias_init=gate_bias_init,
            gate_scope=gate_scope,
            gate_activation=gate_activation,
        )
        self.single_attention = LiteralSingleHeadAttention(hidden_dim, edge_feature_dim)
        self.task_update = LiteralTaskUpdate(hidden_dim)
        self.pair_actor = nn.Sequential(nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))
        self.noop_actor = nn.Linear(hidden_dim, 1)
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 1))

    def _encode(self, nodes: torch.Tensor, edge_types: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        encoded = self.node_encoder(nodes)
        if self.graph_mode == "ppo_mlp":
            return encoded
        original_tasks = encoded[:, self.max_uavs:]
        if self.graph_mode == "literal_single_head":
            encoded = self.single_attention(encoded, edge_types, edge_features, self.max_uavs)
        else:
            encoded = self.literal_attention(
                encoded, edge_types, edge_features, self.max_uavs,
                use_gate=self.graph_mode == "literal",
            )
        encoded = torch.cat((encoded[:, :self.max_uavs], original_tasks), dim=1)
        return self.task_update(encoded, edge_types, self.max_uavs, self.max_tasks)

    def forward(self, nodes: torch.Tensor, edge_types: torch.Tensor, edge_features: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        if nodes.ndim == 2:
            nodes, edge_types, edge_features, action_mask = (
                nodes.unsqueeze(0), edge_types.unsqueeze(0), edge_features.unsqueeze(0), action_mask.unsqueeze(0)
            )
        encoded = self._encode(nodes, edge_types, edge_features)
        uav = encoded[:, :self.max_uavs]
        task = encoded[:, self.max_uavs:self.max_uavs + self.max_tasks]
        pair = torch.cat((
            uav.unsqueeze(2).expand(-1, -1, self.max_tasks, -1),
            task.unsqueeze(1).expand(-1, self.max_uavs, -1, -1),
            edge_features[:, :self.max_uavs, self.max_uavs:],
        ), dim=-1)
        pair_logits = self.pair_actor(pair).reshape(nodes.shape[0], -1)
        active = nodes[..., 2].unsqueeze(-1)
        pooled = torch.sum(encoded * active, dim=1) / active.sum(dim=1).clamp_min(1.0)
        logits = torch.cat((pair_logits, self.noop_actor(pooled)), dim=-1)
        logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return torch.distributions.Categorical(logits=logits), self.critic(pooled)

    @torch.no_grad()
    def act(self, observation: dict[str, torch.Tensor], deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self(**observation)
        action = torch.argmax(distribution.logits, dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value

    def active_parameter_count(self) -> int:
        prefixes = {"node_encoder", "pair_actor", "noop_actor", "critic"}
        if self.graph_mode != "ppo_mlp":
            prefixes.add("task_update")
            prefixes.add("single_attention" if self.graph_mode == "literal_single_head" else "literal_attention")
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.split(".", 1)[0] in prefixes
            and not (
                self.graph_mode == "literal_no_gate"
                and name.startswith("literal_attention.gate.")
            )
        )
