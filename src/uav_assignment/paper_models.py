from __future__ import annotations

import math

import torch
from torch import nn

from .paper_env import (
    SELF_EDGE,
    TASK_PREDECESSOR_EDGE,
    TASK_SUCCESSOR_EDGE,
    UAV_COMM_EDGE,
    UAV_TASK_EDGE,
)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over existing edges, returning zero for rows without edges."""

    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=-1) * mask.to(scores.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class EdgeAwareAttention(nn.Module):
    """The v1 single-head typed Q/K/V attention used as an ablation control."""

    def __init__(self, hidden_dim: int, edge_dim: int, n_edge_types: int = 6):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_value = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.edge_bias = nn.Embedding(n_edge_types, 1)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_types: torch.Tensor,
        edge_features: torch.Tensor,
        allowed_types: tuple[int, ...],
    ) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(x.shape[-1])
        type_mask = torch.zeros_like(edge_types, dtype=torch.bool)
        for edge_type in allowed_types:
            type_mask |= edge_types == edge_type
        scores = scores + self.edge_bias(edge_types.clamp_min(0)).squeeze(-1)
        weights = _masked_softmax(scores, type_mask)
        messages = torch.matmul(weights, v)
        edge_messages = self.edge_value(edge_features)
        messages = messages + torch.sum(weights.unsqueeze(-1) * edge_messages, dim=-2)
        updated = self.norm(x + self.output(messages))
        return torch.where(type_mask.any(dim=-1, keepdim=True), updated, x)


class AdaptiveUAVAttention(nn.Module):
    """AHGNN UAV update following paper Eqs. (1)-(3).

    A task-edge pair forms the extended neighbor representation. RReLU scores
    are normalized over valid heterogeneous edges and multiplied by a learned
    UAV-task gate. Task priority enters the score explicitly rather than only
    through the generic node encoder.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, n_edge_types: int = 6):
        super().__init__()
        self.target = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbor = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        # Eq. (3) uses distinct transforms for the UAV self term and the
        # extended task-edge term.  Communication is a separate relation path
        # and therefore cannot enter the paper task-attention normalization.
        self.self_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.task_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.task_edge_value = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.communication_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.communication_edge_value = nn.Linear(
            edge_dim, hidden_dim, bias=False
        )
        self.attention = nn.Parameter(torch.empty(3 * hidden_dim))
        self.edge_bias = nn.Embedding(n_edge_types, 1)
        self.priority_scale = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # The expected RReLU negative slope keeps PPO old/new log-probs
        # deterministic while preserving the paper's piecewise-linear score.
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.communication_output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.attention.view(1, -1))

    def forward(
        self,
        x: torch.Tensor,
        raw_nodes: torch.Tensor,
        edge_types: torch.Tensor,
        edge_features: torch.Tensor,
        max_uavs: int,
        adaptive_gate: bool = True,
    ) -> torch.Tensor:
        uav_x = x[:, :max_uavs]
        uav_edges = edge_features[:, :max_uavs]
        uav_edge_types = edge_types[:, :max_uavs]
        target = self.target(uav_x).unsqueeze(-2)
        neighbor = self.neighbor(x).unsqueeze(-3)
        edge = self.edge(uav_edges)
        target = target.expand(-1, -1, x.shape[1], -1)
        neighbor = neighbor.expand(-1, max_uavs, -1, -1)
        extended = torch.cat((target, neighbor, edge), dim=-1)
        scores = self.activation(torch.sum(extended * self.attention, dim=-1))
        scores = scores + self.edge_bias(uav_edge_types.clamp_min(0)).squeeze(-1)

        source_priority = raw_nodes[..., 7].unsqueeze(-2).expand_as(scores)
        task_sources = torch.arange(x.shape[1], device=x.device) >= max_uavs
        priority_mask = (uav_edge_types == UAV_TASK_EDGE) & task_sources.view(1, 1, -1)
        scores = scores + self.priority_scale * source_priority * priority_mask

        task_attention_mask = self._task_attention_mask(
            uav_edge_types, max_uavs
        )
        task_attention_weights = _masked_softmax(scores, task_attention_mask)

        priority_feature = source_priority.unsqueeze(-1)
        learned_gates = self.gate(
            torch.cat((extended, priority_feature), dim=-1)
        ).squeeze(-1)
        task_gates = (
            learned_gates
            if adaptive_gate
            else torch.ones_like(learned_gates) + 0.0 * learned_gates
        )
        gates = torch.where(priority_mask, task_gates, torch.ones_like(learned_gates))
        self_mask = uav_edge_types == SELF_EDGE
        self_messages = self.self_value(x).unsqueeze(-3)
        task_messages = (
            self.task_value(x).unsqueeze(-3)
            + self.task_edge_value(uav_edges)
        )
        paper_messages = (
            self_mask.unsqueeze(-1) * self_messages
            + priority_mask.unsqueeze(-1) * gates.unsqueeze(-1) * task_messages
        )
        paper_aggregated = torch.sum(
            task_attention_weights.unsqueeze(-1) * paper_messages, dim=-2
        )

        communication_mask = uav_edge_types == UAV_COMM_EDGE
        communication_weights = _masked_softmax(scores, communication_mask)
        communication_messages = (
            self.communication_value(x).unsqueeze(-3)
            + self.communication_edge_value(uav_edges)
        )
        communication_aggregated = torch.sum(
            communication_weights.unsqueeze(-1) * communication_messages,
            dim=-2,
        )
        fused = self.output(paper_aggregated) + self.communication_output(
            communication_aggregated
        )
        updated = self.norm(uav_x + fused)
        has_messages = task_attention_mask.any(
            dim=-1, keepdim=True
        ) | communication_mask.any(dim=-1, keepdim=True)
        updated = torch.where(has_messages, updated, uav_x)
        result = x.clone()
        result[:, :max_uavs] = updated
        return result

    @staticmethod
    def _task_attention_mask(
        uav_edge_types: torch.Tensor, max_uavs: int
    ) -> torch.Tensor:
        """Eq. (1)-(3) domain: UAV self plus executable task neighbors only."""

        source_is_task = (
            torch.arange(uav_edge_types.shape[-1], device=uav_edge_types.device)
            >= max_uavs
        )
        return (uav_edge_types == SELF_EDGE) | (
            (uav_edge_types == UAV_TASK_EDGE)
            & source_is_task.view(1, 1, -1)
        )


class AdaptiveTaskUpdate(nn.Module):
    """Five-MLP task update following paper Eqs. (4)-(5)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.predecessor_mlp = self._relation_mlp(hidden_dim)
        self.successor_mlp = self._relation_mlp(hidden_dim)
        self.uav_mlp = self._relation_mlp(hidden_dim)
        self.self_mlp = self._relation_mlp(hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _relation_mlp(hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _aggregate(
        x: torch.Tensor, mask: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        weights = mask.to(x.dtype)
        messages = torch.matmul(weights, x)
        if not normalize:
            return messages
        return messages / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_types: torch.Tensor,
        max_uavs: int,
        max_tasks: int,
    ) -> torch.Tensor:
        task_slice = slice(max_uavs, max_uavs + max_tasks)
        task_edges = edge_types[:, task_slice]
        task_x = x[:, task_slice]
        predecessor = self._aggregate(x, task_edges == TASK_PREDECESSOR_EDGE)
        successor = self._aggregate(x, task_edges == TASK_SUCCESSOR_EDGE)
        uav_neighbors = self._aggregate(
            x, task_edges == UAV_TASK_EDGE, normalize=False
        )
        fused = torch.cat(
            (
                self.predecessor_mlp(predecessor),
                self.successor_mlp(successor),
                self.uav_mlp(uav_neighbors),
                self.self_mlp(task_x),
            ),
            dim=-1,
        )
        updated = self.norm(task_x + self.fusion_mlp(fused))
        result = x.clone()
        result[:, task_slice] = updated
        return result


class PaperHeteroActorCritic(nn.Module):
    """Paper-aligned heterogeneous graph actor-critic.

    ``staged`` is retained as a v1 checkpoint-compatible alias for
    ``single_head``. Every graph mode instantiates every encoder branch, so
    parameter counts are exactly equal in controlled ablations.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        max_uavs: int,
        max_tasks: int,
        n_objectives: int = 1,
        hidden_dim: int = 96,
        graph_mode: str = "adaptive",
    ):
        super().__init__()
        if graph_mode not in {
            "none",
            "staged",
            "single_head",
            "adaptive_no_gate",
            "adaptive",
        }:
            raise ValueError(f"unsupported paper graph mode: {graph_mode}")
        self.max_uavs = max_uavs
        self.max_tasks = max_tasks
        self.graph_mode = "single_head" if graph_mode == "staged" else graph_mode
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.uav_attention = EdgeAwareAttention(hidden_dim, edge_feature_dim)
        self.task_attention = EdgeAwareAttention(hidden_dim, edge_feature_dim)
        self.adaptive_uav_attention = AdaptiveUAVAttention(
            hidden_dim, edge_feature_dim
        )
        self.adaptive_task_update = AdaptiveTaskUpdate(hidden_dim)
        self.pair_actor = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.noop_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(
        self,
        nodes: torch.Tensor,
        edge_types: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.node_encoder(nodes)
        if self.graph_mode == "none":
            return encoded
        if self.graph_mode in {"adaptive", "adaptive_no_gate"}:
            uav_stage = self.adaptive_uav_attention(
                encoded,
                nodes,
                edge_types,
                edge_features,
                self.max_uavs,
                adaptive_gate=self.graph_mode == "adaptive",
            )
            encoded = torch.cat(
                (uav_stage[:, : self.max_uavs], encoded[:, self.max_uavs :]), dim=1
            )
            return self.adaptive_task_update(
                encoded, edge_types, self.max_uavs, self.max_tasks
            )

        uav_stage = self.uav_attention(
            encoded,
            edge_types,
            edge_features,
            (SELF_EDGE, UAV_COMM_EDGE, UAV_TASK_EDGE),
        )
        encoded = torch.cat(
            (uav_stage[:, : self.max_uavs], encoded[:, self.max_uavs :]), dim=1
        )
        task_stage = self.task_attention(
            encoded,
            edge_types,
            edge_features,
            (SELF_EDGE, TASK_PREDECESSOR_EDGE, TASK_SUCCESSOR_EDGE, UAV_TASK_EDGE),
        )
        return torch.cat(
            (
                encoded[:, : self.max_uavs],
                task_stage[:, self.max_uavs : self.max_uavs + self.max_tasks],
            ),
            dim=1,
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edge_types: torch.Tensor,
        edge_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        if nodes.ndim == 2:
            nodes = nodes.unsqueeze(0)
            edge_types = edge_types.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        encoded = self._encode(nodes, edge_types, edge_features)
        uav = encoded[:, : self.max_uavs]
        task = encoded[:, self.max_uavs : self.max_uavs + self.max_tasks]
        pair_uav = uav.unsqueeze(2).expand(-1, -1, self.max_tasks, -1)
        pair_task = task.unsqueeze(1).expand(-1, self.max_uavs, -1, -1)
        pair_edges = edge_features[:, : self.max_uavs, self.max_uavs :]
        pair = torch.cat((pair_uav, pair_task, pair_edges), dim=-1)
        pair_logits = self.pair_actor(pair).reshape(nodes.shape[0], -1)
        active_nodes = nodes[..., 2].unsqueeze(-1)
        pooled = torch.sum(encoded * active_nodes, dim=1) / active_nodes.sum(
            dim=1
        ).clamp_min(1.0)
        logits = torch.cat((pair_logits, self.noop_actor(pooled)), dim=-1)
        logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return torch.distributions.Categorical(logits=logits), self.critic(pooled)

    @torch.no_grad()
    def act(
        self,
        observation: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self(
            observation["nodes"],
            observation["edge_types"],
            observation["edge_features"],
            observation["action_mask"],
        )
        action = (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic
            else distribution.sample()
        )
        return action, distribution.log_prob(action), value
