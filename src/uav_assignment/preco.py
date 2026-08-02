from __future__ import annotations

import numpy as np
import torch


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Project a vector onto {w >= 0, sum(w) = 1}."""
    vector = np.asarray(vector, dtype=np.float64)
    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, vector.size + 1)
    valid = sorted_values - cumulative / indices > 0
    rho = indices[valid][-1]
    theta = cumulative[valid][-1] / rho
    return np.maximum(vector - theta, 0.0)


def _project_simplex_torch(vector: torch.Tensor) -> torch.Tensor:
    sorted_values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - 1.0
    indices = torch.arange(
        1, vector.numel() + 1, dtype=vector.dtype, device=vector.device
    )
    valid = sorted_values - cumulative / indices > 0
    rho = torch.nonzero(valid, as_tuple=False)[-1, 0]
    theta = cumulative[rho] / indices[rho]
    return torch.clamp(vector - theta, min=0.0)


def min_norm_weights(
    objective_directions: torch.Tensor,
    bias: torch.Tensor | None = None,
    iterations: int = 80,
) -> torch.Tensor:
    """Solve min_w ||w^T G + bias||^2 over the probability simplex."""
    matrix = objective_directions.detach().double()
    bias_vector = torch.zeros(
        matrix.shape[1], dtype=matrix.dtype, device=matrix.device
    ) if bias is None else bias.detach().to(dtype=matrix.dtype, device=matrix.device)
    objectives = matrix.shape[0]
    weights = torch.full(
        (objectives,), 1.0 / objectives, dtype=matrix.dtype, device=matrix.device
    )
    gram = matrix @ matrix.T
    linear = matrix @ bias_vector
    # A row-sum upper bound avoids an SVD and is sufficient for a stable step.
    lipschitz = (2.0 * torch.sum(torch.abs(gram), dim=1).max()).clamp_min(1e-8)
    step = 1.0 / lipschitz
    for _ in range(iterations):
        gradient = 2.0 * (gram @ weights + linear)
        weights = _project_simplex_torch(weights - step * gradient)
    return weights.to(dtype=objective_directions.dtype)


def cosine_gradient(preference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    p_norm = torch.linalg.vector_norm(preference).clamp_min(eps)
    v_norm = torch.linalg.vector_norm(value).clamp_min(eps)
    cosine = torch.dot(preference, value) / (p_norm * v_norm)
    return preference / (p_norm * v_norm) - cosine * value / v_norm.square()


def preco_similarity_coefficients(
    preference: torch.Tensor, value: torch.Tensor, eta: float = 0.25
) -> torch.Tensor:
    """Stable PreCo-inspired coefficient in value space.

    The paper's practical discrete-action update operates at policy level. This
    implementation follows that design, blending the preference direction with a
    normalized cosine-similarity gradient before the min-norm problem.
    """
    eps = torch.finfo(value.dtype).eps
    positive_value = torch.clamp(value, min=eps)
    p = preference / preference.sum().clamp_min(eps)
    cosine = torch.dot(p, positive_value) / (
        torch.linalg.vector_norm(p).clamp_min(eps)
        * torch.linalg.vector_norm(positive_value).clamp_min(eps)
    )
    similarity_gradient = cosine_gradient(p, positive_value)
    similarity_gradient = similarity_gradient / torch.linalg.vector_norm(
        similarity_gradient
    ).clamp_min(eps)
    blend = torch.exp(-(1.0 - cosine).clamp_min(0.0) / eta)
    return blend * p + (1.0 - blend) * similarity_gradient
