from __future__ import annotations

import numpy as np
import pytest
import torch

from train_paper_faithful import resolve_device, tensors


def test_resolve_device_cpu_is_explicit() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_tensors_respect_requested_device() -> None:
    observation = {
        "nodes": np.zeros((2, 3), dtype=np.float32),
        "edge_types": np.zeros((2, 2), dtype=np.int64),
        "edge_features": np.zeros((2, 2, 1), dtype=np.float32),
        "action_mask": np.ones(3, dtype=np.bool_),
    }
    converted = tensors(observation, torch.device("cpu"))
    assert {value.device.type for value in converted.values()} == {"cpu"}


def test_cuda_request_fails_clearly_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        resolve_device("cuda")
