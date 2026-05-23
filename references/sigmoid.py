"""PyTorch reference: sigmoid."""
from __future__ import annotations

import torch


def sigmoid_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)
