"""PyTorch reference: exp."""
from __future__ import annotations

import torch


def exp_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)
