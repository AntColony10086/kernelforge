"""PyTorch reference: sum along the last dim, keepdim=True."""
from __future__ import annotations

import torch


def sum_last_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=-1, keepdim=True)
