"""PyTorch reference: mean along the last dim, keepdim=True."""
from __future__ import annotations

import torch


def mean_last_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=-1, keepdim=True)
