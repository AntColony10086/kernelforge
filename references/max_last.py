"""PyTorch reference: max along the last dim, keepdim=True (values only)."""
from __future__ import annotations

import torch


def max_last_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x, dim=-1, keepdim=True).values
