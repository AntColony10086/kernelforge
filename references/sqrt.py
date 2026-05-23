"""PyTorch reference: sqrt. Inputs constrained to non-negative."""
from __future__ import annotations

import torch


def sqrt_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x)
