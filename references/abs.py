"""PyTorch reference: abs."""
from __future__ import annotations

import torch


def abs_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)
