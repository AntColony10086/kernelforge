"""PyTorch reference: elementwise add (a + b, broadcast same shape)."""
from __future__ import annotations

import torch


def elementwise_add_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b
