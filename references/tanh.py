"""PyTorch reference: tanh activation.

reference: tanh(x). The simplest possible kernel — included to anchor the
benchmark suite at a level where verification should always pass quickly.
"""
from __future__ import annotations

import torch


def tanh_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)
