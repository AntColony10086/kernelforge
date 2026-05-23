"""PyTorch reference: ReLU."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def relu_ref(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)
