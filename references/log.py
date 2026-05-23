"""PyTorch reference: log (natural log). Inputs constrained to positive."""
from __future__ import annotations

import torch


def log_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)
