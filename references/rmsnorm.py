"""PyTorch reference: RMSNorm.

reference: x * weight / sqrt(mean(x**2, dim=-1, keepdim=True) + eps)
"""
from __future__ import annotations

import torch


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight
