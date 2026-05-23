"""PyTorch reference: LayerNorm with affine weight + bias.

reference: y = (x - mean) / sqrt(var + eps) * weight + bias
along the last dim. Contrast with RMSNorm: LayerNorm subtracts the mean
AND has a bias term. Hand-written kernels often confuse the two.
"""
from __future__ import annotations

import torch


def layernorm_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias
