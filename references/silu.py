"""PyTorch reference: SiLU activation (a.k.a. swish).

reference: SiLU(x) = x * sigmoid(x). Distinct from SwiGLU, which is
SiLU(gate) * up — SwiGLU consumes two tensors, SiLU consumes one.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_ref(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)
