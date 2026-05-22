"""PyTorch reference: SwiGLU (Llama-style: SiLU(gate) * up).

Input gate, up: shape (..., hidden_dim).
Returns: SiLU(gate) * up
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up
