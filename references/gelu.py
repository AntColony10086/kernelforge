"""PyTorch reference: GELU (Gaussian Error Linear Unit, exact variant).

reference: GELU(x) = x * Phi(x), where Phi is the standard normal CDF.
PyTorch's F.gelu(approximate='none') uses the exact CDF (via erf); the
'tanh' approximation is a common source of subtle correctness bugs in
hand-written kernels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gelu_ref(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="none")
