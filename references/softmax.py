"""PyTorch reference: softmax along the last dim.

reference: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))), per row.
The max subtraction is the numerical-stability trick every correct kernel must
implement; this is one of the most common ways an LLM-generated softmax goes
silently wrong.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)
