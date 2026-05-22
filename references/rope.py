"""PyTorch reference: Rotary Position Embedding (split-half layout).

Input  x: shape (batch, seq, head_dim) - head_dim must be even.
Returns rotated x, where the first half is x_real and the second half is x_imag.

This is the canonical Llama / DeepSeek-style split-half layout. The
hidden-holdout suite uses this layout as ground truth; an LLM-generated
kernel that uses the interleaved (x0, x1, x0, x1, ...) layout will pass
a small shape smoke test by coincidence and fail on bigger shapes.
"""
from __future__ import annotations

import torch


def _build_sin_cos(seq_len: int, head_dim: int, base: float, device: torch.device, dtype: torch.dtype):
    assert head_dim % 2 == 0, "head_dim must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    sin = freqs.sin().to(dtype)
    cos = freqs.cos().to(dtype)
    return sin, cos


def rope_ref(x: torch.Tensor, *, base: float = 10000.0) -> torch.Tensor:
    batch, seq, head_dim = x.shape
    sin, cos = _build_sin_cos(seq, head_dim, base, x.device, x.dtype)
    x_real = x[..., : head_dim // 2]
    x_imag = x[..., head_dim // 2 :]
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    return torch.cat([out_real, out_imag], dim=-1)
