"""Pin reference op behavior so refactors don't drift."""
from __future__ import annotations

import torch

from references.rmsnorm import rmsnorm_ref
from references.rope import rope_ref
from references.swiglu import swiglu_ref


def test_rmsnorm_known_values():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    out = rmsnorm_ref(x, w, eps=1e-12)
    rms = (sum(v * v for v in [1, 2, 3, 4]) / 4) ** 0.5
    expected = torch.tensor([[1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms]], dtype=torch.float64)
    assert torch.allclose(out, expected, atol=1e-10)


def test_rope_shape_preserves():
    x = torch.randn(2, 8, 64)
    out = rope_ref(x)
    assert out.shape == x.shape


def test_rope_identity_at_position_zero():
    """RoPE at position 0 is the identity rotation (sin=0, cos=1)."""
    x = torch.randn(1, 1, 64)
    out = rope_ref(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_swiglu_shape_preserves():
    gate = torch.randn(4, 16, 128)
    up = torch.randn(4, 16, 128)
    out = swiglu_ref(gate, up)
    assert out.shape == gate.shape
