"""Hidden holdout test cases per op.

The LLM never sees these inputs. After a failed verification, only the
case identifier + structured diff is fed back into the prompt.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch


def _det_seed(*parts) -> int:
    """Deterministic seed from arbitrary parts. Use this instead of Python's
    built-in hash() because PYTHONHASHSEED randomizes hash() per process.
    """
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big")

from references.rmsnorm import rmsnorm_ref
from references.rope import rope_ref
from references.swiglu import swiglu_ref


@dataclass(frozen=True)
class HoldoutCase:
    name: str
    op: str
    inputs_fn: Callable[[], dict[str, torch.Tensor | float]]
    reference_fn: Callable[..., torch.Tensor]
    dtype: Literal["float32", "float16", "bfloat16"]
    tolerance_abs: float = 1e-4
    tolerance_rel: float = 1e-3
    suspected_bug_hints: tuple[str, ...] = field(default_factory=tuple)


# ---- RoPE ----
def _rope_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, head_dim: int, dtype: str, base: float, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, head_dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"x": torch.randn(batch, seq, head_dim, dtype=t_dtype), "base": base}

        return HoldoutCase(name=name, op="rope", inputs_fn=inputs_fn, reference_fn=rope_ref, dtype=dtype, suspected_bug_hints=hints)

    # NOTE: base is NOT a kernel input — the kernel hardcodes the standard 10000.0
    # base and computes sin/cos internally. So all holdout cases use base=10000.0;
    # a variable-base holdout would be impossible to pass without changing the
    # kernel signature. We test other axes (shape / seq / dtype / position).
    out.append(case("rope_small_smoke", 1, 8, 64, "float32", 10000.0, ()))
    out.append(case("rope_batch_seq", 2, 32, 128, "float32", 10000.0, ("split-half-vs-interleaved layout", "wrong reduction dim")))
    out.append(case("rope_fp16", 4, 16, 256, "float16", 10000.0, ("dtype precision", "fp16 sin/cos cast")))
    out.append(case("rope_large_position", 1, 256, 64, "float32", 10000.0, ("position-id overflow", "freq exponent precision")))
    out.append(case("rope_non_power_two_seq", 2, 24, 128, "float32", 10000.0, ("non-pow-2 sequence handling",)))
    return out


# ---- RMSNorm ----
def _rmsnorm_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, eps: float, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"x": torch.randn(batch, seq, dim, dtype=t_dtype), "weight": torch.ones(dim, dtype=t_dtype), "eps": eps}

        return HoldoutCase(name=name, op="rmsnorm", inputs_fn=inputs_fn, reference_fn=rmsnorm_ref, dtype=dtype, suspected_bug_hints=hints)

    # NOTE: eps is NOT a kernel input — the kernel hardcodes the standard 1e-6.
    # All holdouts share the same eps; we test shape / dtype / magnitude.
    out.append(case("rmsnorm_small", 1, 8, 64, 1e-6, "float32", ()))
    out.append(case("rmsnorm_fp16", 2, 16, 256, 1e-6, "float16", ("fp16 reduction precision",)))
    out.append(case("rmsnorm_non_pow2", 1, 8, 96, 1e-6, "float32", ("non-power-of-two hidden dim",)))
    out.append(case("rmsnorm_tiny_magnitude", 1, 4, 64, 1e-6, "float32", ("very-small input magnitude near eps",)))
    out.append(case("rmsnorm_large_dim", 1, 4, 1024, 1e-6, "float32", ("large hidden-dim reduction precision",)))
    return out


# ---- SwiGLU ----
def _swiglu_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"gate": torch.randn(batch, seq, dim, dtype=t_dtype), "up": torch.randn(batch, seq, dim, dtype=t_dtype)}

        return HoldoutCase(name=name, op="swiglu", inputs_fn=inputs_fn, reference_fn=swiglu_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("swiglu_small", 1, 8, 128, "float32", ()))
    out.append(case("swiglu_fp16", 2, 16, 512, "float16", ("fp16 silu numerics",)))
    out.append(case("swiglu_large_dim", 1, 4, 4096, "float32", ("threadgroup size assumption",)))
    out.append(case("swiglu_extreme_range", 1, 4, 64, "float32", ("silu numeric range",)))
    return out


HOLDOUTS: dict[str, list[HoldoutCase]] = {
    "rope": _rope_cases(),
    "rmsnorm": _rmsnorm_cases(),
    "swiglu": _swiglu_cases(),
}


def cases_for(op: str) -> list[HoldoutCase]:
    return HOLDOUTS[op]
