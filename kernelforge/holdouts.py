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

from references.abs import abs_ref
from references.elementwise_add import elementwise_add_ref
from references.elementwise_mul import elementwise_mul_ref
from references.exp import exp_ref
from references.gelu import gelu_ref
from references.layernorm import layernorm_ref
from references.log import log_ref
from references.matmul import matmul_ref
from references.max_last import max_last_ref
from references.mean_last import mean_last_ref
from references.relu import relu_ref
from references.rmsnorm import rmsnorm_ref
from references.rope import rope_ref
from references.sigmoid import sigmoid_ref
from references.silu import silu_ref
from references.softmax import softmax_ref
from references.sqrt import sqrt_ref
from references.sum_last import sum_last_ref
from references.swiglu import swiglu_ref
from references.tanh import tanh_ref


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


# ---- Softmax ----
def _softmax_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"x": torch.randn(batch, seq, dim, dtype=t_dtype)}

        return HoldoutCase(name=name, op="softmax", inputs_fn=inputs_fn, reference_fn=softmax_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("softmax_small", 1, 8, 64, "float32", ()))
    out.append(case("softmax_fp16", 2, 16, 256, "float16", ("fp16 exp overflow",)))
    out.append(case("softmax_large_dim", 1, 4, 4096, "float32", ("large-dim reduction precision",)))
    out.append(case("softmax_large_magnitude", 1, 4, 128, "float32", ("missing max-subtraction trick", "exp overflow without numerical stability")))
    out.append(case("softmax_negative_range", 1, 4, 128, "float32", ("sign assumptions",)))
    return out


# ---- GELU ----
def _gelu_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"x": torch.randn(batch, seq, dim, dtype=t_dtype)}

        return HoldoutCase(name=name, op="gelu", inputs_fn=inputs_fn, reference_fn=gelu_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("gelu_small", 1, 8, 128, "float32", ()))
    out.append(case("gelu_fp16", 2, 16, 512, "float16", ("fp16 erf precision",)))
    out.append(case("gelu_extreme_range", 1, 4, 64, "float32", ("erf vs tanh-approx confusion", "saturated input range")))
    out.append(case("gelu_negative_only", 1, 4, 128, "float32", ("negative input branch",)))
    return out


# ---- LayerNorm ----
def _layernorm_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {
                "x": torch.randn(batch, seq, dim, dtype=t_dtype),
                "weight": torch.ones(dim, dtype=t_dtype),
                "bias": torch.zeros(dim, dtype=t_dtype),
                "eps": 1e-5,
            }

        return HoldoutCase(name=name, op="layernorm", inputs_fn=inputs_fn, reference_fn=layernorm_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("layernorm_small", 1, 8, 64, "float32", ()))
    out.append(case("layernorm_fp16", 2, 16, 256, "float16", ("fp16 mean/var precision",)))
    out.append(case("layernorm_non_pow2", 1, 8, 96, "float32", ("non-power-of-two hidden dim",)))
    out.append(case("layernorm_vs_rmsnorm", 1, 4, 128, "float32", ("missing mean subtraction (RMSNorm confusion)", "missing bias term")))
    out.append(case("layernorm_large_dim", 1, 4, 1024, "float32", ("large-dim reduction precision",)))
    return out


# ---- SiLU ----
def _silu_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"x": torch.randn(batch, seq, dim, dtype=t_dtype)}

        return HoldoutCase(name=name, op="silu", inputs_fn=inputs_fn, reference_fn=silu_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("silu_small", 1, 8, 128, "float32", ()))
    out.append(case("silu_fp16", 2, 16, 512, "float16", ("fp16 sigmoid precision",)))
    out.append(case("silu_extreme_range", 1, 4, 64, "float32", ("sigmoid saturation",)))
    out.append(case("silu_negative_only", 1, 4, 128, "float32", ("negative input branch",)))
    return out


# ---- Tanh ----
def _tanh_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"x": torch.randn(batch, seq, dim, dtype=t_dtype)}

        return HoldoutCase(name=name, op="tanh", inputs_fn=inputs_fn, reference_fn=tanh_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("tanh_small", 1, 8, 64, "float32", ()))
    out.append(case("tanh_fp16", 2, 16, 256, "float16", ("fp16 tanh precision",)))
    out.append(case("tanh_extreme_range", 1, 4, 64, "float32", ("tanh saturation",)))
    out.append(case("tanh_large_dim", 1, 4, 4096, "float32", ("threadgroup assumption",)))
    return out


def _elementwise_unary_cases(op_name: str, ref_fn, *, positive: bool = False) -> list[HoldoutCase]:
    """Standard 4-case suite for a single-input elementwise op."""
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            t = torch.randn(batch, seq, dim, dtype=t_dtype)
            if positive:
                t = t.abs() + 1e-3
            return {"x": t}

        return HoldoutCase(name=name, op=op_name, inputs_fn=inputs_fn, reference_fn=ref_fn, dtype=dtype, suspected_bug_hints=hints)

    out.append(case(f"{op_name}_small", 1, 8, 64, "float32", ()))
    out.append(case(f"{op_name}_fp16", 2, 16, 256, "float16", ("fp16 precision",)))
    out.append(case(f"{op_name}_large_dim", 1, 4, 4096, "float32", ("threadgroup assumption",)))
    out.append(case(f"{op_name}_non_pow2", 1, 8, 96, "float32", ("non-power-of-two dim",)))
    return out


def _elementwise_binary_cases(op_name: str, ref_fn) -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"a": torch.randn(batch, seq, dim, dtype=t_dtype), "b": torch.randn(batch, seq, dim, dtype=t_dtype)}

        return HoldoutCase(name=name, op=op_name, inputs_fn=inputs_fn, reference_fn=ref_fn, dtype=dtype, suspected_bug_hints=hints)

    out.append(case(f"{op_name}_small", 1, 8, 64, "float32", ()))
    out.append(case(f"{op_name}_fp16", 2, 16, 256, "float16", ("fp16 precision",)))
    out.append(case(f"{op_name}_large_dim", 1, 4, 4096, "float32", ("threadgroup assumption",)))
    out.append(case(f"{op_name}_non_pow2", 1, 8, 96, "float32", ("non-power-of-two dim",)))
    return out


def _reduction_last_cases(op_name: str, ref_fn) -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, batch, seq, dim))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"x": torch.randn(batch, seq, dim, dtype=t_dtype)}

        return HoldoutCase(name=name, op=op_name, inputs_fn=inputs_fn, reference_fn=ref_fn, dtype=dtype, suspected_bug_hints=hints)

    out.append(case(f"{op_name}_small", 1, 8, 64, "float32", ()))
    out.append(case(f"{op_name}_fp16", 2, 16, 256, "float16", ("fp16 reduction precision",)))
    out.append(case(f"{op_name}_large_dim", 1, 4, 4096, "float32", ("reduction-over-large-dim precision",)))
    out.append(case(f"{op_name}_non_pow2", 1, 8, 96, "float32", ("non-power-of-two reduction",)))
    return out


def _matmul_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, m: int, k: int, n: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(_det_seed(name, m, k, n))
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"a": torch.randn(m, k, dtype=t_dtype), "b": torch.randn(k, n, dtype=t_dtype)}

        return HoldoutCase(name=name, op="matmul", inputs_fn=inputs_fn, reference_fn=matmul_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("matmul_small", 16, 16, 16, "float32", ()))
    out.append(case("matmul_fp16", 32, 32, 32, "float16", ("fp16 accumulation precision",)))
    out.append(case("matmul_rect", 16, 64, 8, "float32", ("non-square shapes",)))
    out.append(case("matmul_non_pow2", 12, 17, 13, "float32", ("non-power-of-two K, M, or N",)))
    return out


HOLDOUTS: dict[str, list[HoldoutCase]] = {
    "rope": _rope_cases(),
    "rmsnorm": _rmsnorm_cases(),
    "swiglu": _swiglu_cases(),
    "softmax": _softmax_cases(),
    "gelu": _gelu_cases(),
    "layernorm": _layernorm_cases(),
    "silu": _silu_cases(),
    "tanh": _tanh_cases(),
    "relu": _elementwise_unary_cases("relu", relu_ref),
    "sigmoid": _elementwise_unary_cases("sigmoid", sigmoid_ref),
    "exp": _elementwise_unary_cases("exp", exp_ref),
    "log": _elementwise_unary_cases("log", log_ref, positive=True),
    "sqrt": _elementwise_unary_cases("sqrt", sqrt_ref, positive=True),
    "abs": _elementwise_unary_cases("abs", abs_ref),
    "sum_last": _reduction_last_cases("sum_last", sum_last_ref),
    "max_last": _reduction_last_cases("max_last", max_last_ref),
    "mean_last": _reduction_last_cases("mean_last", mean_last_ref),
    "elementwise_add": _elementwise_binary_cases("elementwise_add", elementwise_add_ref),
    "elementwise_mul": _elementwise_binary_cases("elementwise_mul", elementwise_mul_ref),
    "matmul": _matmul_cases(),
}


def cases_for(op: str) -> list[HoldoutCase]:
    return HOLDOUTS[op]
