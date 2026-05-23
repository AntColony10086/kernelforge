"""Unified op registry: single source of truth for how each op plugs into
the agent loop, the naive baseline, and the verifier. Adding a new op
requires only a new entry here + a reference impl + holdouts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import numpy as np
import torch


def _t2m(t: torch.Tensor) -> mx.array:
    """torch → mlx via numpy (float32 promotion when needed)."""
    if t.dtype != torch.float32:
        return mx.array(t.detach().cpu().to(torch.float32).numpy())
    return mx.array(t.detach().cpu().numpy())


@dataclass(frozen=True)
class OpDef:
    name: str
    input_names: list[str]
    reference_signature: str  # shown to the LLM in the prompt
    extract_inputs: Callable[[dict], list[mx.array]]  # holdout dict → MLX kernel inputs
    output_shape: Callable[[dict], list[tuple]]  # holdout dict → output tensor shapes
    call_reference: Callable[[object, dict], torch.Tensor]  # (case, holdout_dict) → ref tensor


# Module-level reference functions used inside OpDef closures (avoids circular imports
# vs importing references at the top of holdouts.py — the holdouts module already
# imports the refs and stores them on each HoldoutCase).
def _call_via_case(case, inputs: dict) -> torch.Tensor:
    """Default: dispatch to the reference function stored on the holdout case,
    using the inputs whose keys the reference function expects.
    """
    op = case.op
    fn = case.reference_fn
    if op == "rope":
        return fn(inputs["x"], base=inputs["base"])
    if op == "rmsnorm":
        return fn(inputs["x"], inputs["weight"], inputs["eps"])
    if op == "layernorm":
        return fn(inputs["x"], inputs["weight"], inputs["bias"], inputs["eps"])
    if op == "swiglu":
        return fn(inputs["gate"], inputs["up"])
    if op in {"elementwise_add", "elementwise_mul", "matmul"}:
        return fn(inputs["a"], inputs["b"])
    if op in {"silu", "tanh", "relu", "sigmoid", "exp", "log", "sqrt", "abs", "gelu", "softmax", "sum_last", "max_last", "mean_last"}:
        return fn(inputs["x"])
    raise ValueError(f"unknown op {op}")


def _x_inputs(i): return [_t2m(i["x"])]
def _x_shape(i): return [tuple(i["x"].shape)]
def _ab_inputs(i): return [_t2m(i["a"]), _t2m(i["b"])]
def _ab_shape_same_a(i): return [tuple(i["a"].shape)]


REGISTRY: dict[str, OpDef] = {
    "rope": OpDef(
        name="rope",
        input_names=["x"],
        reference_signature="def rope(x: Tensor, *, base: float = 10000.0) -> Tensor: ...  # split-half layout",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "rmsnorm": OpDef(
        name="rmsnorm",
        input_names=["x", "weight"],
        reference_signature="def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor: ...",
        extract_inputs=lambda i: [_t2m(i["x"]), _t2m(i["weight"])],
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "layernorm": OpDef(
        name="layernorm",
        input_names=["x", "weight", "bias"],
        reference_signature="def layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor: ...",
        extract_inputs=lambda i: [_t2m(i["x"]), _t2m(i["weight"]), _t2m(i["bias"])],
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "swiglu": OpDef(
        name="swiglu",
        input_names=["gate", "up"],
        reference_signature="def swiglu(gate: Tensor, up: Tensor) -> Tensor: ...  # SiLU(gate) * up",
        extract_inputs=lambda i: [_t2m(i["gate"]), _t2m(i["up"])],
        output_shape=lambda i: [tuple(i["gate"].shape)],
        call_reference=_call_via_case,
    ),
    "softmax": OpDef(
        name="softmax",
        input_names=["x"],
        reference_signature="def softmax(x: Tensor) -> Tensor: ...  # along last dim, numerically stable",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "gelu": OpDef(
        name="gelu",
        input_names=["x"],
        reference_signature="def gelu(x: Tensor) -> Tensor: ...  # exact (erf) variant, not tanh-approx",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "silu": OpDef(
        name="silu",
        input_names=["x"],
        reference_signature="def silu(x: Tensor) -> Tensor: ...  # x * sigmoid(x)",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "tanh": OpDef(
        name="tanh",
        input_names=["x"],
        reference_signature="def tanh(x: Tensor) -> Tensor: ...",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "relu": OpDef(
        name="relu",
        input_names=["x"],
        reference_signature="def relu(x: Tensor) -> Tensor: ...  # max(x, 0)",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "sigmoid": OpDef(
        name="sigmoid",
        input_names=["x"],
        reference_signature="def sigmoid(x: Tensor) -> Tensor: ...  # 1 / (1 + exp(-x))",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "exp": OpDef(
        name="exp",
        input_names=["x"],
        reference_signature="def exp(x: Tensor) -> Tensor: ...",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "log": OpDef(
        name="log",
        input_names=["x"],
        reference_signature="def log(x: Tensor) -> Tensor: ...  # natural log, x > 0 enforced by caller",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "sqrt": OpDef(
        name="sqrt",
        input_names=["x"],
        reference_signature="def sqrt(x: Tensor) -> Tensor: ...  # x >= 0 enforced by caller",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "abs": OpDef(
        name="abs",
        input_names=["x"],
        reference_signature="def abs(x: Tensor) -> Tensor: ...",
        extract_inputs=_x_inputs,
        output_shape=_x_shape,
        call_reference=_call_via_case,
    ),
    "sum_last": OpDef(
        name="sum_last",
        input_names=["x"],
        reference_signature="def sum_last(x: Tensor) -> Tensor: ...  # sum along last dim, keepdim=True",
        extract_inputs=_x_inputs,
        output_shape=lambda i: [tuple(list(i["x"].shape[:-1]) + [1])],
        call_reference=_call_via_case,
    ),
    "max_last": OpDef(
        name="max_last",
        input_names=["x"],
        reference_signature="def max_last(x: Tensor) -> Tensor: ...  # max along last dim, keepdim=True",
        extract_inputs=_x_inputs,
        output_shape=lambda i: [tuple(list(i["x"].shape[:-1]) + [1])],
        call_reference=_call_via_case,
    ),
    "mean_last": OpDef(
        name="mean_last",
        input_names=["x"],
        reference_signature="def mean_last(x: Tensor) -> Tensor: ...  # mean along last dim, keepdim=True",
        extract_inputs=_x_inputs,
        output_shape=lambda i: [tuple(list(i["x"].shape[:-1]) + [1])],
        call_reference=_call_via_case,
    ),
    "elementwise_add": OpDef(
        name="elementwise_add",
        input_names=["a", "b"],
        reference_signature="def elementwise_add(a: Tensor, b: Tensor) -> Tensor: ...  # a + b, same shape",
        extract_inputs=_ab_inputs,
        output_shape=_ab_shape_same_a,
        call_reference=_call_via_case,
    ),
    "elementwise_mul": OpDef(
        name="elementwise_mul",
        input_names=["a", "b"],
        reference_signature="def elementwise_mul(a: Tensor, b: Tensor) -> Tensor: ...  # a * b, same shape",
        extract_inputs=_ab_inputs,
        output_shape=_ab_shape_same_a,
        call_reference=_call_via_case,
    ),
    "matmul": OpDef(
        name="matmul",
        input_names=["a", "b"],
        reference_signature="def matmul(a: Tensor[M, K], b: Tensor[K, N]) -> Tensor[M, N]: ...  # 2D matmul, a @ b",
        extract_inputs=_ab_inputs,
        output_shape=lambda i: [(int(i["a"].shape[0]), int(i["b"].shape[1]))],
        call_reference=_call_via_case,
    ),
}


def all_ops() -> list[str]:
    return list(REGISTRY.keys())
