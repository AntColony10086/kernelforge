"""run: invoke a compiled kernel on input tensors, return output.

Includes the chaos hook (silently_wrong_output) that corrupts outputs
when CHAOS_KERNEL_LAB_MODE is set. This is what the hidden holdout suite
catches.
"""
from __future__ import annotations

import os
import time

import mlx.core as mx
import numpy as np

from kernel_lab.compile_tool import get_kernel


def _maybe_corrupt(outputs: list[mx.array], current_op: str | None) -> list[mx.array]:
    mode = os.environ.get("CHAOS_KERNEL_LAB_MODE", "none")
    op_filter = os.environ.get("CHAOS_KERNEL_LAB_OP_FILTER", "")
    if mode == "none" or (op_filter and op_filter != (current_op or "")):
        return outputs
    if mode == "silently_wrong_output":
        corrupted = []
        for o in outputs:
            arr = np.array(o)
            corrupted.append(mx.array((arr * 1.013 + 0.007).astype(arr.dtype)))
        return corrupted
    return outputs


def run_kernel(
    *,
    handle: str,
    inputs: list[mx.array],
    grid: tuple,
    threadgroup: tuple,
    output_shapes: list[tuple],
    output_dtype: str = "float32",
) -> tuple[list[mx.array], float]:
    """Returns (outputs, runtime_ms)."""
    k = get_kernel(handle)
    dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[output_dtype]
    t0 = time.perf_counter()
    outputs = k(
        inputs=inputs,
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=output_shapes,
        output_dtypes=[dtype] * len(output_shapes),
    )
    for o in outputs:
        mx.eval(o)
    rt = (time.perf_counter() - t0) * 1000

    current_op = os.environ.get("CHAOS_KERNEL_LAB_CURRENT_OP")
    outputs = _maybe_corrupt(list(outputs), current_op)
    return list(outputs), rt
