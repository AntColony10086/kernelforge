"""bench: measure kernel vs MLX eager, mx.compile, and mx.fast built-ins.

Honest perf: if our hand-rolled kernel loses to MLX's expert built-in,
we report the loss. Never compare against PyTorch CPU as the main number.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

from kernel_lab.run_tool import run_kernel


@dataclass
class BenchReport:
    op: str
    kernel_ms: float
    baseline_ms: dict[str, float]
    speedups: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "kernel_ms": self.kernel_ms,
            "baseline_ms": self.baseline_ms,
            "speedups": self.speedups,
        }


def _time_call(fn: Callable[[], object], *, warmup: int = 3, iters: int = 30) -> float:
    for _ in range(warmup):
        out = fn()
        if isinstance(out, mx.array):
            mx.eval(out)
    mx.synchronize() if hasattr(mx, "synchronize") else None
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn()
        if isinstance(out, mx.array):
            mx.eval(out)
    if hasattr(mx, "synchronize"):
        mx.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def bench_kernel(
    *,
    handle: str,
    op: str,
    mlx_inputs: list[mx.array],
    grid: tuple,
    threadgroup: tuple,
    output_shapes: list[tuple],
    baselines: dict[str, Callable[[], mx.array]],
) -> BenchReport:
    def run_once():
        outs, _ = run_kernel(
            handle=handle,
            inputs=mlx_inputs,
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=output_shapes,
        )
        return outs[0]

    kernel_ms = _time_call(run_once)
    baseline_ms: dict[str, float] = {}
    speedups: dict[str, float] = {}
    for name, fn in baselines.items():
        ms = _time_call(fn)
        baseline_ms[name] = ms
        speedups[name] = ms / kernel_ms if kernel_ms > 0 else float("inf")
    return BenchReport(op=op, kernel_ms=kernel_ms, baseline_ms=baseline_ms, speedups=speedups)
