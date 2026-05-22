"""compile: take Metal source + grid/threadgroup -> compiled handle.

Uses mlx.fast.metal_kernel. We CACHE compiled kernels by source hash to
avoid recompiling identical kernels across iterations.
"""
from __future__ import annotations

import hashlib

import mlx.core as mx


class CompileError(Exception):
    def __init__(self, log: str) -> None:
        super().__init__(log)
        self.log = log


_CACHE: dict[str, object] = {}


def _hash(source: str, grid: tuple, threadgroup: tuple) -> str:
    return hashlib.sha256(f"{source}|{grid}|{threadgroup}".encode()).hexdigest()


def compile_kernel(
    *,
    name: str,
    source: str,
    grid: tuple,
    threadgroup: tuple,
    input_names: list[str],
    output_names: list[str],
) -> str:
    """Returns a string handle (hash) that run_tool can use to invoke the kernel."""
    h = _hash(source, grid, threadgroup)
    if h in _CACHE:
        return h
    try:
        k = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            atomic_outputs=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise CompileError(log=str(exc)) from exc
    _CACHE[h] = k
    return h


def get_kernel(handle: str):
    return _CACHE[handle]


def clear_cache() -> None:
    _CACHE.clear()
