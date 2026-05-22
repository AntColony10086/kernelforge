"""Naive baseline:
- 1 LLM call (deepseek-v4-flash, no escalation).
- Compile.
- Run the SINGLE smoke-test input (first holdout case).
- If shape looks right and no crash, declare success with a fake speedup
  computed from the smoke-test alone.
- No ledger, no holdout verification, no honest perf disclosure.

This is the strawman, but it is a FAITHFUL strawman: it uses the same
DeepSeek + the same first-iteration prompt as KernelForge. The only
difference is what it verifies.
"""
from __future__ import annotations

from dataclasses import dataclass

from kernel_lab.compile_tool import CompileError, compile_kernel
from kernelforge.holdouts import cases_for
from kernelforge.llm_client import LLMClient
from kernelforge.prompts import KernelOutput


@dataclass
class NaiveResult:
    op: str
    claimed_correct: bool
    claimed_speedup: float | None
    kernel: KernelOutput | None
    error: str | None


_REFERENCE_SIGS = {
    "rope": "def rope(x: Tensor, *, base: float = 10000.0) -> Tensor: ...",
    "rmsnorm": "def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor: ...",
    "swiglu": "def swiglu(gate: Tensor, up: Tensor) -> Tensor: ...",
}


def _input_names(op: str) -> list[str]:
    return {"rope": ["x"], "rmsnorm": ["x", "weight"], "swiglu": ["gate", "up"]}[op]


async def naive_run(op: str, llm: LLMClient) -> NaiveResult:
    try:
        kernel = await llm.generate_kernel(
            op=op,
            reference_signature=_REFERENCE_SIGS[op],
            previous_diff=None,
            escalate=False,
        )
    except Exception as exc:  # noqa: BLE001
        return NaiveResult(op=op, claimed_correct=False, claimed_speedup=None, kernel=None, error=f"llm_error: {exc}")

    try:
        compile_kernel(
            name=f"naive_{op}",
            source=kernel.source,
            grid=kernel.grid,
            threadgroup=kernel.threadgroup,
            input_names=_input_names(op),
            output_names=["out"],
        )
    except CompileError as exc:
        return NaiveResult(op=op, claimed_correct=False, claimed_speedup=None, kernel=kernel, error=str(exc))

    # Smoke "verification" - naive trusts the compile + presence of holdout 0 inputs.
    # No actual run here. This is the strawman behavior we critique.
    smoke_case = cases_for(op)[0]
    try:
        _ = smoke_case.inputs_fn()
    except Exception as exc:  # noqa: BLE001
        return NaiveResult(op=op, claimed_correct=False, claimed_speedup=None, kernel=kernel, error=str(exc))

    return NaiveResult(op=op, claimed_correct=True, claimed_speedup=1.4, kernel=kernel, error=None)
