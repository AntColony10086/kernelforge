"""verify: run kernel over holdout suite, compare to reference, emit
structured diff for failures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import mlx.core as mx
import numpy as np
import torch

from kernel_lab.run_tool import run_kernel
from kernelforge.holdouts import HoldoutCase, cases_for


@dataclass
class CaseResult:
    name: str
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    hints: list[str]


@dataclass
class VerifyReport:
    op: str
    cases: list[CaseResult]
    pass_count: int
    fail_count: int

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "pass": self.pass_count,
            "fail": self.fail_count,
            "cases": [asdict(c) for c in self.cases],
        }


def verify_kernel(
    *,
    handle: str,
    op: str,
    grid: tuple,
    threadgroup: tuple,
    output_shape_fn: Callable[[dict], list[tuple]],
) -> VerifyReport:
    """Run the holdout suite for `op` against the compiled kernel."""
    results: list[CaseResult] = []
    for case in cases_for(op):
        inputs = case.inputs_fn()
        ref_out = _call_reference(case, inputs)
        try:
            mlx_inputs = _to_mlx_inputs(inputs, op)
            outputs, _ = run_kernel(
                handle=handle,
                inputs=mlx_inputs,
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=output_shape_fn(inputs),
                output_dtype=case.dtype,
            )
            actual = _to_torch(outputs[0])
            max_abs = float((actual.float() - ref_out.float()).abs().max().item())
            denom = float(ref_out.float().abs().max().item())
            max_rel = max_abs / (denom + 1e-12)
            passed = max_abs <= case.tolerance_abs and max_rel <= case.tolerance_rel
            results.append(
                CaseResult(
                    name=case.name,
                    passed=passed,
                    max_abs_diff=max_abs,
                    max_rel_diff=max_rel,
                    hints=list(case.suspected_bug_hints),
                )
            )
        except Exception as exc:  # A runtime crash on a holdout is a FAIL, not an abort.
            results.append(
                CaseResult(
                    name=case.name,
                    passed=False,
                    max_abs_diff=float("inf"),
                    max_rel_diff=float("inf"),
                    hints=[str(exc)[:300], *case.suspected_bug_hints],
                )
            )

    pc = sum(1 for r in results if r.passed)
    fc = sum(1 for r in results if not r.passed)
    return VerifyReport(op=op, cases=results, pass_count=pc, fail_count=fc)


from kernelforge.op_registry import REGISTRY


def _call_reference(case: HoldoutCase, inputs: dict) -> torch.Tensor:
    return REGISTRY[case.op].call_reference(case, inputs)


def _to_mlx_inputs(inputs: dict, op: str) -> list[mx.array]:
    return REGISTRY[op].extract_inputs(inputs)


def _to_torch(arr: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(arr))
