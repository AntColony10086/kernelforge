"""KernelForge agent: full iteration loop with hidden-holdout verification
and cost-aware Flash -> Pro escalation.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Callable

from kernel_lab.compile_tool import CompileError, compile_kernel
from kernel_lab.run_tool import run_kernel
from kernel_lab.verify_tool import _to_mlx_inputs, verify_kernel
from kernelforge.holdouts import cases_for
from kernelforge.ledger import KernelLedger, LedgerState
from kernelforge.llm_client import LLMClient, LLMRouteChoice
from kernelforge.prompts import KernelOutput


from kernelforge.op_registry import REGISTRY


def _iteration_config(profile: str = "demo") -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "iteration.toml"
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg["profiles"][profile]


def _ref_sig(op: str) -> str:
    return REGISTRY[op].reference_signature


def _output_shape_fn(op: str) -> Callable[[dict], list[tuple]]:
    return REGISTRY[op].output_shape


def _input_names(op: str) -> list[str]:
    return list(REGISTRY[op].input_names)


async def run_kernelforge(op: str, llm: LLMClient, *, profile: str = "demo") -> KernelLedger:
    cfg = _iteration_config(profile)
    max_iter = int(cfg["max_iterations"])
    escalate_after = int(cfg["escalate_after_iteration"])

    led = KernelLedger()
    previous_diff: dict | None = None

    for iteration in range(1, max_iter + 1):
        escalate = iteration > escalate_after
        route = "deepseek-coder" if escalate else "deepseek-v4-flash"
        led.start(op, iteration=iteration, llm_route=route)

        # Generate kernel via LLM
        try:
            kernel = await llm.generate_kernel(
                op=op,
                reference_signature=_ref_sig(op),
                previous_diff=previous_diff,
                escalate=escalate,
            )
            led.advance(op, LedgerState.GENERATED, kernel_source=kernel.source)
        except LLMRouteChoice.ParseError as exc:
            previous_diff = {"failing_case": "llm_parse_error", "max_abs_diff": "n/a", "hints": [str(exc)[:300]]}
            led.advance(op, LedgerState.ABANDONED, error=f"llm_parse_error: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            led.advance(op, LedgerState.ABANDONED, error=f"llm_error: {exc}")
            return led

        # Compile
        try:
            handle = compile_kernel(
                name=f"{op}_iter{iteration}",
                source=kernel.source,
                grid=kernel.grid,
                threadgroup=kernel.threadgroup,
                input_names=_input_names(op),
                output_names=["out"],
            )
            led.advance(op, LedgerState.COMPILED)
        except CompileError as exc:
            previous_diff = {
                "failing_case": "compile_error",
                "max_abs_diff": "n/a",
                "hints": [str(exc)[:300]],
            }
            led.advance(op, LedgerState.ABANDONED, error=f"compile_error: {exc}")
            continue

        # Smoke run on case 0
        smoke = cases_for(op)[0]
        try:
            os.environ["CHAOS_KERNEL_LAB_CURRENT_OP"] = op
            inputs = smoke.inputs_fn()
            mlx_inputs = _to_mlx_inputs(inputs, op)
            run_kernel(
                handle=handle,
                inputs=mlx_inputs,
                grid=kernel.grid,
                threadgroup=kernel.threadgroup,
                output_shapes=_output_shape_fn(op)(inputs),
                output_dtype=smoke.dtype,
            )
            led.advance(op, LedgerState.SMOKE_PASSED)
        except Exception as exc:  # noqa: BLE001
            previous_diff = {
                "failing_case": "smoke_runtime_error",
                "max_abs_diff": "n/a",
                "hints": [str(exc)[:300]],
            }
            led.advance(op, LedgerState.ABANDONED, error=f"smoke_runtime: {exc}")
            continue

        # Full holdout verification
        os.environ["CHAOS_KERNEL_LAB_CURRENT_OP"] = op
        report = verify_kernel(
            handle=handle,
            op=op,
            grid=kernel.grid,
            threadgroup=kernel.threadgroup,
            output_shape_fn=_output_shape_fn(op),
        )

        if report.fail_count == 0:
            led.advance(op, LedgerState.VERIFIED_CORRECT, verify_report=report.to_dict())
            return led

        # Incorrect — pick the worst failing case for diff feedback.
        failing = [c for c in report.cases if not c.passed]
        worst = max(failing, key=lambda c: c.max_abs_diff if c.max_abs_diff != float("inf") else 1e18)
        previous_diff = {
            "failing_case": worst.name,
            "max_abs_diff": worst.max_abs_diff if worst.max_abs_diff != float("inf") else "runtime_error",
            "hints": worst.hints,
        }
        led.advance(op, LedgerState.VERIFIED_INCORRECT, verify_report=report.to_dict())
        # Loop continues for next iteration

    # Exhausted iterations without convergence — finalize the last attempt as abandoned.
    last = led.latest(op)
    if last is not None and last.state == LedgerState.VERIFIED_INCORRECT:
        led.advance(op, LedgerState.ABANDONED, error="iteration cap reached without verification")
    return led
