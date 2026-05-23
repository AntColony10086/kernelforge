"""Scripted demo run. Produces all static artifacts that Remotion consumes:

- demo/artifacts/naive_ledger.jsonl
- demo/artifacts/kf_ledger.jsonl
- demo/artifacts/scorecard_demo.md
- demo/artifacts/scorecard_readme.md
- demo/artifacts/manifest.json   (timeline metadata for Remotion)
- demo/artifacts/run_summary.txt (text summary from final_answer.render)

This script REQUIRES DEEPSEEK_API_KEY in .env to call LLM. If absent, it
prints a clear error and exits.

Architecture: for each op in [rope, rmsnorm, swiglu]:
  1. Run naive baseline (single shot, smoke-only).
  2. Run KernelForge (iteration loop with holdout verify + escalation).
  3. Apply chaos environment per op (rope gets silently_wrong_output;
     others get clean run).

Then compute scorecard and dump artifacts.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from baselines.naive import naive_run
from kernelforge.agent import run_kernelforge
from kernelforge.final_answer import render_final_answer
from kernelforge.ledger import KernelLedger
from kernelforge.llm_client import LLMClient
from kernelforge.llm_transport import HttpTransport
from scorecard.generate import compute_outcomes
from scorecard.render import render_demo_scorecard, render_readme_scorecard


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
# Full 20-op benchmark suite. Chaos middleware corrupts rope only (the
# silent-wrong-output money shot). Other ops should pass cleanly if the
# LLM produces a correct kernel.
OPS = (
    "rope", "rmsnorm", "layernorm", "swiglu",
    "softmax", "gelu", "silu", "tanh",
    "relu", "sigmoid", "exp", "log", "sqrt", "abs",
    "sum_last", "max_last", "mean_last",
    "elementwise_add", "elementwise_mul", "matmul",
)
GROUND_TRUTH = {op: (op != "rope") for op in OPS}


def _check_env() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set. Add it to .env and source .env, then re-run.", file=sys.stderr)
        sys.exit(2)


def _dump_naive(naive_results: list, path: Path) -> None:
    with path.open("w") as f:
        for r in naive_results:
            d = {
                "op": r.op,
                "iteration": 1,
                "llm_route": "deepseek-v4-flash",
                "claimed_correct": r.claimed_correct,
                "state": "verified_correct" if r.claimed_correct else "abandoned",
                "claimed_speedup": r.claimed_speedup,
                "error": r.error,
            }
            f.write(json.dumps(d) + "\n")


def _dump_kf(led_by_op: dict[str, KernelLedger], path: Path) -> None:
    with path.open("w") as f:
        for op, led in led_by_op.items():
            for e in led.all_entries():
                d = {
                    "op": e.op,
                    "iteration": e.iteration,
                    "state": e.state.value,
                    "llm_route": e.llm_route,
                    "verify_report": e.verify_report,
                    "perf_report": e.perf_report,
                    "error": e.error,
                    "timestamp_ms": e.timestamp_ms,
                }
                f.write(json.dumps(d) + "\n")


def _apply_chaos_for_op(op: str) -> None:
    """Set environment so that the rope op gets corrupted output (the demo
    money shot) and other ops run clean."""
    if op == "rope":
        os.environ["CHAOS_KERNEL_LAB_MODE"] = "silently_wrong_output"
        os.environ["CHAOS_KERNEL_LAB_OP_FILTER"] = "rope"
    else:
        os.environ["CHAOS_KERNEL_LAB_MODE"] = "none"
        os.environ["CHAOS_KERNEL_LAB_OP_FILTER"] = ""
    os.environ["CHAOS_KERNEL_LAB_CURRENT_OP"] = op


async def main() -> int:
    _check_env()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    transport = HttpTransport()
    llm = LLMClient(transport=transport)

    naive_results = []
    kf_ledgers: dict[str, KernelLedger] = {}

    for op in OPS:
        _apply_chaos_for_op(op)
        print(f"==> naive {op}")
        naive_results.append(await naive_run(op, llm))
        print(f"==> kernelforge {op}")
        kf_ledgers[op] = await run_kernelforge(op, llm, profile="demo")

    naive_path = ARTIFACTS / "naive_ledger.jsonl"
    kf_path = ARTIFACTS / "kf_ledger.jsonl"
    _dump_naive(naive_results, naive_path)
    _dump_kf(kf_ledgers, kf_path)

    outcomes = compute_outcomes(naive_path, kf_path, GROUND_TRUTH)
    (ARTIFACTS / "scorecard_demo.md").write_text(render_demo_scorecard(outcomes))
    (ARTIFACTS / "scorecard_readme.md").write_text(render_readme_scorecard(outcomes))

    # Manifest: a small json that Remotion reads to build the timeline.
    manifest = {
        "ops": list(OPS),
        "ground_truth": GROUND_TRUTH,
        "outcomes": [
            {
                "op": o.op,
                "naive_claimed_correct": o.naive_claimed_correct,
                "naive_actually_correct": o.naive_actually_correct,
                "kf_claimed_correct": o.kf_claimed_correct,
                "kf_actually_correct": o.kf_actually_correct,
                "kf_iterations": o.kf_iterations,
                "kf_llm_routes": o.kf_llm_routes,
                "kf_speedups": o.kf_speedups,
            }
            for o in outcomes
        ],
    }
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Combined run summary for human inspection.
    summary_lines = ["Naive baseline summary:"]
    for r in naive_results:
        summary_lines.append(f"  {r.op}: claimed_correct={r.claimed_correct} error={r.error!r}")
    summary_lines.append("")
    summary_lines.append("KernelForge ledger summary:")
    combined = KernelLedger()
    for led in kf_ledgers.values():
        for e in led.all_entries():
            combined._entries.append(e)  # noqa: SLF001 — internal aggregation for summary only
    summary_lines.append(render_final_answer(combined))
    (ARTIFACTS / "run_summary.txt").write_text("\n".join(summary_lines))

    print(f"\nArtifacts written to {ARTIFACTS}/")
    print(f"  scorecard_demo.md:\n{(ARTIFACTS / 'scorecard_demo.md').read_text()}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
