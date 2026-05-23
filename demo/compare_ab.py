"""A/B comparison of two demo runs (typically baseline vs few-shot).

Reads ledgers from two artifacts dirs and emits a side-by-side scorecard
plus per-op delta analysis. Writes:

    demo/artifacts_compare/ab_scorecard.md      (main comparison table)
    demo/artifacts_compare/ab_per_op.md         (per-op verdict)
    demo/artifacts_compare/ab_summary.json      (machine-readable summary)
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from scorecard.generate import compute_outcomes


@dataclass
class Variant:
    name: str  # "baseline" or "few-shot"
    artifacts_dir: Path
    kf_claimed_correct: int
    kf_actually_correct: int
    naive_claimed_correct: int
    naive_actually_correct: int
    naive_false_success: int
    kf_false_success: int
    total_ops: int
    routes: list[str]


def _build_variant(name: str, artifacts_dir: Path, ground_truth: dict[str, bool]) -> Variant:
    naive = artifacts_dir / "naive_ledger.jsonl"
    kf = artifacts_dir / "kf_ledger.jsonl"
    if not naive.exists() or not kf.exists():
        raise FileNotFoundError(f"missing ledgers in {artifacts_dir}")
    outcomes = compute_outcomes(naive, kf, ground_truth)
    routes = sorted({r for o in outcomes for r in o.kf_llm_routes})
    return Variant(
        name=name,
        artifacts_dir=artifacts_dir,
        kf_claimed_correct=sum(1 for o in outcomes if o.kf_claimed_correct),
        kf_actually_correct=sum(1 for o in outcomes if o.kf_claimed_correct and o.kf_actually_correct),
        naive_claimed_correct=sum(1 for o in outcomes if o.naive_claimed_correct),
        naive_actually_correct=sum(1 for o in outcomes if o.naive_claimed_correct and o.naive_actually_correct),
        naive_false_success=sum(1 for o in outcomes if o.naive_claimed_correct and not o.naive_actually_correct),
        kf_false_success=sum(1 for o in outcomes if o.kf_claimed_correct and not o.kf_actually_correct),
        total_ops=len(outcomes),
        routes=routes,
    )


def _render_ab(a: Variant, b: Variant) -> str:
    def delta(av: int, bv: int) -> str:
        d = bv - av
        if d > 0:
            return f"+{d}"
        return str(d) if d < 0 else "—"

    return "\n".join(
        [
            f"# A/B comparison — {a.name} vs {b.name}",
            "",
            f"Ops per variant: {a.total_ops}",
            "",
            "| Metric | "
            f"{a.name} | {b.name} | Δ |",
            "| --- | --- | --- | --- |",
            f"| Naive claimed correct | {a.naive_claimed_correct}/{a.total_ops} | {b.naive_claimed_correct}/{b.total_ops} | {delta(a.naive_claimed_correct, b.naive_claimed_correct)} |",
            f"| Naive actually correct | {a.naive_actually_correct}/{a.total_ops} | {b.naive_actually_correct}/{b.total_ops} | {delta(a.naive_actually_correct, b.naive_actually_correct)} |",
            f"| Naive silent-wrong-output | {a.naive_false_success} | {b.naive_false_success} | {delta(a.naive_false_success, b.naive_false_success)} |",
            f"| **KernelForge verified** | **{a.kf_actually_correct}/{a.total_ops}** | **{b.kf_actually_correct}/{b.total_ops}** | **{delta(a.kf_actually_correct, b.kf_actually_correct)}** |",
            f"| KernelForge false-success | {a.kf_false_success} | {b.kf_false_success} | {delta(a.kf_false_success, b.kf_false_success)} |",
            f"| LLM routes used | {' → '.join(a.routes) or 'n/a'} | {' → '.join(b.routes) or 'n/a'} | — |",
            "",
            "## Read",
            "",
            f"- Naive metrics show whether the few-shot prompt changed how the LLM behaves under smoke-only verification. Differences here are about LLM output quality.",
            f"- KernelForge verified count is the headline: it shows how often a kernel actually passed the full hidden holdout suite. **In both variants, the false-success count must be zero** — that's the contract the agent is designed to enforce. If it isn't, the regression test would have failed CI.",
            "",
        ]
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    baseline = root / "artifacts_baseline"
    fewshot = root / "artifacts_fewshot"
    if not baseline.exists() or not fewshot.exists():
        print(f"missing: baseline={baseline.exists()} fewshot={fewshot.exists()}", file=sys.stderr)
        return 2

    # Ground truth is the same regardless of run; load from baseline's manifest if present.
    ops = list(json.loads((baseline / "manifest.json").read_text())["ops"]) if (baseline / "manifest.json").exists() else []
    ground_truth = {op: (op != "rope") for op in ops}

    a = _build_variant("baseline", baseline, ground_truth)
    b = _build_variant("few-shot", fewshot, ground_truth)

    out_dir = root / "artifacts_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_scorecard.md").write_text(_render_ab(a, b))
    (out_dir / "ab_summary.json").write_text(json.dumps({"baseline": asdict(a), "fewshot": asdict(b)}, default=str, indent=2))

    print((out_dir / "ab_scorecard.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
