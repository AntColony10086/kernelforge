"""Scorecard: 4-row demo table + detailed README scorecard.

Reads:
- ledger JSONL files for naive and kernelforge runs.
- bench reports (per-op, only for KernelForge's verified-correct kernels).
- ground truth: which ops the chaos scenario corrupted (so naive's
  claim of correctness can be evaluated against reality).

Emits:
- demo_scorecard.md  (4-row table, shown on screen for 5 sec in the demo).
- readme_scorecard.md  (detailed per-op breakdown for the README).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpOutcome:
    op: str
    naive_claimed_correct: bool
    naive_actually_correct: bool
    kf_claimed_correct: bool
    kf_actually_correct: bool
    kf_iterations: int
    kf_llm_routes: list[str]
    kf_speedups: dict[str, float]


def load_ledger(path: Path) -> dict[str, list[dict]]:
    entries: dict[str, list[dict]] = {}
    if not path.exists():
        return entries
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        entries.setdefault(e["op"], []).append(e)
    return entries


def compute_outcomes(naive_ledger: Path, kf_ledger: Path, ground_truth: dict[str, bool]) -> list[OpOutcome]:
    """Compute per-op outcomes.

    naive_actually_correct prefers the `actually_correct_holdouts` field on
    the naive ledger entry (a post-hoc holdout check that measures truth
    independently of naive's claim). Falls back to `ground_truth[op]` when
    the field is absent — for ledgers from older recorder versions.
    """
    naive = load_ledger(naive_ledger)
    kf = load_ledger(kf_ledger)
    ops = sorted(set(naive) | set(kf))
    out: list[OpOutcome] = []
    for op in ops:
        naive_last = naive.get(op, [])[-1] if op in naive else {}
        kf_last = kf.get(op, [])[-1] if op in kf else {}
        kf_runs = kf.get(op, [])
        naive_claimed = bool(naive_last.get("claimed_correct")) or naive_last.get("state") == "verified_correct"
        kf_claimed = kf_last.get("state") in {"verified_correct", "perf_measured"}
        # Prefer measured truth; fall back to the ground-truth assumption.
        if "actually_correct_holdouts" in naive_last:
            naive_truth = bool(naive_last["actually_correct_holdouts"])
        else:
            naive_truth = ground_truth.get(op, True)
        out.append(
            OpOutcome(
                op=op,
                naive_claimed_correct=naive_claimed,
                naive_actually_correct=naive_truth,
                kf_claimed_correct=kf_claimed,
                kf_actually_correct=kf_claimed,  # KernelForge only claims after holdouts pass
                kf_iterations=max((int(e.get("iteration", 1)) for e in kf_runs), default=0),
                kf_llm_routes=sorted({e.get("llm_route", "") for e in kf_runs if e.get("llm_route")}),
                kf_speedups=(kf_last.get("perf_report") or {}).get("speedups", {}) or {},
            )
        )
    return out
