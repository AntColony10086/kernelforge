"""KernelLedger - single source of truth for kernel state.

Monotonic state transitions per (op, iteration). The final answer renderer
reads from this ledger; the LLM cannot inject completion claims outside
the ledger state.
"""
from __future__ import annotations

import enum
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


class LedgerState(str, enum.Enum):
    ATTEMPTED = "attempted"
    GENERATED = "generated"
    COMPILED = "compiled"
    SMOKE_PASSED = "smoke_passed"
    VERIFIED_CORRECT = "verified_correct"
    VERIFIED_INCORRECT = "verified_incorrect"
    PERF_MEASURED = "perf_measured"
    ABANDONED = "abandoned"


_FORWARD: dict[LedgerState, frozenset[LedgerState]] = {
    LedgerState.ATTEMPTED: frozenset({LedgerState.GENERATED, LedgerState.ABANDONED}),
    LedgerState.GENERATED: frozenset({LedgerState.COMPILED, LedgerState.ABANDONED}),
    LedgerState.COMPILED: frozenset({LedgerState.SMOKE_PASSED, LedgerState.ABANDONED}),
    LedgerState.SMOKE_PASSED: frozenset(
        {LedgerState.VERIFIED_CORRECT, LedgerState.VERIFIED_INCORRECT, LedgerState.ABANDONED}
    ),
    LedgerState.VERIFIED_CORRECT: frozenset({LedgerState.PERF_MEASURED}),
    LedgerState.VERIFIED_INCORRECT: frozenset({LedgerState.ABANDONED}),
    LedgerState.PERF_MEASURED: frozenset(),
    LedgerState.ABANDONED: frozenset(),
}


@dataclass
class LedgerEntry:
    op: str
    iteration: int
    state: LedgerState
    llm_route: str
    kernel_source: str | None = None
    verify_report: dict | None = None
    perf_report: dict | None = None
    error: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    timestamp_ms: int = 0


class KernelLedger:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def start(self, op: str, iteration: int, llm_route: str) -> LedgerEntry:
        if iteration > 1:
            prev = self.latest(op)
            if prev is None or prev.state not in (LedgerState.VERIFIED_INCORRECT, LedgerState.ABANDONED):
                raise ValueError(f"cannot start iteration {iteration} of {op} without prior failure")
        entry = LedgerEntry(
            op=op,
            iteration=iteration,
            state=LedgerState.ATTEMPTED,
            llm_route=llm_route,
            timestamp_ms=_now_ms(),
        )
        self._entries.append(entry)
        return entry

    def advance(self, op: str, new_state: LedgerState, **fields) -> LedgerEntry:
        prev = self.latest(op)
        if prev is None:
            raise ValueError(f"no entry for op {op}")
        allowed = _FORWARD[prev.state]
        if new_state not in allowed:
            raise ValueError(f"illegal transition {prev.state.value} -> {new_state.value} for {op}")
        entry = LedgerEntry(
            op=op,
            iteration=prev.iteration,
            state=new_state,
            llm_route=prev.llm_route,
            timestamp_ms=_now_ms(),
        )
        for k, v in fields.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        # Carry forward sticky fields so the latest entry always has full context.
        entry.kernel_source = entry.kernel_source or prev.kernel_source
        entry.verify_report = entry.verify_report or prev.verify_report
        entry.perf_report = entry.perf_report or prev.perf_report
        self._entries.append(entry)
        return entry

    def latest(self, op: str) -> LedgerEntry | None:
        for e in reversed(self._entries):
            if e.op == op:
                return e
        return None

    def all_entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for e in self._entries:
                d = asdict(e)
                d["state"] = e.state.value
                f.write(json.dumps(d) + "\n")


def _now_ms() -> int:
    return int(time.time() * 1000)
