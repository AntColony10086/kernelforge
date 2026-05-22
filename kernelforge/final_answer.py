"""Renders the final answer from a KernelLedger.

Hard contract: the renderer can ONLY make a positive correctness claim
for an op when the latest ledger entry for that op is VERIFIED_CORRECT
or PERF_MEASURED. There is no LLM call in this path. There is no other
way to introduce a "verified correct" claim into the output text.
"""
from __future__ import annotations

from kernelforge.ledger import KernelLedger, LedgerEntry, LedgerState


_GOOD_STATES = {LedgerState.VERIFIED_CORRECT, LedgerState.PERF_MEASURED}


def render_final_answer(ledger: KernelLedger, *, op: str | None = None) -> str:
    if op is not None:
        entry = ledger.latest(op)
        if entry is None:
            return f"{op}: no attempt made."
        return _render_one(entry)

    lines: list[str] = ["Run summary:"]
    ops_seen: set[str] = set()
    for entry in ledger.all_entries():
        ops_seen.add(entry.op)
    for op_name in sorted(ops_seen):
        e = ledger.latest(op_name)
        lines.append("  " + _render_one(e))
    return "\n".join(lines)


import re

_FORBIDDEN_RE = re.compile(r"verified[\s_]+correct", re.IGNORECASE)


def _sanitize(text: str | None) -> str:
    """Redact the forbidden claim 'verified correct' from any text we splice
    into the rendered output. Defense in depth: error strings, LLM-echoed
    text, or any user-provided field could otherwise leak the forbidden
    claim.

    Matches: "verified correct", "Verified Correct", "VERIFIED CORRECT",
    "verified_correct", "verified  correct", "verified\tcorrect", etc.
    """
    if not text:
        return ""
    return _FORBIDDEN_RE.sub("verified <REDACTED>", text)


def _render_one(entry: LedgerEntry) -> str:
    op = entry.op
    iters = f"iteration {entry.iteration} ({entry.llm_route})"
    state = entry.state

    if state in _GOOD_STATES:
        verify = entry.verify_report or {}
        pass_count = verify.get("pass")
        fail_count = verify.get("fail")
        perf = entry.perf_report or {}
        speedups = perf.get("speedups", {})
        sp_parts = [f"{k}={v:.2f}x" for k, v in speedups.items()]
        sp_str = ", ".join(sp_parts) if sp_parts else "no perf measured"
        return f"{op}: verified correct ({pass_count} pass / {fail_count} fail), {iters}; perf: {sp_str}."

    if state == LedgerState.VERIFIED_INCORRECT:
        verify = entry.verify_report or {}
        return f"{op}: verification FAILED - {verify.get('fail', '?')} holdout cases mismatched ({iters})."

    if state == LedgerState.SMOKE_PASSED:
        return f"{op}: smoke passed; holdout suite was not run ({iters})."

    if state == LedgerState.ABANDONED:
        return f"{op}: abandoned ({iters}) - {_sanitize(entry.error) or 'reason not recorded'}."

    return f"{op}: in state '{state.value}' ({iters})."
