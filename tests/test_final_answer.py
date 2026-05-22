"""Final-answer contract: the renderer must NEVER claim correctness without
a verified_correct ledger entry. This test is the regression guard.
"""
from __future__ import annotations

import pytest

from kernelforge.final_answer import render_final_answer
from kernelforge.ledger import KernelLedger, LedgerState


def test_renders_verified_correct():
    led = KernelLedger()
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    led.advance("rope", LedgerState.GENERATED, kernel_source="k")
    led.advance("rope", LedgerState.COMPILED)
    led.advance("rope", LedgerState.SMOKE_PASSED)
    led.advance("rope", LedgerState.VERIFIED_CORRECT, verify_report={"pass": 5, "fail": 0})
    led.advance("rope", LedgerState.PERF_MEASURED, perf_report={"speedups": {"mx_eager": 1.2, "mx_fast_rope": 0.8}})

    out = render_final_answer(led, op="rope")
    assert "verified correct" in out.lower()
    assert "1.2" in out  # eager speedup
    assert "0.8" in out  # mx_fast speedup honestly reported


def test_refuses_to_claim_correctness_on_verified_incorrect():
    led = KernelLedger()
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    led.advance("rope", LedgerState.GENERATED, kernel_source="k")
    led.advance("rope", LedgerState.COMPILED)
    led.advance("rope", LedgerState.SMOKE_PASSED)
    led.advance("rope", LedgerState.VERIFIED_INCORRECT, verify_report={"pass": 3, "fail": 2})

    out = render_final_answer(led, op="rope")
    assert "verified correct" not in out.lower()
    assert "failed" in out.lower() or "mismatched" in out.lower() or "abandoned" in out.lower()


def test_refuses_to_claim_correctness_on_smoke_passed_only():
    """A smoke-passed kernel without holdout verification is NOT correct."""
    led = KernelLedger()
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    led.advance("rope", LedgerState.GENERATED, kernel_source="k")
    led.advance("rope", LedgerState.COMPILED)
    led.advance("rope", LedgerState.SMOKE_PASSED)

    out = render_final_answer(led, op="rope")
    assert "verified correct" not in out.lower()


def test_full_run_summary_across_ops():
    led = KernelLedger()
    # rope: verified correct
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    led.advance("rope", LedgerState.GENERATED)
    led.advance("rope", LedgerState.COMPILED)
    led.advance("rope", LedgerState.SMOKE_PASSED)
    led.advance("rope", LedgerState.VERIFIED_CORRECT, verify_report={"pass": 5, "fail": 0})
    led.advance("rope", LedgerState.PERF_MEASURED, perf_report={"speedups": {"mx_eager": 1.1}})
    # rmsnorm: abandoned
    led.start("rmsnorm", iteration=1, llm_route="deepseek-v4-flash")
    led.advance("rmsnorm", LedgerState.GENERATED)
    led.advance("rmsnorm", LedgerState.COMPILED)
    led.advance("rmsnorm", LedgerState.SMOKE_PASSED)
    led.advance("rmsnorm", LedgerState.VERIFIED_INCORRECT, verify_report={"pass": 1, "fail": 4})
    led.advance("rmsnorm", LedgerState.ABANDONED)

    summary = render_final_answer(led)
    assert "rope" in summary.lower() and "rmsnorm" in summary.lower()
    assert "verified correct" in summary.lower()
    # honest: rmsnorm did NOT claim correctness
    rmsnorm_line = [line for line in summary.split("\n") if "rmsnorm" in line.lower()][0]
    assert "verified correct" not in rmsnorm_line.lower()


def test_sanitizes_error_string_that_contains_forbidden_phrase():
    """Even if an LLM-echoed error string contains 'verified correct',
    the renderer must not leak it for a non-good state.
    """
    led = KernelLedger()
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    led.advance(
        "rope",
        LedgerState.ABANDONED,
        error="LLM claimed: 'kernel verified correct on first try' but compile threw",
    )
    out = render_final_answer(led, op="rope")
    assert "verified correct" not in out.lower()
    assert "abandoned" in out.lower()


def test_sanitizes_case_and_whitespace_variants():
    """Cover VERIFIED CORRECT, verified_correct, verified  correct, etc."""
    variants = [
        "VERIFIED CORRECT",
        "Verified Correct",
        "verified_correct",
        "verified   correct",
        "verified\tcorrect",
    ]
    for v in variants:
        led = KernelLedger()
        led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
        led.advance("rope", LedgerState.ABANDONED, error=f"LLM said: '{v}'")
        out = render_final_answer(led, op="rope")
        # Case-insensitive, whitespace/underscore-permissive check.
        import re as _re
        assert _re.search(r"verified[\s_]+correct", out, _re.IGNORECASE) is None, (
            f"variant {v!r} leaked into output: {out!r}"
        )


def test_full_run_never_claims_correctness_without_verified_state():
    """Whatever combination of ledger states exists, the renderer must
    never produce the substring 'verified correct' for an op whose
    latest ledger entry is not VERIFIED_CORRECT or PERF_MEASURED.

    This is the regression guard for the entire project.
    """
    for bad_final in (
        LedgerState.GENERATED,
        LedgerState.COMPILED,
        LedgerState.SMOKE_PASSED,
        LedgerState.VERIFIED_INCORRECT,
        LedgerState.ABANDONED,
    ):
        led = KernelLedger()
        led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
        walk = [
            LedgerState.GENERATED,
            LedgerState.COMPILED,
            LedgerState.SMOKE_PASSED,
            LedgerState.VERIFIED_INCORRECT,
            LedgerState.ABANDONED,
        ]
        for s in walk:
            if s == bad_final:
                if s == LedgerState.ABANDONED:
                    led.advance("rope", LedgerState.ABANDONED, error="test")
                else:
                    led.advance("rope", s)
                break
            led.advance("rope", s)
        out = render_final_answer(led, op="rope")
        assert "verified correct" not in out.lower(), (
            f"renderer claimed correctness in state {bad_final.value}: {out!r}"
        )
