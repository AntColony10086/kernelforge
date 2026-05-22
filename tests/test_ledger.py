"""Ledger contract: monotonic transitions, single source of truth for the final answer."""
from __future__ import annotations

import pytest

from kernelforge.ledger import KernelLedger, LedgerState


def test_initial_state_is_attempted():
    led = KernelLedger()
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    e = led.latest("rope")
    assert e is not None
    assert e.state == LedgerState.ATTEMPTED
    assert e.iteration == 1


def test_advance_must_be_monotonic():
    led = KernelLedger()
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    led.advance("rope", LedgerState.GENERATED, kernel_source="...")
    led.advance("rope", LedgerState.COMPILED)
    led.advance("rope", LedgerState.SMOKE_PASSED)
    led.advance("rope", LedgerState.VERIFIED_CORRECT, verify_report={"pass": 5, "fail": 0})
    with pytest.raises(ValueError):
        led.advance("rope", LedgerState.GENERATED)  # backward transition


def test_verified_incorrect_starts_new_iteration():
    led = KernelLedger()
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    led.advance("rope", LedgerState.GENERATED, kernel_source="bad")
    led.advance("rope", LedgerState.COMPILED)
    led.advance("rope", LedgerState.SMOKE_PASSED)
    led.advance("rope", LedgerState.VERIFIED_INCORRECT, verify_report={"pass": 3, "fail": 2})

    led.start("rope", iteration=2, llm_route="deepseek-v4-pro")
    e = led.latest("rope")
    assert e.iteration == 2
    assert e.state == LedgerState.ATTEMPTED


def test_latest_per_op():
    led = KernelLedger()
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    led.start("rmsnorm", iteration=1, llm_route="deepseek-v4-flash")
    led.advance("rope", LedgerState.GENERATED)
    assert led.latest("rope").state == LedgerState.GENERATED
    assert led.latest("rmsnorm").state == LedgerState.ATTEMPTED


def test_serialize_to_jsonl(tmp_path):
    led = KernelLedger()
    led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
    led.advance("rope", LedgerState.GENERATED, kernel_source="x")
    path = tmp_path / "ledger.jsonl"
    led.dump(path)
    text = path.read_text()
    assert "attempted" in text and "generated" in text
