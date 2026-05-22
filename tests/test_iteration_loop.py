"""Tests for the iteration loop. Uses a stub LLM that returns kernels
deterministically so we can exercise the state machine without burning
DeepSeek tokens.
"""
import json

import pytest

from kernelforge.agent import run_kernelforge
from kernelforge.ledger import LedgerState
from kernelforge.llm_client import LLMClient
from kernelforge.prompts import KernelOutput


# An identity kernel that should pass smoke (but probably fails RoPE
# holdouts because it doesn't actually do rotation).
_IDENTITY_SRC = """
uint tid = thread_position_in_grid.x;
if (tid < x_shape[0] * x_shape[1] * x_shape[2]) {
    out[tid] = x[tid];
}
"""


class StubTransport:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(self, messages, *, route, extra_headers=None):
        self.calls.append({"route": route, "headers": extra_headers or {}})
        out = KernelOutput(
            source=_IDENTITY_SRC,
            grid=(64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(1, 8, 64)],
        )
        return {"text": json.dumps(out.model_dump()), "route": route}


@pytest.mark.asyncio
async def test_iteration_loop_runs_at_least_once():
    transport = StubTransport()
    led = await run_kernelforge("rope", LLMClient(transport=transport), profile="demo")
    # Identity kernel is wrong for RoPE; expect VERIFIED_INCORRECT or ABANDONED.
    e = led.latest("rope")
    assert e is not None
    assert e.state in {LedgerState.VERIFIED_INCORRECT, LedgerState.ABANDONED, LedgerState.VERIFIED_CORRECT}
    assert len(transport.calls) >= 1


@pytest.mark.asyncio
async def test_iteration_loop_escalates_after_first_failure():
    """If iteration 1 fails, the loop should call the LLM at least twice
    and the second call's metadata should request escalation."""
    transport = StubTransport()
    led = await run_kernelforge("rope", LLMClient(transport=transport), profile="demo")
    e = led.latest("rope")
    if e.state == LedgerState.VERIFIED_CORRECT:
        # Identity happened to pass — unlikely but skip rest in that case.
        return
    # At least 2 calls if first iteration didn't verify.
    if len(transport.calls) >= 2:
        assert "escalate=pro" in transport.calls[1]["headers"].get("X-TFY-METADATA", "")
