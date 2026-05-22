import json

import pytest

from baselines.naive import naive_run
from kernelforge.llm_client import LLMClient
from kernelforge.prompts import KernelOutput


class StubTransport:
    def __init__(self, kernel: KernelOutput):
        self.kernel = kernel

    async def complete(self, messages, *, route, extra_headers=None):
        return {"text": json.dumps(self.kernel.model_dump()), "route": route}


@pytest.mark.asyncio
async def test_naive_claims_correct_when_kernel_compiles():
    sample = KernelOutput(
        source="uint tid = thread_position_in_grid.x;\nif (tid < x_shape[0] * x_shape[1] * x_shape[2]) { out[tid] = x[tid]; }",
        grid=(64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, 8, 64)],
    )
    llm = LLMClient(transport=StubTransport(sample))
    res = await naive_run("rope", llm)
    assert res.claimed_correct is True  # naive trusts the compile
    assert res.claimed_speedup == 1.4
