"""Tests for llm_client. Uses a fake transport so we don't burn DeepSeek credits."""
from __future__ import annotations

import json

import pytest

from kernelforge.llm_client import LLMClient, LLMRouteChoice
from kernelforge.prompts import KernelOutput


class FakeTransport:
    def __init__(self, payloads: list[dict | Exception]):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    async def complete(self, messages, *, route: str, extra_headers: dict | None = None):
        self.calls.append({"messages": messages, "route": route, "headers": extra_headers or {}})
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


@pytest.mark.asyncio
async def test_uses_flash_on_happy_path():
    sample = KernelOutput(
        source="// trivial",
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1,)],
    ).model_dump()
    fake = FakeTransport([{"text": json.dumps(sample), "route": "deepseek-v4-flash"}])
    client = LLMClient(transport=fake, default_route="deepseek-v4-flash")
    out = await client.generate_kernel(
        op="rope", reference_signature="def rope(x): ...", previous_diff=None, escalate=False
    )
    assert isinstance(out, KernelOutput)
    assert fake.calls[0]["route"] == "deepseek-v4-flash"
    assert "escalate=flash" in fake.calls[0]["headers"]["X-TFY-METADATA"]


@pytest.mark.asyncio
async def test_escalates_to_pro_via_metadata_header():
    sample = KernelOutput(
        source="// v2",
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1,)],
    ).model_dump()
    fake = FakeTransport([{"text": json.dumps(sample), "route": "deepseek-v4-pro"}])
    client = LLMClient(transport=fake, default_route="deepseek-v4-flash")
    out = await client.generate_kernel(
        op="rope", reference_signature="def rope(x): ...", previous_diff={"failing_case": "x"}, escalate=True
    )
    assert isinstance(out, KernelOutput)
    assert "escalate=pro" in fake.calls[0]["headers"]["X-TFY-METADATA"]


@pytest.mark.asyncio
async def test_parse_failure_raises_specific_error():
    fake = FakeTransport([{"text": "not json", "route": "deepseek-v4-flash"}])
    client = LLMClient(transport=fake, default_route="deepseek-v4-flash")
    with pytest.raises(LLMRouteChoice.ParseError):
        await client.generate_kernel(op="rope", reference_signature="x", previous_diff=None, escalate=False)


@pytest.mark.asyncio
async def test_validation_failure_raises_parse_error():
    fake = FakeTransport([{"text": "{}", "route": "deepseek-v4-flash"}])  # missing required fields
    client = LLMClient(transport=fake, default_route="deepseek-v4-flash")
    with pytest.raises(LLMRouteChoice.ParseError):
        await client.generate_kernel(op="rope", reference_signature="x", previous_diff=None, escalate=False)
