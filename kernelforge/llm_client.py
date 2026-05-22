"""LLM client that talks to TrueFoundry AI Gateway (or local_gateway fallback).

Owns: cost-aware escalation via X-TFY-METADATA header; strict pydantic schema parsing.
Does NOT own: provider failover logic - that lives in the gateway's routing_config.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from kernelforge.prompts import KernelOutput, generate_prompt


class _TransportProtocol(Protocol):
    async def complete(
        self,
        messages: list[dict],
        *,
        route: str,
        extra_headers: dict | None = None,
    ) -> dict: ...


@dataclass
class LLMRouteChoice:
    route: str

    class ParseError(ValueError):
        pass


class LLMClient:
    def __init__(self, transport: _TransportProtocol, *, default_route: str = "deepseek-v4-flash") -> None:
        self._transport = transport
        self._default_route = default_route

    async def generate_kernel(
        self,
        *,
        op: str,
        reference_signature: str,
        previous_diff: dict | None,
        escalate: bool,
    ) -> KernelOutput:
        messages = generate_prompt(op=op, reference_signature=reference_signature, previous_diff=previous_diff)
        route = "deepseek-v4-pro" if escalate else self._default_route
        run_id = os.environ.get("KERNELFORGE_RUN_ID", "")
        # Emit metadata as a semicolon-separated key=value string. local_gateway
        # checks for the literal substring "escalate=pro" so this form works.
        meta_value = f"escalate={'pro' if escalate else 'flash'};op={op};run_id={run_id}"
        headers = {"X-TFY-METADATA": meta_value}

        response = await self._transport.complete(messages, route=route, extra_headers=headers)
        text: str = response["text"]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMRouteChoice.ParseError(f"LLM did not return JSON: {exc}") from exc
        try:
            return KernelOutput.model_validate(parsed)
        except ValidationError as exc:
            raise LLMRouteChoice.ParseError(f"LLM JSON does not match KernelOutput: {exc}") from exc
