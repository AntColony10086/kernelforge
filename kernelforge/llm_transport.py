"""HTTP transport wiring for LLMClient. Picks TrueFoundry gateway if
TFY_GATEWAY_BASE_URL is set, else falls back to local_gateway at
127.0.0.1:8765. Either way, the contract is OpenAI-compatible
/v1/chat/completions.
"""
from __future__ import annotations

import os

import httpx


class HttpTransport:
    def __init__(self) -> None:
        self.base_url = os.environ.get("TFY_GATEWAY_BASE_URL", "http://127.0.0.1:8765")
        self.api_key = os.environ.get("TFY_GATEWAY_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")

    async def complete(self, messages: list[dict], *, route: str, extra_headers: dict | None = None) -> dict:
        body = {
            "model": route,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self.base_url}/v1/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        payload = r.json()

        text = payload["choices"][0]["message"]["content"]
        gateway_route = r.headers.get("x-tfy-routing", route)
        return {"text": text, "route": gateway_route}
