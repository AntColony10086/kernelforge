"""local_gateway: a small FastAPI proxy that mimics TrueFoundry AI Gateway
routing semantics for the case where TrueFoundry SaaS access is not
available before D5. Honestly labeled - every response carries a
`x-local-gateway: yes` header so the demo never claims this IS TrueFoundry.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

app = FastAPI(
    title="local_gateway",
    description="TrueFoundry AI Gateway fallback for KernelForge demo only.",
)


_DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
_FALLBACK_STATUS = {408, 429, 500, 502, 503, 504}


def _select_route(default_route: str, x_tfy_metadata: str | None) -> tuple[str, str]:
    """Return (route_name, deepseek_model_id). Mimics TF routing_config."""
    if x_tfy_metadata and "escalate=pro" in x_tfy_metadata:
        return "deepseek-v4-pro", "deepseek-v4-pro"
    return default_route, default_route


@app.post("/v1/chat/completions")
async def chat_completions(req: Request, x_tfy_metadata: str | None = Header(default=None)) -> JSONResponse:
    body: dict[str, Any] = await req.json()
    requested_model = body.get("model", "deepseek-v4-flash")
    route_name, model = _select_route(requested_model, x_tfy_metadata)

    body["model"] = model
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        first = await client.post(
            f"{_DEEPSEEK_BASE}/v1/chat/completions",
            headers=headers,
            json=body,
        )

    fallback_applied = False
    if first.status_code in _FALLBACK_STATUS and route_name != "deepseek-v4-pro":
        body["model"] = "deepseek-v4-pro"
        async with httpx.AsyncClient(timeout=60) as client:
            second = await client.post(
                f"{_DEEPSEEK_BASE}/v1/chat/completions",
                headers=headers,
                json=body,
            )
        if second.status_code == 200:
            first = second
            fallback_applied = True
            route_name = "deepseek-v4-pro"

    resp_headers = {
        "x-local-gateway": "yes",
        "x-tfy-routing": (
            f"from={requested_model} to={route_name} "
            f"reason={'quality-escalation' if fallback_applied else 'configured'}"
        ),
    }
    if x_tfy_metadata:
        resp_headers["x-tfy-metadata-echo"] = x_tfy_metadata

    try:
        content = first.json()
    except Exception:  # noqa: BLE001
        content = {"error": "upstream returned non-JSON", "status_code": first.status_code, "text": first.text[:500]}

    return JSONResponse(content=content, status_code=first.status_code, headers=resp_headers)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "gateway": "local_gateway"}
