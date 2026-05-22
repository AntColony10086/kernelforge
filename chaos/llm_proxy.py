"""FastAPI reverse proxy in front of local_gateway / TrueFoundry, with
deterministic fault injection per chaos.toml. Used only in demo and tests.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="chaos_llm_proxy")

_UPSTREAM = os.environ.get("CHAOS_LLM_UPSTREAM", "http://127.0.0.1:8765")
_CONFIG_PATH = os.environ.get("CHAOS_CONFIG_PATH", "configs/chaos.toml")


def _faults() -> list[dict]:
    scenario = os.environ.get("CHAOS_SCENARIO", "no_chaos")
    cfg_path = Path(_CONFIG_PATH)
    if not cfg_path.exists():
        return []
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("scenario", {}).get(scenario, {}).get("faults", [])


@app.post("/v1/chat/completions")
async def proxy(req: Request) -> Response:
    for fault in _faults():
        if not fault.get("target", "").startswith("llm:"):
            continue
        mode = fault.get("mode", "none")
        if mode == "503":
            return JSONResponse(status_code=503, content={"error": "chaos-503", "fault": fault["target"]})
        if mode == "429":
            return JSONResponse(status_code=429, content={"error": "chaos-429", "fault": fault["target"]})

    body = await req.body()
    async with httpx.AsyncClient(timeout=120) as client:
        upstream = await client.post(
            f"{_UPSTREAM}/v1/chat/completions",
            content=body,
            headers={k: v for k, v in req.headers.items() if k.lower() not in {"host", "content-length"}},
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items() if k.lower() not in {"content-length", "transfer-encoding"}},
    )
