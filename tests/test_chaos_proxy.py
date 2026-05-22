"""Tests for chaos middleware: LLM-side 503 injection + kernel_lab corruption hook."""
from __future__ import annotations

import importlib

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient


def test_chaos_503_when_scenario_is_demo_main(monkeypatch, tmp_path):
    """When CHAOS_SCENARIO points at a config that injects llm 503,
    the chaos LLM proxy should return 503 without hitting upstream.
    """
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "chaos.toml").write_text(
        """
[scenario.demo_main]
description = "test"

[[scenario.demo_main.faults]]
target = "llm:deepseek-v4-flash"
mode = "503"
"""
    )
    monkeypatch.setenv("CHAOS_SCENARIO", "demo_main")
    monkeypatch.setenv("CHAOS_CONFIG_PATH", str(cfg / "chaos.toml"))

    import chaos.llm_proxy as mod
    importlib.reload(mod)
    client = TestClient(mod.app)
    r = client.post("/v1/chat/completions", json={"model": "deepseek-v4-flash", "messages": []})
    assert r.status_code == 503
    assert "chaos-503" in r.text


def test_no_chaos_passes_through(monkeypatch, tmp_path):
    """no_chaos scenario should attempt to call upstream (which will fail
    because no upstream is running — but we check the proxy did NOT short-
    circuit with a 503/429 chaos response.
    """
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "chaos.toml").write_text("[scenario.no_chaos]\ndescription = 'test'\n")
    monkeypatch.setenv("CHAOS_SCENARIO", "no_chaos")
    monkeypatch.setenv("CHAOS_CONFIG_PATH", str(cfg / "chaos.toml"))
    monkeypatch.setenv("CHAOS_LLM_UPSTREAM", "http://127.0.0.1:1")  # nothing listening

    import chaos.llm_proxy as mod
    importlib.reload(mod)
    client = TestClient(mod.app, raise_server_exceptions=False)
    r = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    # Either the proxy returned an upstream connection error (5xx that is
    # NOT 503 with our 'chaos-503' marker) OR a TestClient default error.
    # The important assertion: we did not short-circuit with 503 chaos.
    assert not (r.status_code == 503 and "chaos-503" in r.text)


def test_kernel_lab_corruption_changes_output(monkeypatch):
    """The CHAOS_KERNEL_LAB_MODE=silently_wrong_output hook should
    perturb run_kernel's outputs above the holdout tolerance threshold.
    """
    monkeypatch.setenv("CHAOS_KERNEL_LAB_MODE", "silently_wrong_output")
    monkeypatch.setenv("CHAOS_KERNEL_LAB_CURRENT_OP", "rope")
    from kernel_lab.run_tool import _maybe_corrupt

    arr = mx.array([1.0, 2.0, 3.0, 4.0])
    out = _maybe_corrupt([arr], "rope")
    diff = (mx.array(out[0]) - arr).abs().sum().item()
    assert diff > 0.01  # well above the 1e-4 holdout tolerance
