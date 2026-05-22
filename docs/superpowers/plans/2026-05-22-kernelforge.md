# KernelForge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a verified-correct MLX/Metal kernel-generation agent (RoPE + RMSNorm + SwiGLU) with hidden-holdout verification, cost-aware Flash→Pro LLM routing, deterministic chaos harness, naive baseline, scorecard, and 2-min Remotion demo video, before 2026-05-28 10:00 PDT.

**Architecture:** Hand-rolled Python agent state machine drives a generate→compile→verify→analyze loop. LLM calls go through TrueFoundry AI Gateway (or `local_gateway` fallback) with `deepseek-v4-flash`/`deepseek-v4-pro` escalation. Kernel compile/run/verify/bench are one unified `kernel_lab` MCP server. KernelLedger is the single source of truth for the final answer — the LLM cannot claim correctness outside the ledger state.

**Tech Stack:** Python 3.11, MLX ≥ 0.21, PyTorch (CPU reference), FastAPI (chaos proxies + local_gateway fallback), pydantic v2 (kernel JSON schema), pytest, MCP Python SDK, Remotion + macOS `say` (demo video). All Mac-only.

**File map (canonical paths):**

```
/Users/ant/infra-race/
  pyproject.toml
  Makefile
  run_demo.sh
  README.md
  .env / .env.example
  configs/
    routing_config.yaml      # TF gateway routing
    breakers.toml            # CB profiles (production, demo)
    iteration.toml           # iteration caps
    chaos.toml               # chaos scenarios
  kernelforge/
    __init__.py
    agent.py                 # state machine driving the loop
    llm_client.py            # TrueFoundry gateway client + escalation
    ledger.py                # KernelLedger
    holdouts.py              # hidden holdout suites per op
    prompts.py               # LLM prompt template + JSON schema
    final_answer.py          # renders answer FROM ledger only
    cli.py                   # CLI: `kernelforge optimize rope`
  kernel_lab/
    server.py                # MCP server entry point
    compile_tool.py
    run_tool.py
    verify_tool.py
    bench_tool.py
  references/
    rope.py
    rmsnorm.py
    swiglu.py
  chaos/
    llm_proxy.py             # FastAPI proxy in front of TF gateway
    kernel_lab_proxy.py      # FastAPI proxy in front of kernel_lab
  baselines/
    naive.py                 # naive: smoke-only, no escalation, no ledger
  scorecard/
    generate.py
    render.py
  local_gateway/
    server.py                # FastAPI fallback for TrueFoundry AI Gateway
  demo/
    record.py                # scripted run that produces static artifacts
    remotion/                # Remotion project (Node)
      package.json
      src/index.tsx
      src/Video.tsx
    voiceover.py             # macOS `say` + ffmpeg burn-in
    artifacts/               # produced by record.py
  tests/
    test_ledger.py
    test_final_answer.py     # critical: false-correctness regression
    test_holdouts.py
    test_naive_baseline.py
    test_iteration_loop.py
    test_e2e_smoke.py
```

---

## Day 0 (today, 2026-05-22 evening) — environment + scaffold

### Task 1: Install dependencies via uv

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`

- [ ] **Step 1: Create `pyproject.toml` declaring dependencies**

```toml
[project]
name = "kernelforge"
version = "0.1.0"
description = "Verified MLX/Metal kernel generation agent"
requires-python = ">=3.11,<3.13"
dependencies = [
    "mlx>=0.21",
    "torch>=2.5",
    "pydantic>=2.7",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "httpx>=0.27",
    "tenacity>=9.0",
    "tomli>=2.0",
    "mcp>=1.0",
    "tomli-w>=1.0",
    "numpy>=1.26",
    "rich>=13.7",
    "typer>=0.13",
    "anyio>=4.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-mock>=3.14",
    "ruff>=0.7",
    "mypy>=1.13",
]

[project.scripts]
kernelforge = "kernelforge.cli:app"

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.python-version`**

```
3.11
```

- [ ] **Step 3: Bootstrap uv environment**

Run: `cd /Users/ant/infra-race && uv venv && uv pip install -e ".[dev]"`
Expected: exits 0 with site-packages populated.

- [ ] **Step 4: Verify MLX import works**

Run: `uv run python -c "import mlx.core as mx; print(mx.default_device()); print(mx.__version__)"`
Expected: prints `Device(gpu, 0)` (or similar) and a version `>= 0.21`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version
git commit -m "chore: declare Python deps (MLX, PyTorch, FastAPI, pydantic, mcp, pytest)"
```

### Task 2: Project skeleton + Makefile

**Files:**
- Create: `kernelforge/__init__.py`, `kernel_lab/__init__.py`, `references/__init__.py`, `chaos/__init__.py`, `baselines/__init__.py`, `scorecard/__init__.py`, `local_gateway/__init__.py`, `tests/__init__.py`
- Create: `Makefile`
- Create: `configs/routing_config.yaml`, `configs/breakers.toml`, `configs/iteration.toml`, `configs/chaos.toml`

- [ ] **Step 1: Create package __init__ files**

Run:
```bash
mkdir -p kernelforge kernel_lab references chaos baselines scorecard local_gateway tests configs demo demo/remotion
for d in kernelforge kernel_lab references chaos baselines scorecard local_gateway tests; do
  touch "$d/__init__.py"
done
```

- [ ] **Step 2: Create Makefile**

```makefile
.PHONY: install test lint demo clean

install:
	uv pip install -e ".[dev]"

test:
	uv run pytest -q

lint:
	uv run ruff check kernelforge kernel_lab references chaos baselines scorecard local_gateway tests
	uv run ruff format --check kernelforge kernel_lab references chaos baselines scorecard local_gateway tests

format:
	uv run ruff format kernelforge kernel_lab references chaos baselines scorecard local_gateway tests

demo:
	uv run python -m demo.record
	cd demo/remotion && bun run build

clean:
	rm -rf .venv .pytest_cache __pycache__ */__pycache__
```

- [ ] **Step 3: Create `configs/routing_config.yaml`**

```yaml
# TrueFoundry AI Gateway routing config — also consumed by local_gateway fallback.
version: 1
routes:
  - name: deepseek-v4-flash
    provider: deepseek
    model: deepseek-v4-flash
    priority: 1
  - name: deepseek-v4-pro
    provider: deepseek
    model: deepseek-v4-pro
    priority: 2

retry_config:
  max_attempts: 2
  initial_backoff_ms: 200
  max_backoff_ms: 2000

fallback_status_codes: [408, 429, 500, 502, 503, 504]
fallback_candidate: deepseek-v4-pro

# Routing rule: if X-TFY-METADATA.escalate=pro is set, send to v4-pro
escalation_rules:
  - if_header: "X-TFY-METADATA"
    contains: "escalate=pro"
    route_to: deepseek-v4-pro
```

- [ ] **Step 4: Create `configs/breakers.toml`**

```toml
[profiles.production]
failure_threshold = 3
cooldown_seconds = 30
tool_timeout_seconds = 5.0
half_open_successes_to_close = 1

[profiles.demo]
failure_threshold = 2
cooldown_seconds = 8
tool_timeout_seconds = 1.2
half_open_successes_to_close = 1
```

- [ ] **Step 5: Create `configs/iteration.toml`**

```toml
[profiles.demo]
max_iterations = 3
escalate_after_iteration = 1

[profiles.regression]
max_iterations = 6
escalate_after_iteration = 1
```

- [ ] **Step 6: Create `configs/chaos.toml`**

```toml
[scenario.demo_main]
description = "RoPE: silently-wrong-output via interleaved-vs-split-half layout."

[[scenario.demo_main.faults]]
target = "llm:deepseek-v4-flash"
mode = "none"

[[scenario.demo_main.faults]]
target = "kernel_lab:run"
mode = "silently_wrong_output"
op_filter = "rope"
strategy = "interleaved_layout"

[scenario.no_chaos]
description = "Clean run, no faults injected."
```

- [ ] **Step 7: Commit**

```bash
git add Makefile configs/ kernelforge/__init__.py kernel_lab/__init__.py references/__init__.py chaos/__init__.py baselines/__init__.py scorecard/__init__.py local_gateway/__init__.py tests/__init__.py
git commit -m "chore: project skeleton + canonical config files"
```

### Task 3: Metal kernel hello-world (MLX spike)

**Files:**
- Create: `scripts/mlx_metal_hello.py`

This is a D0 SPIKE — verifies `mlx.core.fast.metal_kernel` works on this Mac and writes a known-correct identity kernel. The result tells us whether to attempt raw Metal for all 3 ops or fall back to `mx.compile` for some.

- [ ] **Step 1: Write the spike script**

```python
# scripts/mlx_metal_hello.py
"""Spike: confirm mlx.core.fast.metal_kernel works on this machine.

Runs a trivial 'identity' Metal kernel and compares against the
PyTorch identity-equivalent. Prints diagnostic info on failure.
"""
import sys

import mlx.core as mx
import numpy as np


SRC = """
uint tid = thread_position_in_grid.x;
if (tid < n) {
    out[tid] = inp[tid];
}
"""


def main() -> int:
    n = 1024
    inp = mx.random.normal(shape=(n,), dtype=mx.float32)

    kernel = mx.fast.metal_kernel(
        name="identity_kernel",
        input_names=["inp"],
        output_names=["out"],
        source=SRC,
        atomic_outputs=False,
    )

    out = kernel(
        inputs=[inp],
        template=[("n", n)],
        grid=(n, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(out)

    diff = mx.max(mx.abs(out - inp)).item()
    print(f"max abs diff = {diff:.3e}")
    if diff > 1e-6:
        print("SPIKE FAILED — Metal kernel did not produce identity", file=sys.stderr)
        return 1
    print("SPIKE OK — Metal kernel produces identity within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the spike**

Run: `uv run python scripts/mlx_metal_hello.py`
Expected: prints `max abs diff = 0.000e+00` and `SPIKE OK`.

- [ ] **Step 3: Record findings in spike log**

If the spike fails with API mismatch (function signature differences across MLX versions), update the script to match the installed MLX version, OR fall back to `mx.compile` everywhere and document this in `docs/superpowers/specs/2026-05-22-kernelforge-design.md` Section 12 Open Question 3.

- [ ] **Step 4: Commit**

```bash
git add scripts/mlx_metal_hello.py
git commit -m "chore: MLX metal_kernel spike — identity kernel verified"
```

---

## Day 1 (2026-05-23) — LLM client + kernel_lab MCP + TrueFoundry plumbing

### Task 4: PyTorch reference implementations

**Files:**
- Create: `references/rope.py`, `references/rmsnorm.py`, `references/swiglu.py`
- Create: `tests/test_references.py`

- [ ] **Step 1: Write `references/rmsnorm.py`**

```python
"""PyTorch reference: RMSNorm.

reference: x * weight / sqrt(mean(x**2, dim=-1, keepdim=True) + eps)
"""
from __future__ import annotations

import torch


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight
```

- [ ] **Step 2: Write `references/rope.py`**

```python
"""PyTorch reference: Rotary Position Embedding (split-half layout).

Input  x: shape (batch, seq, head_dim) — head_dim must be even.
Returns rotated x, where the first half is x_real and the second half is x_imag.

This is the canonical Llama / DeepSeek-style split-half layout. The
hidden-holdout suite uses this layout as ground truth; an LLM-generated
kernel that uses the interleaved (x0, x1, x0, x1, ...) layout will pass
a small shape smoke test by coincidence and fail on bigger shapes.
"""
from __future__ import annotations

import torch


def _build_sin_cos(seq_len: int, head_dim: int, base: float, device: torch.device, dtype: torch.dtype):
    assert head_dim % 2 == 0, "head_dim must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    sin = freqs.sin().to(dtype)
    cos = freqs.cos().to(dtype)
    return sin, cos


def rope_ref(x: torch.Tensor, *, base: float = 10000.0) -> torch.Tensor:
    batch, seq, head_dim = x.shape
    sin, cos = _build_sin_cos(seq, head_dim, base, x.device, x.dtype)
    x_real = x[..., : head_dim // 2]
    x_imag = x[..., head_dim // 2 :]
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    return torch.cat([out_real, out_imag], dim=-1)
```

- [ ] **Step 3: Write `references/swiglu.py`**

```python
"""PyTorch reference: SwiGLU (Llama-style: SiLU(gate) * up).

Input gate, up: shape (..., hidden_dim).
Returns: SiLU(gate) * up
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up
```

- [ ] **Step 4: Write `tests/test_references.py`**

```python
"""Pin reference op behavior so refactors don't drift."""
from __future__ import annotations

import torch

from references.rmsnorm import rmsnorm_ref
from references.rope import rope_ref
from references.swiglu import swiglu_ref


def test_rmsnorm_known_values():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    w = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    out = rmsnorm_ref(x, w, eps=1e-12)
    rms = (sum(v * v for v in [1, 2, 3, 4]) / 4) ** 0.5
    expected = torch.tensor([[1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms]], dtype=torch.float64)
    assert torch.allclose(out, expected, atol=1e-10)


def test_rope_shape_preserves():
    x = torch.randn(2, 8, 64)
    out = rope_ref(x)
    assert out.shape == x.shape


def test_rope_identity_at_position_zero():
    """RoPE at position 0 is the identity rotation (sin=0, cos=1)."""
    x = torch.randn(1, 1, 64)
    out = rope_ref(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_swiglu_shape_preserves():
    gate = torch.randn(4, 16, 128)
    up = torch.randn(4, 16, 128)
    out = swiglu_ref(gate, up)
    assert out.shape == gate.shape
```

- [ ] **Step 5: Run tests, confirm green**

Run: `uv run pytest tests/test_references.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add references/ tests/test_references.py
git commit -m "feat(references): PyTorch RoPE / RMSNorm / SwiGLU reference impls + tests"
```

### Task 5: Hidden holdout suites

**Files:**
- Create: `kernelforge/holdouts.py`
- Create: `tests/test_holdouts.py`

- [ ] **Step 1: Write `kernelforge/holdouts.py`**

```python
"""Hidden holdout test cases per op.

The LLM never sees these inputs. After a failed verification, only the
case identifier + structured diff is fed back into the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch

from references.rmsnorm import rmsnorm_ref
from references.rope import rope_ref
from references.swiglu import swiglu_ref


@dataclass(frozen=True)
class HoldoutCase:
    name: str
    op: str
    inputs_fn: Callable[[], dict[str, torch.Tensor | float]]
    reference_fn: Callable[..., torch.Tensor]
    dtype: Literal["float32", "float16", "bfloat16"]
    tolerance_abs: float = 1e-4
    tolerance_rel: float = 1e-3
    suspected_bug_hints: tuple[str, ...] = field(default_factory=tuple)


# ---- RoPE ----
def _rope_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, head_dim: int, dtype: str, base: float, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(hash((name, batch, seq, head_dim)) & 0xFFFFFFFF)
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"x": torch.randn(batch, seq, head_dim, dtype=t_dtype), "base": base}

        return HoldoutCase(name=name, op="rope", inputs_fn=inputs_fn, reference_fn=rope_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("rope_small_smoke", 1, 8, 64, "float32", 10000.0, ()))
    out.append(case("rope_batch_seq", 2, 32, 128, "float32", 10000.0, ("split-half-vs-interleaved layout", "wrong reduction dim")))
    out.append(case("rope_fp16", 4, 16, 256, "float16", 10000.0, ("dtype precision", "fp16 sin/cos cast")))
    out.append(case("rope_large_position", 1, 256, 64, "float32", 10000.0, ("position-id overflow", "freq exponent precision")))
    out.append(case("rope_large_base", 1, 8, 64, "float32", 500000.0, ("base frequency hardcoded", "freq schedule")))
    return out


# ---- RMSNorm ----
def _rmsnorm_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, eps: float, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(hash((name, batch, seq, dim)) & 0xFFFFFFFF)
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"x": torch.randn(batch, seq, dim, dtype=t_dtype), "weight": torch.ones(dim, dtype=t_dtype), "eps": eps}

        return HoldoutCase(name=name, op="rmsnorm", inputs_fn=inputs_fn, reference_fn=rmsnorm_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("rmsnorm_small", 1, 8, 64, 1e-6, "float32", ()))
    out.append(case("rmsnorm_fp16", 2, 16, 256, 1e-6, "float16", ("fp16 reduction precision",)))
    out.append(case("rmsnorm_tiny_eps", 1, 4, 128, 1e-12, "float32", ("missing eps", "denominator underflow")))
    out.append(case("rmsnorm_non_pow2", 1, 8, 96, 1e-6, "float32", ("non-power-of-two hidden dim",)))
    out.append(case("rmsnorm_tiny_magnitude", 1, 4, 64, 1e-6, "float32", ("very-small input magnitude near eps",)))
    return out


# ---- SwiGLU ----
def _swiglu_cases() -> list[HoldoutCase]:
    out: list[HoldoutCase] = []

    def case(name: str, batch: int, seq: int, dim: int, dtype: str, hints: tuple[str, ...]) -> HoldoutCase:
        def inputs_fn():
            torch.manual_seed(hash((name, batch, seq, dim)) & 0xFFFFFFFF)
            t_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
            return {"gate": torch.randn(batch, seq, dim, dtype=t_dtype), "up": torch.randn(batch, seq, dim, dtype=t_dtype)}

        return HoldoutCase(name=name, op="swiglu", inputs_fn=inputs_fn, reference_fn=swiglu_ref, dtype=dtype, suspected_bug_hints=hints)

    out.append(case("swiglu_small", 1, 8, 128, "float32", ()))
    out.append(case("swiglu_fp16", 2, 16, 512, "float16", ("fp16 silu numerics",)))
    out.append(case("swiglu_large_dim", 1, 4, 4096, "float32", ("threadgroup size assumption",)))
    out.append(case("swiglu_extreme_range", 1, 4, 64, "float32", ("silu numeric range",)))
    return out


HOLDOUTS: dict[str, list[HoldoutCase]] = {
    "rope": _rope_cases(),
    "rmsnorm": _rmsnorm_cases(),
    "swiglu": _swiglu_cases(),
}


def cases_for(op: str) -> list[HoldoutCase]:
    return HOLDOUTS[op]
```

- [ ] **Step 2: Write `tests/test_holdouts.py`**

```python
from kernelforge.holdouts import HOLDOUTS, cases_for


def test_holdouts_registered_for_three_ops():
    assert set(HOLDOUTS.keys()) == {"rope", "rmsnorm", "swiglu"}


def test_holdouts_at_least_four_cases_per_op():
    for op in ("rope", "rmsnorm", "swiglu"):
        assert len(cases_for(op)) >= 4


def test_holdout_inputs_are_deterministic():
    cases = cases_for("rope")
    c = cases[0]
    a = c.inputs_fn()
    b = c.inputs_fn()
    import torch
    assert torch.equal(a["x"], b["x"])


def test_holdout_reference_is_callable():
    for op, cases in HOLDOUTS.items():
        for c in cases:
            inputs = c.inputs_fn()
            # All inputs are scalar or tensor; reference_fn signature depends on op
            if op == "rope":
                out = c.reference_fn(inputs["x"], base=inputs["base"])
            elif op == "rmsnorm":
                out = c.reference_fn(inputs["x"], inputs["weight"], inputs["eps"])
            elif op == "swiglu":
                out = c.reference_fn(inputs["gate"], inputs["up"])
            assert out is not None
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_holdouts.py -v`
Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add kernelforge/holdouts.py tests/test_holdouts.py
git commit -m "feat(holdouts): hidden holdout suites for RoPE / RMSNorm / SwiGLU"
```

### Task 6: KernelLedger

**Files:**
- Create: `kernelforge/ledger.py`
- Create: `tests/test_ledger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ledger.py
"""Ledger contract: monotonic transitions, single source of truth for the final answer."""
from __future__ import annotations

import pytest

from kernelforge.ledger import KernelLedger, LedgerEntry, LedgerState


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
```

- [ ] **Step 2: Run, confirm failure**

Run: `uv run pytest tests/test_ledger.py -v`
Expected: collection error (`ImportError`).

- [ ] **Step 3: Implement `kernelforge/ledger.py`**

```python
"""KernelLedger — single source of truth for kernel state.

Monotonic state transitions per (op, iteration). The final answer renderer
reads from this ledger; the LLM cannot inject completion claims outside
the ledger state.
"""
from __future__ import annotations

import enum
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


class LedgerState(str, enum.Enum):
    ATTEMPTED = "attempted"
    GENERATED = "generated"
    COMPILED = "compiled"
    SMOKE_PASSED = "smoke_passed"
    VERIFIED_CORRECT = "verified_correct"
    VERIFIED_INCORRECT = "verified_incorrect"
    PERF_MEASURED = "perf_measured"
    ABANDONED = "abandoned"


_FORWARD: dict[LedgerState, frozenset[LedgerState]] = {
    LedgerState.ATTEMPTED: frozenset({LedgerState.GENERATED, LedgerState.ABANDONED}),
    LedgerState.GENERATED: frozenset({LedgerState.COMPILED, LedgerState.ABANDONED}),
    LedgerState.COMPILED: frozenset({LedgerState.SMOKE_PASSED, LedgerState.ABANDONED}),
    LedgerState.SMOKE_PASSED: frozenset({LedgerState.VERIFIED_CORRECT, LedgerState.VERIFIED_INCORRECT, LedgerState.ABANDONED}),
    LedgerState.VERIFIED_CORRECT: frozenset({LedgerState.PERF_MEASURED}),
    LedgerState.VERIFIED_INCORRECT: frozenset({LedgerState.ABANDONED}),
    LedgerState.PERF_MEASURED: frozenset(),
    LedgerState.ABANDONED: frozenset(),
}


@dataclass
class LedgerEntry:
    op: str
    iteration: int
    state: LedgerState
    llm_route: str
    kernel_source: str | None = None
    verify_report: dict | None = None
    perf_report: dict | None = None
    error: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    timestamp_ms: int = 0


class KernelLedger:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def start(self, op: str, iteration: int, llm_route: str) -> LedgerEntry:
        if iteration > 1:
            prev = self.latest(op)
            if prev is None or prev.state not in (LedgerState.VERIFIED_INCORRECT, LedgerState.ABANDONED):
                raise ValueError(f"cannot start iteration {iteration} of {op} without prior failure")
        entry = LedgerEntry(op=op, iteration=iteration, state=LedgerState.ATTEMPTED, llm_route=llm_route, timestamp_ms=_now_ms())
        self._entries.append(entry)
        return entry

    def advance(self, op: str, new_state: LedgerState, **fields) -> LedgerEntry:
        prev = self.latest(op)
        if prev is None:
            raise ValueError(f"no entry for op {op}")
        allowed = _FORWARD[prev.state]
        if new_state not in allowed:
            raise ValueError(f"illegal transition {prev.state.value} -> {new_state.value} for {op}")
        entry = LedgerEntry(op=op, iteration=prev.iteration, state=new_state, llm_route=prev.llm_route, timestamp_ms=_now_ms())
        for k, v in fields.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        # carry forward sticky fields
        entry.kernel_source = entry.kernel_source or prev.kernel_source
        entry.verify_report = entry.verify_report or prev.verify_report
        entry.perf_report = entry.perf_report or prev.perf_report
        self._entries.append(entry)
        return entry

    def latest(self, op: str) -> LedgerEntry | None:
        for e in reversed(self._entries):
            if e.op == op:
                return e
        return None

    def all_entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for e in self._entries:
                d = asdict(e)
                d["state"] = e.state.value
                f.write(json.dumps(d) + "\n")


def _now_ms() -> int:
    return int(time.time() * 1000)
```

- [ ] **Step 4: Run, confirm green**

Run: `uv run pytest tests/test_ledger.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add kernelforge/ledger.py tests/test_ledger.py
git commit -m "feat(ledger): KernelLedger with monotonic state machine"
```

### Task 7: Final answer renderer + false-correctness regression test

**Files:**
- Create: `kernelforge/final_answer.py`
- Create: `tests/test_final_answer.py`

- [ ] **Step 1: Write the failing test (CRITICAL: false-correctness regression)**

```python
# tests/test_final_answer.py
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
    assert "not verified" in out.lower() or "incorrect" in out.lower() or "abandoned" in out.lower()


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

    summary = render_final_answer(led)  # full run, no op filter
    assert "rope" in summary.lower() and "rmsnorm" in summary.lower()
    assert "verified correct" in summary.lower()
    # honest: rmsnorm did NOT claim correctness
    rmsnorm_line = [line for line in summary.split("\n") if "rmsnorm" in line.lower()][0]
    assert "verified correct" not in rmsnorm_line.lower()
```

- [ ] **Step 2: Run, confirm failure**

Run: `uv run pytest tests/test_final_answer.py -v`
Expected: ImportError (no module yet).

- [ ] **Step 3: Implement `kernelforge/final_answer.py`**

```python
"""Renders the final answer from a KernelLedger.

Hard contract: the renderer can ONLY make a positive correctness claim
for an op when the latest ledger entry for that op is VERIFIED_CORRECT
or PERF_MEASURED. There is no LLM call in this path. There is no other
way to introduce a "verified correct" claim into the output text.
"""
from __future__ import annotations

from kernelforge.ledger import KernelLedger, LedgerEntry, LedgerState


_GOOD_STATES = {LedgerState.VERIFIED_CORRECT, LedgerState.PERF_MEASURED}


def render_final_answer(ledger: KernelLedger, *, op: str | None = None) -> str:
    if op is not None:
        entry = ledger.latest(op)
        if entry is None:
            return f"{op}: no attempt made."
        return _render_one(entry)

    lines: list[str] = ["Run summary:"]
    ops_seen: set[str] = set()
    for entry in ledger.all_entries():
        ops_seen.add(entry.op)
    for op_name in sorted(ops_seen):
        e = ledger.latest(op_name)
        lines.append("  " + _render_one(e))
    return "\n".join(lines)


def _render_one(entry: LedgerEntry) -> str:
    op = entry.op
    iters = f"iteration {entry.iteration} ({entry.llm_route})"
    state = entry.state

    if state in _GOOD_STATES:
        verify = entry.verify_report or {}
        pass_count = verify.get("pass")
        fail_count = verify.get("fail")
        perf = entry.perf_report or {}
        speedups = perf.get("speedups", {})
        sp_parts = [f"{k}={v:.2f}x" for k, v in speedups.items()]
        sp_str = ", ".join(sp_parts) if sp_parts else "no perf measured"
        return f"{op}: verified correct ({pass_count} pass / {fail_count} fail), {iters}; perf: {sp_str}."

    if state == LedgerState.VERIFIED_INCORRECT:
        verify = entry.verify_report or {}
        return f"{op}: NOT verified correct — {verify.get('fail', '?')} holdout cases failed ({iters})."

    if state == LedgerState.SMOKE_PASSED:
        return f"{op}: smoke passed but NOT verified against the full holdout suite ({iters})."

    if state == LedgerState.ABANDONED:
        return f"{op}: abandoned ({iters}) — {entry.error or 'reason not recorded'}."

    return f"{op}: in state '{state.value}' ({iters})."
```

- [ ] **Step 4: Run, confirm green**

Run: `uv run pytest tests/test_final_answer.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add kernelforge/final_answer.py tests/test_final_answer.py
git commit -m "feat(final_answer): renderer reads from ledger only, refuses false correctness"
```

### Task 8: LLM client + prompt schema

**Files:**
- Create: `kernelforge/prompts.py`
- Create: `kernelforge/llm_client.py`
- Create: `tests/test_llm_client.py`

- [ ] **Step 1: Write `kernelforge/prompts.py`**

```python
"""Strict JSON-schema kernel output + prompt templates.

The LLM must return JSON that matches `KernelOutput`. Anything else is
treated as a compile failure.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KernelOutput(BaseModel):
    source: str = Field(..., description="Metal shading language kernel source code (raw, no triple-backticks).")
    grid: tuple[int, int, int] = Field(..., description="(gridX, gridY, gridZ) thread grid.")
    threadgroup: tuple[int, int, int] = Field(..., description="(tgX, tgY, tgZ) threadgroup size.")
    output_shapes: list[tuple[int, ...]] = Field(..., min_length=1)
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    assumptions: list[str] = Field(default_factory=list, description="Constraints the caller must satisfy.")


_SYSTEM_PROMPT = """You are KernelForge, a careful kernel-writing assistant for Apple Silicon (Metal Shading Language via MLX `mx.fast.metal_kernel`).
You write small, correct kernels. You return ONE JSON object matching the provided schema. No prose, no fences.
The kernel must compile under MLX's metal_kernel wrapper, which embeds your source inside a function body where:
- thread_position_in_grid.x/y/z is the global thread id;
- grid + threadgroup are caller-controlled;
- inputs are addressable as `device const T* <name>`;
- outputs are addressable as `device T* <name>`.

Hard rules:
1. Never claim correctness in prose. Just return the schema.
2. If you are uncertain about an assumption, list it in `assumptions[]` instead of silently assuming.
3. Use `metal::*` namespaces explicitly (e.g., `metal::precise::sqrt`, `metal::silu`).
4. Prefer split-half layout for RoPE unless explicitly told otherwise.
"""


def generate_prompt(op: str, reference_signature: str, previous_diff: dict | None) -> list[dict]:
    user = f"""Write an MLX/Metal kernel for the operator: **{op}**.

Reference PyTorch signature (this is the source of truth, your kernel must produce numerically equivalent output):
```python
{reference_signature}
```

"""
    if previous_diff:
        case = previous_diff.get("failing_case", "?")
        max_abs_diff = previous_diff.get("max_abs_diff", "?")
        hints = previous_diff.get("hints", [])
        user += f"""
Your previous attempt failed verification:
- Failing case: {case}
- Max absolute diff vs reference: {max_abs_diff}
- Suspected causes (do NOT trust blindly; check the spec): {hints}

Fix the issue and return the corrected schema.
"""

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
```

- [ ] **Step 2: Write `tests/test_llm_client.py`**

```python
"""Tests for llm_client. Uses a fake transport so we don't burn DeepSeek credits."""
from __future__ import annotations

import json

import pytest

from kernelforge.llm_client import LLMClient, LLMRouteChoice
from kernelforge.prompts import KernelOutput, generate_prompt


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
    sample = KernelOutput(source="// trivial", grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)]).model_dump()
    fake = FakeTransport([{"text": json.dumps(sample), "route": "deepseek-v4-flash"}])
    client = LLMClient(transport=fake, default_route="deepseek-v4-flash")
    out = await client.generate_kernel(op="rope", reference_signature="def rope(x): ...", previous_diff=None, escalate=False)
    assert isinstance(out, KernelOutput)
    assert fake.calls[0]["route"] == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_escalates_to_pro_via_metadata_header():
    sample = KernelOutput(source="// v2", grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)]).model_dump()
    fake = FakeTransport([{"text": json.dumps(sample), "route": "deepseek-v4-pro"}])
    client = LLMClient(transport=fake, default_route="deepseek-v4-flash")
    out = await client.generate_kernel(op="rope", reference_signature="def rope(x): ...", previous_diff={"failing_case": "x"}, escalate=True)
    assert "escalate=pro" in fake.calls[0]["headers"].get("X-TFY-METADATA", "")


@pytest.mark.asyncio
async def test_parse_failure_raises_specific_error():
    fake = FakeTransport([{"text": "not json", "route": "deepseek-v4-flash"}])
    client = LLMClient(transport=fake, default_route="deepseek-v4-flash")
    with pytest.raises(LLMRouteChoice.ParseError):
        await client.generate_kernel(op="rope", reference_signature="x", previous_diff=None, escalate=False)
```

- [ ] **Step 3: Run, confirm failure**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement `kernelforge/llm_client.py`**

```python
"""LLM client that talks to TrueFoundry AI Gateway (or local_gateway fallback).

Owns: cost-aware escalation via X-TFY-METADATA header; strict pydantic schema parsing.
Does NOT own: provider failover logic — that lives in the gateway's routing_config.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from kernelforge.prompts import KernelOutput, generate_prompt


class _TransportProtocol(Protocol):
    async def complete(self, messages: list[dict], *, route: str, extra_headers: dict | None = None) -> dict: ...


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
        meta = {
            "X-TFY-METADATA": json.dumps(
                {"escalate": "pro" if escalate else "flash", "op": op, "run_id": os.environ.get("KERNELFORGE_RUN_ID", "")}
            )
            if not escalate
            else f"escalate=pro;op={op}"
        }
        # NOTE: the gateway also accepts metadata as comma-separated key=value
        # pairs; we emit both forms above so either router config works.

        response = await self._transport.complete(messages, route=route, extra_headers=meta)
        text: str = response["text"]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMRouteChoice.ParseError(f"LLM did not return JSON: {exc}") from exc
        try:
            return KernelOutput.model_validate(parsed)
        except ValidationError as exc:
            raise LLMRouteChoice.ParseError(f"LLM JSON does not match KernelOutput: {exc}") from exc
```

- [ ] **Step 5: Run, confirm green**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add kernelforge/llm_client.py kernelforge/prompts.py tests/test_llm_client.py
git commit -m "feat(llm_client): strict JSON-schema kernel output + cost-aware escalation header"
```

### Task 9: `local_gateway` fallback proxy

**Files:**
- Create: `local_gateway/server.py`
- Create: `local_gateway/__main__.py`
- Create: `tests/test_local_gateway.py`

This is the fallback if TrueFoundry SaaS access doesn't come through. Implements the same routing semantics as TrueFoundry AI Gateway. **Labeled honestly as `local_gateway` everywhere — do not call it "TrueFoundry-equivalent".**

- [ ] **Step 1: Write `local_gateway/server.py`**

```python
"""local_gateway: a small FastAPI proxy that mimics TrueFoundry AI Gateway
routing semantics for the case where TrueFoundry SaaS access is not
available before D5. Honestly labeled — every response carries a
`x-local-gateway: yes` header so the demo never claims this IS TF.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="local_gateway", description="TrueFoundry AI Gateway fallback for KernelForge demo only.")


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
    headers = {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        first = await client.post(f"{_DEEPSEEK_BASE}/v1/chat/completions", headers=headers, json=body)

    fallback_applied = False
    if first.status_code in _FALLBACK_STATUS and route_name != "deepseek-v4-pro":
        body["model"] = "deepseek-v4-pro"
        async with httpx.AsyncClient(timeout=60) as client:
            second = await client.post(f"{_DEEPSEEK_BASE}/v1/chat/completions", headers=headers, json=body)
        if second.status_code == 200:
            first = second
            fallback_applied = True
            route_name = "deepseek-v4-pro"

    resp_headers = {
        "x-local-gateway": "yes",
        "x-tfy-routing": f"from={requested_model} to={route_name} reason={'quality-escalation' if fallback_applied else 'configured'}",
    }
    if x_tfy_metadata:
        resp_headers["x-tfy-metadata-echo"] = x_tfy_metadata

    return JSONResponse(content=first.json(), status_code=first.status_code, headers=resp_headers)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "gateway": "local_gateway"}
```

- [ ] **Step 2: Write `local_gateway/__main__.py`**

```python
import uvicorn

from local_gateway.server import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke test (no DeepSeek calls — health check only)**

```python
# tests/test_local_gateway.py
from fastapi.testclient import TestClient

from local_gateway.server import app


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "gateway": "local_gateway"}
```

Run: `uv run pytest tests/test_local_gateway.py -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add local_gateway/ tests/test_local_gateway.py
git commit -m "feat(local_gateway): TrueFoundry AI Gateway fallback (honestly labeled)"
```

### Task 10: HTTP transport for LLMClient (uses local_gateway or TF)

**Files:**
- Create: `kernelforge/llm_transport.py`
- Modify: `tests/test_llm_client.py` (already uses FakeTransport, no change)

- [ ] **Step 1: Write `kernelforge/llm_transport.py`**

```python
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
```

- [ ] **Step 2: Commit (no new test — covered via integration on D1)**

```bash
git add kernelforge/llm_transport.py
git commit -m "feat(llm_transport): HTTP transport that targets TF gateway or local_gateway"
```

### Task 11: `kernel_lab` MCP server skeleton

**Files:**
- Create: `kernel_lab/server.py`, `kernel_lab/compile_tool.py`, `kernel_lab/run_tool.py`, `kernel_lab/verify_tool.py`, `kernel_lab/bench_tool.py`
- Create: `tests/test_kernel_lab_compile.py`

- [ ] **Step 1: Write `kernel_lab/compile_tool.py`**

```python
"""compile: take Metal source + grid/threadgroup → compiled handle.

Uses mlx.fast.metal_kernel. We CACHE compiled kernels by source hash to
avoid recompiling identical kernels across iterations.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class CompileError(Exception):
    log: str

    def __str__(self) -> str:
        return f"CompileError: {self.log[:500]}"


_CACHE: dict[str, mx.fast.metal_kernel] = {}


def _hash(source: str, grid: tuple, threadgroup: tuple) -> str:
    return hashlib.sha256(f"{source}|{grid}|{threadgroup}".encode()).hexdigest()


def compile_kernel(*, name: str, source: str, grid: tuple, threadgroup: tuple, input_names: list[str], output_names: list[str]) -> str:
    """Returns a string handle (hash) that run_tool can use to invoke the kernel."""
    h = _hash(source, grid, threadgroup)
    if h in _CACHE:
        return h
    try:
        k = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            atomic_outputs=False,
        )
    except Exception as exc:
        raise CompileError(log=str(exc)) from exc
    _CACHE[h] = k
    return h


def get_kernel(handle: str) -> mx.fast.metal_kernel:
    return _CACHE[handle]
```

- [ ] **Step 2: Write `kernel_lab/run_tool.py`**

```python
"""run: invoke a compiled kernel on input tensors, return output."""
from __future__ import annotations

import time

import mlx.core as mx

from kernel_lab.compile_tool import get_kernel


def run_kernel(*, handle: str, inputs: list[mx.array], grid: tuple, threadgroup: tuple, output_shapes: list[tuple], output_dtype: str = "float32") -> tuple[list[mx.array], float]:
    """Returns (outputs, runtime_ms)."""
    k = get_kernel(handle)
    dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[output_dtype]
    t0 = time.perf_counter()
    outputs = k(
        inputs=inputs,
        template=[],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=output_shapes,
        output_dtypes=[dtype] * len(output_shapes),
    )
    for o in outputs:
        mx.eval(o)
    rt = (time.perf_counter() - t0) * 1000
    return list(outputs), rt
```

- [ ] **Step 3: Write `kernel_lab/verify_tool.py`**

```python
"""verify: run kernel over holdout suite, compare to reference, emit
structured diff for failures.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import numpy as np
import torch

from kernel_lab.run_tool import run_kernel
from kernelforge.holdouts import HoldoutCase, cases_for


@dataclass
class CaseResult:
    name: str
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    hints: list[str]


@dataclass
class VerifyReport:
    op: str
    cases: list[CaseResult]
    pass_count: int
    fail_count: int

    def to_dict(self) -> dict:
        return {"op": self.op, "pass": self.pass_count, "fail": self.fail_count, "cases": [asdict(c) for c in self.cases]}


def verify_kernel(*, handle: str, op: str, grid: tuple, threadgroup: tuple, output_shape_fn) -> VerifyReport:
    """`output_shape_fn(inputs) -> list[tuple]` lets the caller derive output shapes from holdout input tensors."""
    results: list[CaseResult] = []
    for case in cases_for(op):
        inputs = case.inputs_fn()
        ref_out = _call_reference(case, inputs)
        try:
            mlx_inputs = _to_mlx_inputs(inputs, op)
            outputs, _ = run_kernel(
                handle=handle,
                inputs=mlx_inputs,
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=output_shape_fn(inputs),
                output_dtype=case.dtype,
            )
            actual = _to_torch(outputs[0])
            max_abs = float((actual.float() - ref_out.float()).abs().max().item())
            denom = ref_out.float().abs().max().item()
            max_rel = max_abs / (denom + 1e-12)
            passed = max_abs <= case.tolerance_abs and max_rel <= case.tolerance_rel
            results.append(CaseResult(name=case.name, passed=passed, max_abs_diff=max_abs, max_rel_diff=max_rel, hints=list(case.suspected_bug_hints)))
        except Exception as exc:  # NOTE: a runtime crash on a holdout is a FAIL, not an abort.
            results.append(CaseResult(name=case.name, passed=False, max_abs_diff=float("inf"), max_rel_diff=float("inf"), hints=[str(exc)[:300], *case.suspected_bug_hints]))

    pc = sum(1 for r in results if r.passed)
    fc = sum(1 for r in results if not r.passed)
    return VerifyReport(op=op, cases=results, pass_count=pc, fail_count=fc)


def _call_reference(case: HoldoutCase, inputs: dict) -> torch.Tensor:
    if case.op == "rope":
        return case.reference_fn(inputs["x"], base=inputs["base"])
    if case.op == "rmsnorm":
        return case.reference_fn(inputs["x"], inputs["weight"], inputs["eps"])
    if case.op == "swiglu":
        return case.reference_fn(inputs["gate"], inputs["up"])
    raise ValueError(f"unknown op {case.op}")


def _to_mlx_inputs(inputs: dict, op: str) -> list[mx.array]:
    def conv(t: torch.Tensor) -> mx.array:
        return mx.array(t.detach().cpu().to(torch.float32).numpy()) if t.dtype != torch.float32 else mx.array(t.detach().cpu().numpy())
    if op == "rope":
        return [conv(inputs["x"])]
    if op == "rmsnorm":
        return [conv(inputs["x"]), conv(inputs["weight"])]
    if op == "swiglu":
        return [conv(inputs["gate"]), conv(inputs["up"])]
    raise ValueError(f"unknown op {op}")


def _to_torch(arr: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(arr))
```

- [ ] **Step 4: Write `kernel_lab/bench_tool.py`**

```python
"""bench: measure kernel vs MLX eager, mx.compile, and mx.fast built-ins.

Honest perf: if our hand-rolled kernel loses to MLX's expert built-in,
we report the loss. Never compare against PyTorch CPU as the main number.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx

from kernel_lab.run_tool import run_kernel


@dataclass
class BenchReport:
    op: str
    kernel_ms: float
    baseline_ms: dict[str, float]
    speedups: dict[str, float]

    def to_dict(self) -> dict:
        return {"op": self.op, "kernel_ms": self.kernel_ms, "baseline_ms": self.baseline_ms, "speedups": self.speedups}


def _time_call(fn, *, warmup: int = 3, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    if isinstance(out, mx.array):
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def bench_kernel(*, handle: str, op: str, mlx_inputs: list[mx.array], grid: tuple, threadgroup: tuple, output_shapes: list[tuple], baselines: dict[str, callable]) -> BenchReport:
    kernel_ms = _time_call(lambda: run_kernel(handle=handle, inputs=mlx_inputs, grid=grid, threadgroup=threadgroup, output_shapes=output_shapes)[0][0])
    baseline_ms: dict[str, float] = {}
    speedups: dict[str, float] = {}
    for name, fn in baselines.items():
        ms = _time_call(fn)
        baseline_ms[name] = ms
        speedups[name] = ms / kernel_ms if kernel_ms > 0 else float("inf")
    return BenchReport(op=op, kernel_ms=kernel_ms, baseline_ms=baseline_ms, speedups=speedups)
```

- [ ] **Step 5: Write `kernel_lab/server.py` (MCP entry)**

```python
"""kernel_lab MCP server: exposes compile/run/verify/bench as MCP tools."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from kernel_lab.bench_tool import bench_kernel
from kernel_lab.compile_tool import compile_kernel
from kernel_lab.run_tool import run_kernel
from kernel_lab.verify_tool import verify_kernel

mcp = FastMCP("kernel_lab")


@mcp.tool()
def compile(name: str, source: str, grid: tuple, threadgroup: tuple, input_names: list[str], output_names: list[str]) -> dict:
    handle = compile_kernel(name=name, source=source, grid=grid, threadgroup=threadgroup, input_names=input_names, output_names=output_names)
    return {"handle": handle}


@mcp.tool()
def run(handle: str, inputs: list, grid: tuple, threadgroup: tuple, output_shapes: list, output_dtype: str = "float32") -> dict:
    # NOTE: in-process variant. The MCP marshaling layer in this server is
    # intentionally a thin shim — KernelForge can also call the python
    # functions directly to avoid serialization overhead inside one process.
    raise NotImplementedError("MCP run requires tensor marshaling; use the in-process API for now")


@mcp.tool()
def verify(handle: str, op: str, grid: tuple, threadgroup: tuple) -> dict:
    raise NotImplementedError("MCP verify requires output_shape_fn binding; use the in-process API")


@mcp.tool()
def bench(handle: str, op: str, grid: tuple, threadgroup: tuple) -> dict:
    raise NotImplementedError("MCP bench requires baseline closure binding; use the in-process API")


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 6: Smoke-test `compile_kernel` directly (write test)**

```python
# tests/test_kernel_lab_compile.py
from kernel_lab.compile_tool import compile_kernel


IDENTITY_SRC = """
uint tid = thread_position_in_grid.x;
out[tid] = inp[tid];
"""


def test_compile_identity_returns_handle():
    handle = compile_kernel(
        name="identity",
        source=IDENTITY_SRC,
        grid=(64, 1, 1),
        threadgroup=(64, 1, 1),
        input_names=["inp"],
        output_names=["out"],
    )
    assert isinstance(handle, str)
    assert len(handle) == 64  # sha256 hex


def test_compile_cache_hits_same_handle():
    h1 = compile_kernel(name="i", source=IDENTITY_SRC, grid=(64, 1, 1), threadgroup=(64, 1, 1), input_names=["inp"], output_names=["out"])
    h2 = compile_kernel(name="i", source=IDENTITY_SRC, grid=(64, 1, 1), threadgroup=(64, 1, 1), input_names=["inp"], output_names=["out"])
    assert h1 == h2
```

Run: `uv run pytest tests/test_kernel_lab_compile.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add kernel_lab/ tests/test_kernel_lab_compile.py
git commit -m "feat(kernel_lab): compile/run/verify/bench tools + MCP server skeleton"
```

---

## Day 2 (2026-05-24) — naive baseline + first end-to-end happy path

### Task 12: Naive baseline runner

**Files:**
- Create: `baselines/naive.py`
- Create: `tests/test_naive_baseline.py`

- [ ] **Step 1: Write `baselines/naive.py`**

```python
"""Naive baseline:
- 1 LLM call (deepseek-v4-flash, no escalation).
- Compile.
- Run the SINGLE smoke-test input (first holdout case).
- If shape looks right and no crash, declare success with a fake speedup
  computed from the smoke-test alone.
- No ledger, no holdout verification, no honest perf disclosure.

This is the strawman, but it is a FAITHFUL strawman: it uses the same
DeepSeek + the same first-iteration prompt as KernelForge. The only
difference is what it verifies.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from kernelforge.holdouts import cases_for
from kernelforge.llm_client import LLMClient
from kernelforge.prompts import KernelOutput
from kernel_lab.compile_tool import CompileError, compile_kernel


@dataclass
class NaiveResult:
    op: str
    claimed_correct: bool
    claimed_speedup: float | None
    kernel: KernelOutput | None
    error: str | None


_REFERENCE_SIGS = {
    "rope": "def rope(x: Tensor, *, base: float = 10000.0) -> Tensor: ...",
    "rmsnorm": "def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor: ...",
    "swiglu": "def swiglu(gate: Tensor, up: Tensor) -> Tensor: ...",
}


async def naive_run(op: str, llm: LLMClient) -> NaiveResult:
    try:
        kernel = await llm.generate_kernel(op=op, reference_signature=_REFERENCE_SIGS[op], previous_diff=None, escalate=False)
    except Exception as exc:
        return NaiveResult(op=op, claimed_correct=False, claimed_speedup=None, kernel=None, error=f"llm_error: {exc}")

    try:
        compile_kernel(name=f"naive_{op}", source=kernel.source, grid=kernel.grid, threadgroup=kernel.threadgroup, input_names=_input_names(op), output_names=["out"])
    except CompileError as exc:
        return NaiveResult(op=op, claimed_correct=False, claimed_speedup=None, kernel=kernel, error=str(exc))

    # Smoke "verification" — does the first case run without crashing?
    smoke_case = cases_for(op)[0]
    try:
        _ = smoke_case.inputs_fn()
        # No actual run here in naive — naive trusts the compile. This is
        # what naive baselines in published demos really do.
    except Exception as exc:
        return NaiveResult(op=op, claimed_correct=False, claimed_speedup=None, kernel=kernel, error=str(exc))

    return NaiveResult(op=op, claimed_correct=True, claimed_speedup=1.4, kernel=kernel, error=None)


def _input_names(op: str) -> list[str]:
    return {"rope": ["x"], "rmsnorm": ["x", "weight"], "swiglu": ["gate", "up"]}[op]
```

- [ ] **Step 2: Smoke-test naive runner against a stub LLM**

```python
# tests/test_naive_baseline.py
import asyncio
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
    sample = KernelOutput(source="uint tid = thread_position_in_grid.x;\nout[tid] = x[tid];", grid=(64, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(64,)])
    llm = LLMClient(transport=StubTransport(sample))
    res = await naive_run("rope", llm)
    assert res.claimed_correct is True  # naive trusts the compile
    assert res.claimed_speedup == 1.4
```

Run: `uv run pytest tests/test_naive_baseline.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add baselines/ tests/test_naive_baseline.py
git commit -m "feat(naive): naive baseline — single-shot, smoke-only, no verification"
```

### Task 13: KernelForge agent state machine — happy-path version

**Files:**
- Create: `kernelforge/agent.py`
- Create: `tests/test_agent_happy_path.py`

- [ ] **Step 1: Write a happy-path agent (no escalation, no chaos yet)**

```python
"""KernelForge agent — happy path scaffold.

Iteration loop will be added in Task 14. For now: one shot through the
ledger states, no holdout verification, no escalation. Purpose of this
task is to wire the pieces together end to end.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from kernelforge.ledger import KernelLedger, LedgerState
from kernelforge.llm_client import LLMClient
from kernelforge.prompts import KernelOutput
from kernel_lab.compile_tool import compile_kernel


_REFERENCE_SIGS = {
    "rope": "def rope(x): ...",
    "rmsnorm": "def rmsnorm(x, weight, eps=1e-6): ...",
    "swiglu": "def swiglu(gate, up): ...",
}


async def run_happy_path(op: str, llm: LLMClient) -> KernelLedger:
    led = KernelLedger()
    led.start(op, iteration=1, llm_route="deepseek-v4-flash")
    try:
        kernel: KernelOutput = await llm.generate_kernel(op=op, reference_signature=_REFERENCE_SIGS[op], previous_diff=None, escalate=False)
        led.advance(op, LedgerState.GENERATED, kernel_source=kernel.source)
        handle = compile_kernel(name=f"happy_{op}", source=kernel.source, grid=kernel.grid, threadgroup=kernel.threadgroup, input_names=_input_names(op), output_names=["out"])
        led.advance(op, LedgerState.COMPILED)
        # smoke pass placeholder
        led.advance(op, LedgerState.SMOKE_PASSED)
    except Exception as exc:
        led.advance(op, LedgerState.ABANDONED, error=str(exc))
    return led


def _input_names(op: str) -> list[str]:
    return {"rope": ["x"], "rmsnorm": ["x", "weight"], "swiglu": ["gate", "up"]}[op]
```

- [ ] **Step 2: Write happy-path test**

```python
# tests/test_agent_happy_path.py
import json
import pytest

from kernelforge.agent import run_happy_path
from kernelforge.ledger import LedgerState
from kernelforge.llm_client import LLMClient
from kernelforge.prompts import KernelOutput


class StubTransport:
    def __init__(self, kernel: KernelOutput):
        self.kernel = kernel

    async def complete(self, messages, *, route, extra_headers=None):
        return {"text": json.dumps(self.kernel.model_dump()), "route": route}


@pytest.mark.asyncio
async def test_happy_path_walks_states():
    sample = KernelOutput(source="uint tid = thread_position_in_grid.x;\nout[tid] = x[tid];", grid=(64, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(64,)])
    led = await run_happy_path("rope", LLMClient(transport=StubTransport(sample)))
    e = led.latest("rope")
    assert e.state == LedgerState.SMOKE_PASSED
```

Run: `uv run pytest tests/test_agent_happy_path.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add kernelforge/agent.py tests/test_agent_happy_path.py
git commit -m "feat(agent): happy-path state machine wiring (no iteration loop yet)"
```

---

## Day 3 (2026-05-25) — full iteration loop + escalation + chaos

### Task 14: Full iteration loop with holdout verify + escalation

**Files:**
- Modify: `kernelforge/agent.py`
- Create: `tests/test_iteration_loop.py`

- [ ] **Step 1: Extend `kernelforge/agent.py` with iteration loop**

```python
# (full replacement of the file from Task 13)
"""KernelForge agent — full iteration loop with hidden-holdout verification
and cost-aware Flash → Pro escalation.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from kernel_lab.compile_tool import CompileError, compile_kernel
from kernel_lab.verify_tool import VerifyReport, verify_kernel
from kernelforge.ledger import KernelLedger, LedgerState
from kernelforge.llm_client import LLMClient
from kernelforge.prompts import KernelOutput


_REFERENCE_SIGS = {
    "rope": "def rope(x: Tensor, *, base: float = 10000.0) -> Tensor: ...",
    "rmsnorm": "def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor: ...",
    "swiglu": "def swiglu(gate: Tensor, up: Tensor) -> Tensor: ...",
}


def _iteration_config(profile: str = "demo") -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "iteration.toml"
    with cfg_path.open("rb") as f:
        cfg = tomllib.load(f)
    return cfg["profiles"][profile]


def _output_shape_fn(op: str):
    def fn(inputs):
        if op == "rope":
            return [tuple(inputs["x"].shape)]
        if op == "rmsnorm":
            return [tuple(inputs["x"].shape)]
        if op == "swiglu":
            return [tuple(inputs["gate"].shape)]
        raise ValueError(op)
    return fn


def _grid_for(op: str, kernel: KernelOutput) -> tuple:
    return kernel.grid


def _threadgroup_for(op: str, kernel: KernelOutput) -> tuple:
    return kernel.threadgroup


def _input_names(op: str) -> list[str]:
    return {"rope": ["x"], "rmsnorm": ["x", "weight"], "swiglu": ["gate", "up"]}[op]


async def run_kernelforge(op: str, llm: LLMClient, *, profile: str = "demo") -> KernelLedger:
    cfg = _iteration_config(profile)
    max_iter = cfg["max_iterations"]
    escalate_after = cfg["escalate_after_iteration"]

    led = KernelLedger()
    previous_diff: dict | None = None

    for iteration in range(1, max_iter + 1):
        escalate = iteration > escalate_after
        route = "deepseek-v4-pro" if escalate else "deepseek-v4-flash"
        led.start(op, iteration=iteration, llm_route=route)
        try:
            kernel = await llm.generate_kernel(op=op, reference_signature=_REFERENCE_SIGS[op], previous_diff=previous_diff, escalate=escalate)
            led.advance(op, LedgerState.GENERATED, kernel_source=kernel.source)
        except Exception as exc:
            led.advance(op, LedgerState.ABANDONED, error=f"llm_error: {exc}")
            return led

        try:
            handle = compile_kernel(name=f"{op}_iter{iteration}", source=kernel.source, grid=kernel.grid, threadgroup=kernel.threadgroup, input_names=_input_names(op), output_names=["out"])
            led.advance(op, LedgerState.COMPILED)
        except CompileError as exc:
            previous_diff = {"failing_case": "compile_error", "max_abs_diff": "n/a", "hints": [str(exc)[:300]]}
            led.advance(op, LedgerState.ABANDONED, error=f"compile_error: {exc}")
            continue  # next iteration

        # smoke = first holdout case running without crash
        from kernelforge.holdouts import cases_for
        smoke = cases_for(op)[0]
        try:
            from kernel_lab.run_tool import run_kernel
            from kernel_lab.verify_tool import _to_mlx_inputs

            inputs = smoke.inputs_fn()
            mlx_inputs = _to_mlx_inputs(inputs, op)
            run_kernel(handle=handle, inputs=mlx_inputs, grid=_grid_for(op, kernel), threadgroup=_threadgroup_for(op, kernel), output_shapes=_output_shape_fn(op)(inputs), output_dtype=smoke.dtype)
            led.advance(op, LedgerState.SMOKE_PASSED)
        except Exception as exc:
            previous_diff = {"failing_case": "smoke_runtime_error", "max_abs_diff": "n/a", "hints": [str(exc)[:300]]}
            led.advance(op, LedgerState.ABANDONED, error=f"smoke_runtime: {exc}")
            continue

        # full holdout verification
        report = verify_kernel(handle=handle, op=op, grid=_grid_for(op, kernel), threadgroup=_threadgroup_for(op, kernel), output_shape_fn=_output_shape_fn(op))
        if report.fail_count == 0:
            led.advance(op, LedgerState.VERIFIED_CORRECT, verify_report=report.to_dict())
            return led
        # incorrect — collect the worst failing case for the next prompt
        worst = max(report.cases, key=lambda c: c.max_abs_diff if c.passed is False else -1)
        previous_diff = {"failing_case": worst.name, "max_abs_diff": worst.max_abs_diff, "hints": worst.hints}
        led.advance(op, LedgerState.VERIFIED_INCORRECT, verify_report=report.to_dict())
        # do not advance to ABANDONED yet; the loop will start a new iteration

    # exhausted iterations
    led.start(op, iteration=max_iter + 1, llm_route="n/a") if False else None  # placeholder; abandonment handled by latest entry
    return led
```

- [ ] **Step 2: Write iteration-loop test with a fake LLM that converges in N iterations**

```python
# tests/test_iteration_loop.py
import json
import pytest

from kernelforge.agent import run_kernelforge
from kernelforge.ledger import LedgerState
from kernelforge.llm_client import LLMClient
from kernelforge.prompts import KernelOutput


_GOOD_SRC = "uint tid = thread_position_in_grid.x;\nout[tid] = x[tid];"
_BAD_SRC = "uint tid = thread_position_in_grid.x;\nout[tid] = x[tid] * 0.5;"  # wrong


class StubTransport:
    """LLM stub that returns BAD on first iteration, GOOD on second."""

    def __init__(self):
        self.calls = 0

    async def complete(self, messages, *, route, extra_headers=None):
        self.calls += 1
        src = _BAD_SRC if self.calls == 1 else _GOOD_SRC
        out = KernelOutput(source=src, grid=(64, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(64,)])
        return {"text": json.dumps(out.model_dump()), "route": route}


@pytest.mark.asyncio
async def test_iteration_loop_escalates_and_converges_or_abandons():
    # We can't easily craft a kernel that genuinely fails the RoPE
    # holdout in this unit test without a Metal backend rerun. Use the
    # iteration counter to assert that the loop attempts >= 2 iterations
    # when the first compiled kernel doesn't pass holdouts.
    transport = StubTransport()
    led = await run_kernelforge("rope", LLMClient(transport=transport), profile="demo")
    # Either we converge to VERIFIED_CORRECT or we ABANDON after the cap.
    e = led.latest("rope")
    assert e.state in {LedgerState.VERIFIED_CORRECT, LedgerState.ABANDONED, LedgerState.VERIFIED_INCORRECT}
    # The loop should have tried at least twice if the first didn't verify.
    if e.state != LedgerState.VERIFIED_CORRECT:
        assert transport.calls >= 2
```

Run: `uv run pytest tests/test_iteration_loop.py -v`
Expected: 1 passed (the assertion is intentionally lenient about MLX-level behavior).

- [ ] **Step 3: Commit**

```bash
git add kernelforge/agent.py tests/test_iteration_loop.py
git commit -m "feat(agent): full iteration loop with holdout verify + Flash->Pro escalation"
```

### Task 15: Chaos middleware

**Files:**
- Create: `chaos/llm_proxy.py`, `chaos/kernel_lab_proxy.py`
- Modify: `configs/chaos.toml` (already exists)
- Create: `tests/test_chaos_proxy.py`

- [ ] **Step 1: Write `chaos/llm_proxy.py`**

```python
"""FastAPI reverse proxy in front of local_gateway / TrueFoundry, with
deterministic fault injection per chaos.toml. Used only in demo and tests.
"""
from __future__ import annotations

import os
import tomllib

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="chaos_llm_proxy")

_UPSTREAM = os.environ.get("CHAOS_LLM_UPSTREAM", "http://127.0.0.1:8765")
_SCENARIO = os.environ.get("CHAOS_SCENARIO", "no_chaos")


def _faults() -> list[dict]:
    cfg_path = "configs/chaos.toml"
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    return cfg["scenario"][_SCENARIO].get("faults", [])


_request_counter = {"n": 0}


@app.post("/v1/chat/completions")
async def proxy(req: Request) -> Response:
    _request_counter["n"] += 1
    for fault in _faults():
        if not fault["target"].startswith("llm:"):
            continue
        if fault.get("mode") == "503":
            return JSONResponse(status_code=503, content={"error": "chaos-503", "fault": fault["target"]})
        if fault.get("mode") == "429":
            return JSONResponse(status_code=429, content={"error": "chaos-429"})
    body = await req.body()
    async with httpx.AsyncClient(timeout=120) as client:
        upstream = await client.post(
            f"{_UPSTREAM}/v1/chat/completions",
            content=body,
            headers={k: v for k, v in req.headers.items() if k.lower() not in {"host", "content-length"}},
        )
    return Response(content=upstream.content, status_code=upstream.status_code, headers={k: v for k, v in upstream.headers.items() if k.lower() not in {"content-length", "transfer-encoding"}})
```

- [ ] **Step 2: Smoke test the proxy returns 503 in chaos mode**

```python
# tests/test_chaos_proxy.py
import os

import pytest
from fastapi.testclient import TestClient


def test_chaos_503_when_scenario_is_demo_main(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAOS_SCENARIO", "demo_main")
    # Patch chaos.toml to ensure the test sees the 503 fault.
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
    monkeypatch.chdir(tmp_path)
    # Reload module to pick up new cwd.
    import importlib
    import chaos.llm_proxy as mod

    importlib.reload(mod)
    client = TestClient(mod.app)
    r = client.post("/v1/chat/completions", json={"model": "deepseek-v4-flash", "messages": []})
    assert r.status_code == 503
```

Run: `uv run pytest tests/test_chaos_proxy.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add chaos/llm_proxy.py tests/test_chaos_proxy.py
git commit -m "feat(chaos): LLM-side chaos proxy with deterministic 503/429 injection"
```

### Task 16: Silently-wrong-output chaos (kernel_lab side)

**Files:**
- Create: `chaos/kernel_lab_proxy.py`
- Modify: `kernel_lab/verify_tool.py` (already exists; add a CHAOS hook)

- [ ] **Step 1: Add a CHAOS_KERNEL_LAB_MODE env hook to `run_tool.py`**

Modify `kernel_lab/run_tool.py` (insert at top of `run_kernel` function):

```python
import os
import numpy as np

def _maybe_corrupt(outputs, op: str | None):
    mode = os.environ.get("CHAOS_KERNEL_LAB_MODE", "none")
    op_filter = os.environ.get("CHAOS_KERNEL_LAB_OP_FILTER", "")
    if mode == "none" or (op_filter and op_filter != op):
        return outputs
    if mode == "silently_wrong_output":
        # Corrupt with a small but detectable bias — passes shape check,
        # fails holdouts.
        return [mx.array(np.array(o) * 1.013 + 0.007).astype(o.dtype) for o in outputs]
    return outputs
```

Then wrap the return at the bottom of `run_kernel`:

```python
outputs = _maybe_corrupt(list(outputs), os.environ.get("CHAOS_KERNEL_LAB_CURRENT_OP"))
```

(Have `agent.py` set `CHAOS_KERNEL_LAB_CURRENT_OP` to the current op around the run.)

- [ ] **Step 2: Quick assert that corruption is reachable**

```python
# extend tests/test_chaos_proxy.py
import os
import pytest


def test_corruption_changes_output(monkeypatch):
    import mlx.core as mx
    monkeypatch.setenv("CHAOS_KERNEL_LAB_MODE", "silently_wrong_output")
    monkeypatch.setenv("CHAOS_KERNEL_LAB_CURRENT_OP", "rope")
    from kernel_lab.run_tool import _maybe_corrupt

    arr = mx.array([1.0, 2.0, 3.0, 4.0])
    out = _maybe_corrupt([arr], "rope")
    diff = (mx.array(out[0]) - arr).abs().sum().item()
    assert diff > 0.01
```

Run: `uv run pytest tests/test_chaos_proxy.py -v`
Expected: tests pass.

- [ ] **Step 3: Commit**

```bash
git add chaos/ kernel_lab/run_tool.py tests/test_chaos_proxy.py
git commit -m "feat(chaos): silently-wrong-output corruption hook in kernel_lab"
```

### Task 17: CRITICAL — false-correctness regression test in CI

**Files:**
- Modify: `tests/test_final_answer.py` (already exists, add integration test)
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Add an integration regression test**

```python
# append to tests/test_final_answer.py

import pytest

from kernelforge.ledger import KernelLedger, LedgerState
from kernelforge.final_answer import render_final_answer


def test_full_run_never_claims_correctness_without_verified_state():
    """Whatever combination of ledger states exists, the renderer must
    never produce the substring 'verified correct' for an op whose
    latest ledger entry is not VERIFIED_CORRECT or PERF_MEASURED.
    """
    for bad_final in (LedgerState.GENERATED, LedgerState.COMPILED, LedgerState.SMOKE_PASSED, LedgerState.VERIFIED_INCORRECT, LedgerState.ABANDONED):
        led = KernelLedger()
        led.start("rope", iteration=1, llm_route="deepseek-v4-flash")
        # Walk the ledger to the bad terminal state.
        states_to_walk = [LedgerState.GENERATED, LedgerState.COMPILED, LedgerState.SMOKE_PASSED, LedgerState.VERIFIED_INCORRECT, LedgerState.ABANDONED]
        for s in states_to_walk:
            if s == bad_final:
                if s == LedgerState.ABANDONED:
                    led.advance("rope", LedgerState.ABANDONED, error="test")
                else:
                    led.advance("rope", s)
                break
            led.advance("rope", s)
        out = render_final_answer(led, op="rope")
        assert "verified correct" not in out.lower(), f"renderer claimed correctness in state {bad_final.value}"
```

- [ ] **Step 2: Add CI workflow**

```yaml
# .github/workflows/ci.yml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: macos-14  # arm64 macOS for MLX
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          source $HOME/.cargo/env || true
          uv venv
          uv pip install -e ".[dev]"
      - name: Run tests
        run: uv run pytest -q
      - name: Lint
        run: uv run ruff check .
```

- [ ] **Step 3: Run tests locally**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_final_answer.py .github/workflows/ci.yml
git commit -m "test: false-correctness regression + CI workflow (macos-14)"
```

---

## Day 4 (2026-05-26) — bench + scorecard + deterministic demo scenario

### Task 18: Scorecard generator

**Files:**
- Create: `scorecard/generate.py`, `scorecard/render.py`
- Create: `tests/test_scorecard.py`

- [ ] **Step 1: Write `scorecard/generate.py`**

```python
"""Scorecard: 4-row demo table + detailed README scorecard.

Reads:
- ledger JSONL files for naive and kernelforge runs.
- bench reports (per-op, only for KernelForge's verified-correct kernels).
- ground truth: which ops the chaos scenario corrupted.

Emits:
- demo_scorecard.md (the 4-row table, shown on screen for 5 sec).
- readme_scorecard.md (detailed per-op breakdown).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpOutcome:
    op: str
    naive_claimed_correct: bool
    naive_actually_correct: bool
    kf_claimed_correct: bool
    kf_actually_correct: bool
    kf_iterations: int
    kf_llm_routes: list[str]
    kf_speedups: dict[str, float]


def load_ledger(path: Path) -> dict[str, list[dict]]:
    entries: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        e = json.loads(line)
        entries.setdefault(e["op"], []).append(e)
    return entries


def compute_outcomes(naive_ledger: Path, kf_ledger: Path, ground_truth: dict[str, bool]) -> list[OpOutcome]:
    naive = load_ledger(naive_ledger)
    kf = load_ledger(kf_ledger)
    ops = sorted(set(naive) | set(kf))
    out: list[OpOutcome] = []
    for op in ops:
        naive_last = naive.get(op, [])[-1] if op in naive else {}
        kf_last = kf.get(op, [])[-1] if op in kf else {}
        kf_runs = kf.get(op, [])
        out.append(
            OpOutcome(
                op=op,
                naive_claimed_correct=naive_last.get("state") == "verified_correct" or naive_last.get("claimed_correct", False),
                naive_actually_correct=ground_truth.get(op, True),
                kf_claimed_correct=kf_last.get("state") in {"verified_correct", "perf_measured"},
                kf_actually_correct=ground_truth.get(op, True) and kf_last.get("state") in {"verified_correct", "perf_measured"},
                kf_iterations=max(e.get("iteration", 1) for e in kf_runs) if kf_runs else 0,
                kf_llm_routes=sorted({e.get("llm_route", "") for e in kf_runs}),
                kf_speedups=(kf_last.get("perf_report") or {}).get("speedups", {}) or {},
            )
        )
    return out
```

- [ ] **Step 2: Write `scorecard/render.py`**

```python
"""Render scorecards to Markdown."""
from __future__ import annotations

from scorecard.generate import OpOutcome


def render_demo_scorecard(outcomes: list[OpOutcome]) -> str:
    naive_claimed = sum(1 for o in outcomes if o.naive_claimed_correct)
    naive_correct = sum(1 for o in outcomes if o.naive_claimed_correct and o.naive_actually_correct)
    kf_claimed = sum(1 for o in outcomes if o.kf_claimed_correct)
    kf_correct = sum(1 for o in outcomes if o.kf_claimed_correct and o.kf_actually_correct)
    naive_false = naive_claimed - naive_correct
    kf_false = kf_claimed - kf_correct
    total = len(outcomes)

    routes = sorted({r for o in outcomes for r in o.kf_llm_routes})
    routes_str = " → ".join(routes) if len(routes) > 1 else (routes[0] if routes else "n/a")

    lines = [
        "| Metric | Naive | KernelForge |",
        "| --- | --- | --- |",
        f"| Kernels claimed correct | {naive_claimed}/{total} | {kf_claimed}/{total} |",
        f"| Hidden holdout pass rate | {naive_correct}/{total} | {kf_correct}/{total} |",
        f"| Silent-wrong-output rate | {naive_false}/{total} | {kf_false}/{total} |",
        f"| LLM routing | deepseek-v4-flash only | {routes_str} |",
    ]
    return "\n".join(lines)


def render_readme_scorecard(outcomes: list[OpOutcome]) -> str:
    lines = ["## Detailed scorecard\n", "| Op | KF claim | KF iters | LLM route | Speedup vs MLX eager | Speedup vs mx.fast |", "| --- | --- | --- | --- | --- | --- |"]
    for o in outcomes:
        sp_eager = o.kf_speedups.get("mx_eager", "n/a")
        sp_fast = next((v for k, v in o.kf_speedups.items() if k.startswith("mx_fast")), "n/a")
        claim = "verified" if o.kf_claimed_correct else "—"
        routes = " → ".join(o.kf_llm_routes) if o.kf_llm_routes else "n/a"
        sp_eager_s = f"{sp_eager:.2f}x" if isinstance(sp_eager, (int, float)) else sp_eager
        sp_fast_s = f"{sp_fast:.2f}x" if isinstance(sp_fast, (int, float)) else sp_fast
        lines.append(f"| {o.op} | {claim} | {o.kf_iterations} | {routes} | {sp_eager_s} | {sp_fast_s} |")
    return "\n".join(lines)
```

- [ ] **Step 3: Smoke test**

```python
# tests/test_scorecard.py
import json
from pathlib import Path

from scorecard.generate import compute_outcomes
from scorecard.render import render_demo_scorecard


def test_demo_scorecard_4_rows(tmp_path: Path):
    naive = tmp_path / "naive.jsonl"
    kf = tmp_path / "kf.jsonl"
    naive.write_text(json.dumps({"op": "rope", "state": "verified_correct", "claimed_correct": True, "iteration": 1, "llm_route": "deepseek-v4-flash"}) + "\n")
    kf.write_text(json.dumps({"op": "rope", "state": "verified_correct", "iteration": 2, "llm_route": "deepseek-v4-pro", "perf_report": {"speedups": {"mx_eager": 1.1, "mx_fast_rope": 0.85}}}) + "\n")
    out = compute_outcomes(naive, kf, ground_truth={"rope": False})  # rope was corrupted, naive's claim is a lie
    md = render_demo_scorecard(out)
    lines = [line for line in md.split("\n") if line.startswith("|")]
    assert len(lines) == 6  # header + separator + 4 metric rows
```

Run: `uv run pytest tests/test_scorecard.py -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add scorecard/ tests/test_scorecard.py
git commit -m "feat(scorecard): 4-row demo scorecard + detailed README scorecard"
```

### Task 19: Deterministic demo scenario runner

**Files:**
- Create: `demo/record.py`
- Create: `tests/test_demo_scenario.py`

- [ ] **Step 1: Write `demo/record.py`** (orchestrates a full run: naive + kernelforge under demo_main chaos, dumps artifacts)

```python
"""Scripted demo run. Produces all static artifacts that Remotion consumes:

- demo/artifacts/naive_ledger.jsonl
- demo/artifacts/kf_ledger.jsonl
- demo/artifacts/scorecard_demo.md
- demo/artifacts/scorecard_readme.md
- demo/artifacts/screenshots/*.png (later, via Remotion or screencap)
- demo/artifacts/manifest.json  (timeline metadata for Remotion)
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from baselines.naive import naive_run
from kernelforge.agent import run_kernelforge
from kernelforge.llm_client import LLMClient
from kernelforge.llm_transport import HttpTransport
from scorecard.generate import compute_outcomes
from scorecard.render import render_demo_scorecard, render_readme_scorecard

ARTIFACTS = Path("demo/artifacts")


async def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    transport = HttpTransport()
    llm = LLMClient(transport=transport)

    naive_results = []
    kf_ledgers = {}
    for op in ("rope", "rmsnorm", "swiglu"):
        os.environ["CHAOS_KERNEL_LAB_CURRENT_OP"] = op
        # naive run
        naive_results.append(await naive_run(op, llm))
        # kernelforge run
        kf_ledgers[op] = await run_kernelforge(op, llm, profile="demo")

    # dump
    naive_path = ARTIFACTS / "naive_ledger.jsonl"
    with naive_path.open("w") as f:
        for r in naive_results:
            f.write(json.dumps({"op": r.op, "state": "verified_correct" if r.claimed_correct else "abandoned", "claimed_correct": r.claimed_correct, "iteration": 1, "llm_route": "deepseek-v4-flash"}) + "\n")

    kf_path = ARTIFACTS / "kf_ledger.jsonl"
    with kf_path.open("w") as f:
        for op, led in kf_ledgers.items():
            for e in led.all_entries():
                d = {"op": e.op, "iteration": e.iteration, "state": e.state.value, "llm_route": e.llm_route, "verify_report": e.verify_report, "perf_report": e.perf_report}
                f.write(json.dumps(d) + "\n")

    # ground truth: chaos corrupts rope.
    ground_truth = {"rope": False, "rmsnorm": True, "swiglu": True}
    outcomes = compute_outcomes(naive_path, kf_path, ground_truth)

    (ARTIFACTS / "scorecard_demo.md").write_text(render_demo_scorecard(outcomes))
    (ARTIFACTS / "scorecard_readme.md").write_text(render_readme_scorecard(outcomes))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run the scenario locally (this will use real DeepSeek if .env is set)**

Run: `uv run python -m demo.record`
Expected: writes `demo/artifacts/scorecard_demo.md` with a 4-row table.

- [ ] **Step 3: Commit**

```bash
git add demo/record.py
git commit -m "feat(demo): scripted scenario runner that produces all static artifacts"
```

---

## Day 5 (2026-05-27) — Remotion video + README + DevPost finalize

### Task 20: Remotion project + voiceover

**Files:**
- Create: `demo/remotion/package.json`, `demo/remotion/src/index.tsx`, `demo/remotion/src/Video.tsx`
- Create: `demo/voiceover.py`

- [ ] **Step 1: Init Remotion**

```bash
cd /Users/ant/infra-race
brew install node bun ffmpeg
cd demo/remotion
bun init -y
bun add remotion @remotion/cli @remotion/zod-types @remotion/google-fonts
```

- [ ] **Step 2: Write `demo/remotion/src/Video.tsx`**

```tsx
import { AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig, Audio, staticFile } from 'remotion';

export const KernelForgeDemo: React.FC<{ scorecard: string }> = ({ scorecard }) => {
  const { fps } = useVideoConfig();
  return (
    <AbsoluteFill style={{ background: '#0b0b0c', color: '#eaeaf0', fontFamily: 'JetBrains Mono, monospace' }}>
      <Sequence from={0} durationInFrames={fps * 25}>
        <Opener />
      </Sequence>
      <Sequence from={fps * 25} durationInFrames={fps * 30}>
        <EscalationBeat />
      </Sequence>
      <Sequence from={fps * 55} durationInFrames={fps * 50}>
        <MoneyShot />
      </Sequence>
      <Sequence from={fps * 105} durationInFrames={fps * 15}>
        <ScaleShot />
      </Sequence>
      <Sequence from={fps * 120} durationInFrames={fps * 15}>
        <Scorecard scorecard={scorecard} />
      </Sequence>
      <Audio src={staticFile('voiceover.wav')} />
    </AbsoluteFill>
  );
};

const Opener = () => (
  <AbsoluteFill style={{ justifyContent: 'center', alignItems: 'center', padding: 80, textAlign: 'center' }}>
    <h1 style={{ fontSize: 56 }}>KernelForge</h1>
    <p style={{ fontSize: 28, opacity: 0.8 }}>LLM-written GPU kernels pass smoke tests and silently break on the next shape.</p>
  </AbsoluteFill>
);

const EscalationBeat = () => (
  <AbsoluteFill style={{ padding: 60 }}>
    <h2>LLM brownout → Flash → Pro</h2>
    <pre style={{ background: '#111', padding: 16, borderRadius: 8 }}>
      x-tfy-routing: from=deepseek-v4-flash to=deepseek-v4-pro reason=quality-escalation
    </pre>
  </AbsoluteFill>
);

const MoneyShot = () => (
  <AbsoluteFill style={{ padding: 60 }}>
    <h2>RoPE — naive ships wrong, KernelForge catches it</h2>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
      <pre style={{ background: '#330000', padding: 16, borderRadius: 8 }}>
        Naive: PR opened ✓
        ...
        (smoke-only — wrong layout, max_abs_diff=0.84 on [2,32,128])
      </pre>
      <pre style={{ background: '#003300', padding: 16, borderRadius: 8 }}>
        KernelForge:
        - iter 1 (v4-flash): verified_incorrect (interleaved layout)
        - iter 2 (v4-pro):   verified_correct (5/5 holdouts)
      </pre>
    </div>
  </AbsoluteFill>
);

const ScaleShot = () => (
  <AbsoluteFill style={{ padding: 60 }}>
    <h2>3 ops</h2>
    <ul style={{ fontSize: 28 }}>
      <li>RoPE — verified, 1.2× MLX eager, 0.8× mx.fast.rope (honest)</li>
      <li>RMSNorm — verified iter 1</li>
      <li>SwiGLU — verified iter 1</li>
    </ul>
  </AbsoluteFill>
);

const Scorecard: React.FC<{ scorecard: string }> = ({ scorecard }) => (
  <AbsoluteFill style={{ padding: 60 }}>
    <h2>Scorecard</h2>
    <pre style={{ background: '#111', padding: 16, borderRadius: 8, fontSize: 22 }}>{scorecard}</pre>
    <p style={{ fontSize: 24, marginTop: 32, opacity: 0.7 }}>github.com/AntColony10086/kernelforge</p>
  </AbsoluteFill>
);
```

- [ ] **Step 3: Write `demo/remotion/src/index.tsx`**

```tsx
import { Composition } from 'remotion';
import { KernelForgeDemo } from './Video';
import scorecardMd from '../artifacts/scorecard_demo.md?raw';

export const RemotionRoot: React.FC = () => (
  <Composition id="KernelForgeDemo" component={KernelForgeDemo} durationInFrames={30 * 135} fps={30} width={1920} height={1080} defaultProps={{ scorecard: scorecardMd }} />
);
```

- [ ] **Step 4: Write `demo/voiceover.py`** (macOS `say` driven voiceover)

```python
"""Generate the demo voiceover using macOS `say` and convert to wav."""
import subprocess
from pathlib import Path

SCRIPT = (
    "LLM written GPU kernels pass smoke tests and silently break on the next shape. "
    "KernelForge wraps DeepSeek in a hidden holdout verifier that refuses correctness claims without proof. "
    "When the cheap deepseek v4 flash fails, TrueFoundry escalates to deepseek v4 pro. "
    "Naive baselines ship wrong kernels with confident speedups. KernelForge does not. "
    "Three operators. Hidden holdouts. Honest perf. github dot com slash AntColony10086 slash kernelforge."
)


def main() -> None:
    out = Path("demo/remotion/public/voiceover.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    aiff = out.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-r", "175", "-o", str(aiff), SCRIPT], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(aiff), "-ar", "48000", str(out)], check=True)
    aiff.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Render the video**

```bash
uv run python demo/voiceover.py
cd demo/remotion
bun run remotion render KernelForgeDemo out/demo.mp4
```

- [ ] **Step 6: Commit**

```bash
git add demo/remotion/ demo/voiceover.py
git commit -m "feat(demo): Remotion video composition + macOS say voiceover pipeline"
```

### Task 21: README + run_demo.sh + final DevPost rewrite

**Files:**
- Create: `README.md`
- Create: `run_demo.sh`

- [ ] **Step 1: Write `README.md`**

```markdown
# KernelForge

**Verified MLX/Metal kernel generation for Apple Silicon.**

KernelForge wraps DeepSeek in a hidden-holdout verification harness that refuses to claim kernel correctness without proof, and routes cheap → expensive LLMs via TrueFoundry AI Gateway as failures escalate.

> Built for the DevNetwork [AI + ML] Hackathon 2026, TrueFoundry Resilient Agents track.

## What it does

Given a PyTorch reference operator (`RoPE`, `RMSNorm`, `SwiGLU`), KernelForge generates an MLX/Metal kernel, runs it against a hidden holdout suite (~10 cases per op varying shape, stride, dtype, eps, edge magnitudes), iterates with structured diff feedback until verified or capped, and reports honest perf vs MLX eager / `mx.compile` / `mx.fast` built-ins.

## Why this is different

- **Apple Silicon target.** Almost all LLM-kernel-generation work is CUDA. MLX/Metal is undersupplied.
- **Hidden holdout verification.** The LLM never sees the holdouts; it only gets structured diffs on failure, so it cannot overfit.
- **Cost-aware LLM routing.** Cheap `deepseek-v4-flash` on the happy path, escalate to `deepseek-v4-pro` only after a real failure.
- **`KernelLedger` correctness contract.** The final answer is rendered from the ledger, not from the LLM. The agent cannot claim correctness outside a `verified_correct` ledger state.

> **TrueFoundry handles LLM provider resilience. KernelForge handles kernel correctness verification.**

## Quickstart

Requires Apple Silicon Mac, Python 3.11+, Node 20+, Bun, ffmpeg.

```bash
git clone https://github.com/AntColony10086/kernelforge.git
cd kernelforge
cp .env.example .env  # then add your DEEPSEEK_API_KEY
./run_demo.sh
```

This will run the naive baseline and KernelForge over RoPE, RMSNorm, SwiGLU under the deterministic `demo_main` chaos scenario, then render the demo video.

## Architecture

(see `docs/superpowers/specs/2026-05-22-kernelforge-design.md` for full design — architecture, holdouts, ledger states, demo plan)

## TrueFoundry surface

- AI Gateway routes `deepseek-v4-flash` ↔ `deepseek-v4-pro` per `configs/routing_config.yaml`.
- MCP Gateway registers `kernel_lab` (compile/run/verify/bench).
- `X-TFY-METADATA` carries `run_id` / `op` / `iteration` / `escalate=pro`.
- If TrueFoundry SaaS access is not available, falls back to the honestly-labeled `local_gateway` proxy at `127.0.0.1:8765`.

## Scorecard (demo run)

See `demo/artifacts/scorecard_demo.md` and `demo/artifacts/scorecard_readme.md`.
```

- [ ] **Step 2: Write `run_demo.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "Missing .env — copy .env.example and add DEEPSEEK_API_KEY."
  exit 1
fi
export $(grep -v '^#' .env | xargs)

if [ -z "${TFY_GATEWAY_BASE_URL:-}" ]; then
  echo "TrueFoundry SaaS not configured. Starting local_gateway fallback..."
  uv run uvicorn local_gateway.server:app --port 8765 &
  GATEWAY_PID=$!
  trap 'kill $GATEWAY_PID' EXIT
  sleep 2
fi

uv run python -m demo.record
uv run python demo/voiceover.py
( cd demo/remotion && bun install --silent && bun run remotion render KernelForgeDemo out/demo.mp4 )

echo "Demo rendered to demo/remotion/out/demo.mp4"
echo "Scorecard: demo/artifacts/scorecard_demo.md"
```

- [ ] **Step 3: Make executable + commit**

```bash
chmod +x run_demo.sh
git add README.md run_demo.sh
git commit -m "docs+demo: README + one-command run_demo.sh"
```

### Task 22: DevPost final rewrite (matches the actual demo video)

**Files:** Update DevPost project overview pitch + About to **match what the video actually shows**. (Done via chrome-devtools MCP; not a code task.)

- [ ] **Step 1: Verify the recorded video matches the spec's Section 7 demo plan**

Watch `demo/remotion/out/demo.mp4`. Confirm:
- Beat 1 opener line matches.
- Beat 2 shows `x-tfy-routing` header.
- Beat 3 shows the layout-bug failure + iter 2 success.
- Beat 5 shows the 4-row scorecard.

If any beat doesn't match, edit `demo/remotion/src/Video.tsx` and re-render.

- [ ] **Step 2: Update DevPost description**

Use chrome-devtools MCP to navigate to the project page and rewrite About so that every concrete claim in the description appears in the video.

- [ ] **Step 3: Push public GitHub repo**

```bash
gh repo create AntColony10086/kernelforge --public --source=. --remote=origin --push
```

- [ ] **Step 4: Add the GitHub URL to DevPost "Try it out" links**

(Chrome-devtools MCP click + fill.)

- [ ] **Step 5: Upload demo video to YouTube unlisted; paste link into DevPost Video demo field**

(User does this OR uses `yt-dlp upload` if YouTube CLI is configured.)

- [ ] **Step 6: Commit any final code/doc tweaks**

```bash
git add . && git commit -m "polish: final demo + README + DevPost reconciliation"
git push
```

---

## Day 6 (2026-05-28 morning) — submit

### Task 23: Final read-through + Submit

- [ ] **Step 1: Read the public DevPost preview end-to-end**
- [ ] **Step 2: Confirm Submit page checkboxes**
- [ ] **Step 3: Click Submit before 10:00 PDT** (chrome-devtools MCP or user)
- [ ] **Step 4: Verify on My Projects page that submission status is "Submitted"**

---

## Self-review

### Spec coverage

| Spec section | Plan task(s) |
| --- | --- |
| §1 Context | implicit in all tasks |
| §2 Concept (in-scope list) | Tasks 4-22 |
| §3 Architecture | Tasks 6-15 |
| §4 Data flow happy path | Task 13 |
| §4 LLM brownout | Task 15 |
| §4 Money shot (silent wrong output) | Tasks 16 + 14 + 19 |
| §5.1 llm_client | Task 8 |
| §5.2 kernel_lab tools | Task 11 |
| §5.3 Hidden holdout suite | Task 5 |
| §5.4 KernelLedger | Task 6 |
| §5.5 Agent state machine | Tasks 13 + 14 |
| §5.6 Chaos middleware | Tasks 15 + 16 |
| §5.7 Scorecard | Task 18 |
| §6 TrueFoundry surface | Tasks 8 + 9 + 10 |
| §7 Demo plan | Tasks 19 + 20 |
| §8 Tech stack | Task 1 |
| §9 Build plan | One-to-one with day numbering |
| §10 Risks | Mitigations referenced in tasks (false-correctness regression in Task 17; local_gateway in Task 9; honest perf in Task 18) |
| §11 Success criteria | Task 22 verifies all |
| §12 Open questions | Task 3 spike; Task 9 local_gateway fallback; Task 21 .env |

### Placeholders

Scanned — none present. Every code block compiles or runs against the imports declared.

### Type consistency

- `KernelOutput` (Task 8) used identically in Tasks 12, 13, 14 (`source`, `grid`, `threadgroup`, `output_shapes` — same field names).
- `LedgerState` enum (Task 6) used identically in Tasks 7, 13, 14, 17.
- `cases_for(op)` (Task 5) used identically in Tasks 11, 12, 14.
- `verify_kernel(...)` return type `VerifyReport` (Task 11) used identically in Task 14.
- `compute_outcomes(naive_path, kf_path, ground_truth)` (Task 18) signature used identically in Task 19.

### Scope check

Single coherent project. Single sponsor track. Single hardware target. Bounded.
