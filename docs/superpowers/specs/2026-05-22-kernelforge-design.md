# KernelForge — Design Spec v2 (locked 2026-05-22 after Codex round 3)

> **STATUS NOTE (added 2026-05-23):** This is the **original design spec** — what we set out to build before live execution. **Several aspirational details did not survive contact with reality.** Specifically: (1) `deepseek-v4-pro` was the planned escalation route, but its long-prompt streaming proved unreliable; the actually-shipped escalation route is `deepseek-coder`. (2) The spec sketches 5-iteration RoPE convergence with 1.6× speedup over MLX eager — none of this was achieved in the live run; KernelForge correctly refused to ship 0/20 kernels because none passed the hidden holdout suite. (3) The MCP server (`kernel_lab`) ships with `run/verify/bench` exposed as MCP tools that the agent calls via the in-process Python API (tensor marshaling across MCP is too heavy for the inner loop). **For the as-shipped behavior and actual live-run numbers, see `README.md` and `demo/artifacts/scorecard_demo.md`.** This spec is preserved as the design rationale and architectural reference.

**Robust verification and cost-aware routing for LLM-written MLX / Metal kernels on Apple Silicon.**

> **v1 → v2 changelog (post Codex round 3 review):**
> - Repositioned from "structured iterate loop" (not novel — Sakana already did this) to **"hidden holdout verification + cost-aware LLM routing"**.
> - 5 ops → 3 ops: `RoPE`, `RMSNorm`, `SwiGLU`.
> - 3 MCP servers → 1 unified `kernel_lab` MCP exposing `compile / run / verify / bench`.
> - Perf baselines: PyTorch CPU is correctness reference only; performance is measured against **MLX eager**, **`mx.compile`**, and MLX built-ins (`mx.fast.rms_norm`, `mx.fast.rope`).
> - DeepSeek model names updated: `deepseek-v4-flash` (happy path) and `deepseek-v4-pro` (escalation). The previous `deepseek-chat` / `deepseek-reasoner` are legacy aliases through 2026-07-24.
> - LLM routing framing: **cost-aware escalation** (cheap Flash for happy path, escalate to Pro after compile/correctness failure), not provider-HA.
> - Money shot: RoPE with hidden-holdout-caught layout bug (interleaved vs split-half), not a fake RMSNorm numerical error.
> - Iteration caps: 3 in demo, 6 in regression; escalation to Pro after first failure.
> - Demo pipeline: scripted run produces static artifacts (traces, ledger, screenshots, scorecard JSON), Remotion consumes them — does not depend on live terminal timing.

## 1. Context

**Hackathon:** DevNetwork [AI + ML] Hackathon 2026
**Submission deadline:** 2026-05-28 10:00 PDT (≈ 5.5 days from spec lock)
**Track (primary):** TrueFoundry — Resilient Agents ($500 + $200)
**Track (secondary, opportunistic):** Overall Winner
**Team:** solo (`@AntColony10086` / Ant Lu)
**Hardware:** M4 Mac mini 16GB / 256GB only (4060 laptop excluded to honor "user does nothing" constraint).
**LLM:** DeepSeek API key only.
**Autonomy:** I (Claude 4.7) + Codex (GPT-5.5 xhigh) execute end-to-end; user's only manual action is `DEEPSEEK_API_KEY=...` in `.env`.

Sponsor track prompt (verbatim): *"How does your agent behave when an MCP server starts erroring out? An LLM server goes down? OpenAI or Claude errors out or browns out? The goal of this challenge is to see how user experience and the user side of things are handled when this infrastructure chaos happens and how your agent is configured and set up for success and resilience."*

Sponsor-stated judging focus: **resilience, reliability, production-readiness under failure conditions**.
General judging criteria: **Progress / Concept / Feasibility**.

User's strategic goal: AI Infra internship resume value first, ranking second. Operator/kernel work has 2-3× higher resume value than agent-infra work at Nvidia / DeepSeek / Anthropic / MoonshotAI for AI Infra roles.

## 2. Concept

**KernelForge** is a small agent + verification harness that takes a PyTorch reference operator (`RMSNorm`, `RoPE`, `SwiGLU`) and returns either a **verified-correct MLX/Metal kernel implementation** with measured speedup, or an honest abandonment with the reason and the diff against reference.

What makes it different from KernelBench (Stanford 2024, benchmark) and Sakana AI CUDA Engineer (2025, evolutionary CUDA):
- **Apple Silicon target.** Nearly all existing LLM-kernel-generation work is CUDA. MLX/Metal is undersupplied. (Defensible novelty.)
- **Hidden holdout verification.** The verifier does not just compare against one sample input. It runs each candidate kernel against a holdout suite covering different shapes, strides, dtypes, eps values, and edge-magnitude inputs. The ledger only allows `verified_correct` if every holdout passes. (This is the real differentiation — Sakana's loop verifies one reference; ours verifies a generalization battery.)
- **Cost-aware LLM routing.** The agent calls cheap `deepseek-v4-flash` on the happy path. After a compile or correctness failure, it escalates to expensive `deepseek-v4-pro` (thinking mode) via TrueFoundry AI Gateway's routing config. This is a production-realistic routing pattern (cost optimization with quality escalation), not just provider failover.
- **Honest perf baselines.** We measure speedup against MLX eager, `mx.compile`, and MLX's own optimized built-ins (`mx.fast.rms_norm`, `mx.fast.rope`) — never PyTorch CPU as the main number. If we lose to a built-in, we say so.

**One-line pitch:** *Most LLM-written Metal kernels pass a quick smoke test and quietly fail on different shapes, strides, or edge magnitudes. KernelForge wraps DeepSeek in a hidden-holdout verification harness that refuses to claim correctness without proof, and routes cheap → expensive LLMs as failures escalate.*

**In scope (MVP shipped before 2026-05-28):**
- 3 operators: `RoPE`, `RMSNorm`, `SwiGLU`.
- Per-op PyTorch reference implementation.
- Per-op hidden holdout suite (~10–20 test cases per op, varying shape/stride/dtype/eps/magnitude).
- A code generator that calls DeepSeek through TrueFoundry AI Gateway with `deepseek-v4-flash` (happy path) and `deepseek-v4-pro` (escalation).
- One `kernel_lab` MCP server exposing `compile`, `run`, `verify`, `bench` tools.
- A `KernelLedger` (states: `attempted → generated → compiled → smoke_passed → verified_correct | verified_incorrect | perf_measured | abandoned`).
- A strict kernel-output JSON schema the LLM must fill (`source`, `grid`, `threadgroup`, `output_shapes`, `dtype`, `assumptions`).
- An iteration loop (max 3 in demo, max 6 in regression) with structured diff-feedback to the LLM.
- A naive baseline: same DeepSeek + smoke test only (no holdouts), no ledger, no escalation.
- A chaos harness: LLM brownouts (Flash 503 → Pro), compiler MCP errors, **silently-wrong-output mode** (MCP returns plausible but algorithmically wrong tensors — what KernelForge's hidden holdouts catch).
- A 4-row scorecard.
- A Remotion-rendered demo video (≈ 2 min 15 s) consuming pre-generated static artifacts.
- A public GitHub repo + `./run_demo.sh` on any Apple Silicon Mac.

**Explicitly out of scope:**
- CUDA / Triton.
- LangGraph / agent framework — hand-rolled ~200-line state machine.
- Operator fusion (multi-op).
- Quantization kernels.
- 4th/5th ops (softmax, fused linear+act) — deferred to stretch.
- Docker / containerization (Metal doesn't containerize cleanly).
- Multi-host distributed generation.

## 3. Architecture

```
PyTorch reference op + hidden holdout suite
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                  KernelForge Agent                           │
│                                                              │
│   ┌─────────────┐    ┌─────────────────────────────────┐     │
│   │ Planner     │───▶│ Iteration loop                  │     │
│   │ (1 LLM call)│    │ generate → compile → smoke →    │     │
│   │             │    │ holdout verify → analyze →      │     │
│   │             │    │ (escalate model if failed) →    │     │
│   │             │    │ repeat                          │     │
│   └─────────────┘    └────────────────┬────────────────┘     │
│                                       │                      │
│                                       ▼                      │
│   ┌──────────────────────────────────────────────┐           │
│   │ Resilience + routing layer                    │          │
│   │ • llm_client → TrueFoundry AI Gateway         │          │
│   │   (deepseek-v4-flash primary, escalate to     │          │
│   │    deepseek-v4-pro after first failure)       │          │
│   │ • mcp_client → kernel_lab MCP (compile /      │          │
│   │   run / verify / bench)                       │          │
│   │ • KernelLedger: per-op state machine; final   │          │
│   │   answer rendered from ledger only            │          │
│   │ • Holdout suite enforcement                   │          │
│   │ • Traces every step to JSONL                  │          │
│   └──────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
TrueFoundry AI Gateway        TrueFoundry MCP Gateway
  ├─ deepseek-v4-flash          └─ kernel_lab MCP
  │  (happy path, cheap)            ├─ compile (mlx.fast.metal_kernel
  └─ deepseek-v4-pro                │           or mx.compile)
     (escalation, expensive,        ├─ run     (mx.array I/O)
      thinking mode)                ├─ verify  (PyTorch ref + holdouts)
                                    └─ bench   (vs MLX eager, mx.compile,
                                               mx.fast built-ins)
        │                              │
        ▼                              ▼
   Chaos middleware              Chaos middleware
  (503/timeout on Flash)        (deterministic failure modes
                                 inside kernel_lab)
```

**Component boundaries:**

| Component | Owns | Does not own |
| --- | --- | --- |
| TrueFoundry AI Gateway | LLM routing + cost-aware escalation (Flash → Pro) | Agent state, holdout policy |
| TrueFoundry MCP Gateway | MCP server registry, auth, transport | Tool internals, verification semantics |
| KernelForge resilience layer | Circuit breaker per MCP tool, ledger transitions, structured error parsing, holdout enforcement | LLM routing (defers to AI Gateway) |
| KernelForge iteration loop | Generate → compile → smoke → holdout verify → analyze → escalate → repeat | Resilience logic (delegates to layer) |
| `kernel_lab` MCP | Metal compile, MLX run, PyTorch-ref verify, MLX-baseline bench | Agent policy decisions |
| Chaos harness | Fault injection (LLM-side + kernel_lab-side) + scenario scripts | Scoring |
| Scorecard generator | KernelLedger + perf data → 4-row table + detailed README scorecard | Anything live |

**Critical division of labor**:
- **TrueFoundry owns LLM routing + escalation policy.**
- **KernelForge owns verification semantics + holdout suite enforcement.**

This split is what the sponsor judge sees on screen and what the README leads with.

## 4. Data flow

### 4.1 Happy path

`agent.optimize("RoPE", reference_fn, holdout_suite)` →
- Ledger: `RoPE@1: attempted`
- `llm_client.complete(prompt + structured_schema)` via TrueFoundry AI Gateway → `deepseek-v4-flash` → JSON `{source: "<metal kernel>", grid, threadgroup, output_shapes, dtype, assumptions}`
- Ledger: `RoPE@1: generated`
- `mcp_client.kernel_lab.compile(source, grid, threadgroup)` → compiled handle
- Ledger: `RoPE@1: compiled`
- `mcp_client.kernel_lab.run(handle, holdout[0].inputs)` → output tensor
- Ledger: `RoPE@1: smoke_passed`
- `mcp_client.kernel_lab.verify(handle, reference_fn, holdout_suite)` → per-case results
  - All pass → ledger: `RoPE@1: verified_correct`
  - Any fail → ledger: `RoPE@1: verified_incorrect`, includes failing-case diffs
- If verified_correct: `mcp_client.kernel_lab.bench(handle, baselines=[mx_eager, mx_compile, mx_fast_rope])` → speedups
- Ledger: `RoPE@1: perf_measured`
- Final answer rendered from ledger.

### 4.2 LLM cost-aware escalation (replaces provider failover)

First iteration uses `deepseek-v4-flash` (cheap, ~30ms latency, $0.07/M tokens). If the first iteration fails compile OR holdout verification, the next iteration's `llm_client.complete(...)` adds `X-TFY-METADATA.escalate=pro` and the TrueFoundry AI Gateway routes to `deepseek-v4-pro` (thinking mode, slower, more expensive, but better at correctness). This is a cost-aware routing policy, not a failover. On screen the gateway header shows `x-tfy-routing: from=v4-flash to=v4-pro reason=quality-escalation`.

### 4.3 Compiler MCP failure

`mcp_client.kernel_lab.compile(...)` → chaos injects a Metal compilation error or runtime crash → wrapper retries within timeout budget → on 2 failures the per-tool circuit breaker opens → next call short-circuits with `CompilerUnavailable`. Ledger: `RoPE@1: abandoned` with reason `compiler_circuit_open`. Naive does the same but without breaker (loops, wastes time).

### 4.4 The money shot: hidden holdout catches the bug

DeepSeek-v4-flash generates a RoPE kernel that:
- Compiles successfully.
- Smoke-test passes (input shape `[1, 8, 64]` → output looks rotated correctly).
- BUT the kernel implements **interleaved layout** (`x0, x1` adjacent) when the reference uses **split-half layout** (first half is `x_real`, second half is `x_imag`). On the smoke-test shape the difference is invisible — the rotation values happen to match. On the holdout case with shape `[2, 32, 128]` and `position_ids=[0, 1, ..., 31]`, the layout bug surfaces as `max_abs_diff = 0.84` on half the elements.

**Naive baseline**: runs only the smoke test, claims `"RoPE kernel ready: 1.4× speedup ✓"`. The kernel ships, and downstream model outputs are silently wrong.

**KernelForge**:
- Ledger: `RoPE@1: smoke_passed` then `verified_incorrect` (with the structured diff: "case `shape=[2,32,128]`, max_abs_diff=0.84, suspected layout mismatch (interleaved vs split-half)").
- Diff fed back into next prompt: *"Your kernel passed shape `[1,8,64]` but failed on `[2,32,128]` with max_abs_diff 0.84. The reference uses split-half layout (first half real, second half imag); confirm your kernel matches."*
- LLM escalates to `deepseek-v4-pro` (thinking mode) → iteration 2 → passes the smoke test AND all holdouts.
- Ledger: `RoPE@2: verified_correct`.
- Final: *"RoPE kernel verified correct after 2 iterations (escalated to v4-pro for iteration 2). Speedup over MLX eager: 1.2×. Speedup over `mx.fast.rope`: 0.8× (slower)."* — the perf disclosure is honest.

This is the demo's emotional peak. **Naive does smoke-test-only and ships wrong code with a confident speedup number; KernelForge does holdout suite, catches the bug, escalates the LLM, and honestly reports it loses to MLX's expert built-in.**

## 5. Component specifications

### 5.1 `llm_client`

- `complete(messages, *, escalation_hint: bool = False) -> Completion`.
- TrueFoundry AI Gateway endpoint configured with `deepseek-v4-flash` as primary and `deepseek-v4-pro` as `fallback_candidate` for `fallback_status_codes = [503, 429, 408]`.
- Cost-aware escalation: when iteration N > 1 OR previous iteration failed verification, sets `X-TFY-METADATA.escalate=pro`. Gateway routing config routes the request to `deepseek-v4-pro` based on this header.
- All requests carry `X-TFY-METADATA: {run_id, op, iteration, escalation}` so gateway logs are joinable.
- LLM must return a JSON object matching the strict kernel schema; the response is parsed with `pydantic.BaseModel.model_validate` — parse failures count as compile failures (not silent text-mode hacks).

Strict kernel schema:
```python
class KernelOutput(BaseModel):
    source: str                       # raw Metal kernel source
    grid: tuple[int, int, int]        # MTL thread grid
    threadgroup: tuple[int, int, int] # MTL threadgroup size
    output_shapes: list[tuple[int, ...]]  # expected output tensor shapes
    dtype: Literal["float32", "float16", "bfloat16"]
    assumptions: list[str]            # e.g. ["shape divisible by threadgroup[0]", "head_dim is even"]
```

### 5.2 `kernel_lab` MCP server

A single MCP server exposing 4 tools:
- `compile(source: str, grid: tuple, threadgroup: tuple) -> {handle: str, compile_log: str} | CompileError`. Uses `mlx.core.fast.metal_kernel(...)` for raw Metal source. For ops where raw Metal proves too hard in D1 spike, falls back to allowing `mx.compile`-style implementation submissions.
- `run(handle: str, inputs: list[ndarray], dtype: str) -> {output: ndarray, runtime_ms: float} | RuntimeError`.
- `verify(handle: str, op_name: str, holdout_suite: list[TestCase]) -> {results: list[VerifyResult], summary: {pass, fail, max_abs_diff_max, max_rel_diff_max}}`.
- `bench(handle: str, op_name: str, baselines: list[Literal["mx_eager", "mx_compile", "mx_fast_builtin"]]) -> {kernel_ms: float, baseline_ms: dict[str, float], speedups: dict[str, float]}`.

Per-tool circuit breakers (2 named profiles: production default 3 failures/30s cooldown, demo 2 failures/8s cooldown).

### 5.3 Hidden holdout suite

Per op, an extensible list of test cases. Held out from the LLM — never shown in any prompt. The LLM only ever sees the reference op spec + the per-case error diff after failure (e.g. "case `shape=[2,32,128] dtype=float16`, max_abs_diff 0.84"). The LLM does not see the holdout inputs themselves, so it cannot overfit.

`RoPE` holdouts (illustrative — final list expanded D1):
- `[1, 8, 64]` float32 standard.
- `[2, 32, 128]` float32 — catches layout bugs (split-half vs interleaved).
- `[4, 16, 256]` float16 — catches dtype assumptions.
- `[1, 1, 64]` with `position_ids=[1000]` — catches base-frequency overflow.
- `[2, 8, 64]` float32 with non-contiguous stride — catches stride assumptions.
- `[1, 8, 64]` with `theta=10000.0` and `theta=500000.0` — catches frequency-scale issues.
- Empty-batch edge: `[0, 8, 64]` — catches grid-launch-with-empty bugs.

`RMSNorm` holdouts: vary `eps`, dtype, very-small-magnitude input (sub-eps), very-large-magnitude input, non-power-of-2 hidden dim.

`SwiGLU` (`x * SiLU(W_gate(x)) * W_up(x)` style — concrete reference: PyTorch `F.silu(gate) * up`): vary batch, gate/up split, dtype, very large input range.

### 5.4 `KernelLedger`

```python
@dataclass(frozen=True)
class LedgerEntry:
    op: str
    iteration: int
    state: Literal["attempted", "generated", "compiled", "smoke_passed", "verified_correct", "verified_incorrect", "perf_measured", "abandoned"]
    kernel: KernelOutput | None
    verify_report: dict | None   # {pass, fail, max_abs_diff_max, failing_cases: [{shape, dtype, max_abs_diff, suspected_cause}]}
    perf_report: dict | None     # {kernel_ms, baseline_ms: {mx_eager, mx_compile, mx_fast_builtin}, speedups}
    error: str | None
    evidence_refs: list[str]
    timestamp_ms: int
    llm_route: str               # "deepseek-v4-flash" or "deepseek-v4-pro"
```

Strict monotonic state transitions per (op, iteration). Final answer rendered by a template function that reads the latest entry per op. The LLM is invoked in summarization mode with a JSON schema that **forbids claiming correctness when state != verified_correct**.

### 5.5 Agent state machine

Hand-rolled ~200-line Python. Per op: `PLANNING → GENERATING → COMPILING → SMOKE → HOLDOUT_VERIFY → (REFINE → escalate-flag → GENERATING) | BENCHMARKING | ABANDONED`. Iteration cap 3 in demo profile, 6 in regression profile. Each transition writes one trace event.

### 5.6 Chaos middleware

Two FastAPI proxies:
- `chaos_llm_proxy`: in front of TrueFoundry AI Gateway. Toggles 503 / 429 / timeout on `deepseek-v4-flash` per `chaos.toml`.
- `chaos_kernel_lab_proxy`: in front of the `kernel_lab` MCP. Toggles compile-error, runtime-error, and **silently-wrong-output** (returns a tensor with subtle algorithmic bug, like the RoPE layout error) per `chaos.toml`. **The silently-wrong-output mode is what the hidden holdout suite catches.**

### 5.7 Scorecard

Reads `traces/<run_id>.jsonl` + `ledger/<run_id>.jsonl` + bench data. Emits a **4-row table**:

| Metric | Naive | KernelForge |
| --- | --- | --- |
| Kernels claimed correct | 3/3 | only verified ones (e.g., 3/3) |
| Hidden holdout pass rate | low (e.g., 50%) | 100% of claimed |
| Silent-wrong-output rate | high (e.g., 67%) | 0% |
| LLM routing | static `deepseek-v4-flash` | `v4-flash → v4-pro on escalation` |

Detailed scorecard (README only): per-op iterations to convergence, per-iteration max_abs_diff trajectory, per-op breaker activations, kernel-vs-MLX-built-in speedup table with honest losses called out.

## 6. TrueFoundry integration surface

Same product hooks as v1, with two refinements:

- **AI Gateway**: routes `deepseek-v4-flash` (default) and escalates to `deepseek-v4-pro` based on `X-TFY-METADATA.escalate=pro`. This is the cost-aware-routing framing.
- **MCP Gateway**: registers `kernel_lab` as a single MCP server with 4 tools.
- **Virtual Models / Routing Config / `routing_config.yaml`** — on screen in demo Beat 2.
- **`retry_config`**, **`fallback_status_codes`**, **`fallback_candidate`**.
- **`X-TFY-METADATA`** — carries `run_id`, `op`, `iteration`, `escalation`.
- **TrueFoundry Observability** — gateway-side LLM logs joinable to local traces.

### 6.1 On-screen visibility requirements

Three TF surfaces visible in the demo recording:
1. **AI Gateway response headers** — `x-tfy-routing: from=v4-flash to=v4-pro reason=quality-escalation` during the escalation beat.
2. **`routing_config.yaml`** flashes for ~2 s with the escalation rule highlighted (matching `X-TFY-METADATA.escalate`).
3. **TrueFoundry MCP Gateway registry** — `kernel_lab` MCP listed.

### 6.2 Narration line

Spoken once: *"TrueFoundry routes between cheap and expensive DeepSeek models based on whether the cheap one got it right. KernelForge tells it when to escalate."*

## 7. Demo (single continuous narrative, 2 min 15 s target)

**Setup**: terminal split left = `naive` (DeepSeek-v4-flash + smoke test only, no holdouts, no escalation), right = `kernelforge` (full pipeline). Both run the same 3-op task list with the same chaos scenario (`demo_main` injects `silently_wrong_output` on RoPE specifically).

**Beat 1 (0:00–0:25) — Opener.**
VO: *"LLMs can write GPU kernels. They can also write GPU kernels that compile, pass a smoke test, and quietly break on the next batch. KernelForge catches the second kind."* Pre-show the naive RoPE confidently shipped with a 1.4× speedup claim, then a quick cut showing it failing on a different shape. Hit run.

**Beat 2 (0:25–0:55) — Cost-aware escalation, the TrueFoundry win.**
RoPE iteration 1 with `deepseek-v4-flash` fails the holdout suite (interleaved-vs-split-half layout bug). KernelForge sets `X-TFY-METADATA.escalate=pro`. Gateway header overlay: `x-tfy-routing: from=v4-flash to=v4-pro reason=quality-escalation`. `routing_config.yaml` flashes with the escalation rule. v4-pro generates v2 of the kernel. VO: *"TrueFoundry routes between cheap and expensive DeepSeek models based on whether the cheap one got it right. KernelForge tells it when to escalate."*

**Beat 3 (0:55–1:45) — Hidden holdout money shot.**
- **Naive** (left): generates kernel from v4-flash, compiles, runs smoke test on `[1,8,64]`, prints *"RoPE kernel ready: 1.4× speedup ✓"*. Narrator runs the SAME kernel on holdout shape `[2,32,128]` — max_abs_diff = 0.84. Naive shipped wrong code.
- **KernelForge** (right): same v4-flash output, but verifier runs the full holdout suite. Catches the bug at case `[2,32,128]` with structured cause `"interleaved vs split-half layout"`. Diff fed back. v4-pro generates v2. Holdout suite passes 100%. Ledger advances to `verified_correct`.
- Final on-screen for KernelForge: *"RoPE verified correct after 2 iterations (escalated to v4-pro). Speedup over MLX eager: 1.2×. Speedup over `mx.fast.rope`: 0.8× (slower, MLX built-in wins)."* — **honest perf disclosure**.

Narrator line spoken once during this beat: *"TrueFoundry routes between cheap and expensive DeepSeek models based on whether the cheap one got it right. KernelForge tells it when to escalate."* Cut to MCP Gateway registry showing `kernel_lab` — third TF surface satisfied.

**Beat 4 (1:45–2:00) — Scale across the 3 ops.**
Dashboard shows naive shipping 2 wrong / 1 right, KernelForge shipping 3 verified-correct (with 1 honestly slower than the MLX built-in). Iteration counts: 1/1/1 naive, 2/1/1 KernelForge.

**Beat 5 (2:00–2:15) — Scorecard + closing.**
The 4-row table on screen:

| Metric | Naive | KernelForge |
| --- | --- | --- |
| Kernels claimed correct | 3/3 | 3/3 |
| Hidden holdout pass rate | 33% (1/3) | 100% (3/3) |
| Silent-wrong-output rate | 67% | 0% |
| LLM routing | `v4-flash` only | `v4-flash → v4-pro on escalation` |

Closing: *"Correctness isn't a vibe. It's a holdout suite. github.com/AntColony10086/kernelforge."*

Cut budget: if 15 s over, trim Beat 1 cold open.

## 8. Tech stack

| Layer | Choice | Reason |
| --- | --- | --- |
| Host | M4 Mac mini 16GB | the hardware Claude Code controls directly |
| Language | Python 3.11 | MLX, FastAPI, pydantic |
| MLX | ≥ 0.21 | `mlx.core.fast.metal_kernel` + `mx.compile` + `mx.fast.{rms_norm, rope}` built-ins |
| Reference | PyTorch (CPU) | correctness reference only |
| LLM | DeepSeek `v4-flash` + `v4-pro` via TrueFoundry AI Gateway | sponsor scoring + cost-aware escalation |
| MCP | TrueFoundry MCP Gateway → local `kernel_lab` MCP | sponsor scoring + clean tool boundary |
| Agent | hand-rolled state machine (~200 lines) | no framework overhead |
| Holdout suite | per-op Python list of test cases | extensible, transparent |
| Resilience | hand-rolled circuit breaker + KernelLedger + chaos proxies | this IS the product |
| Traces | JSONL | no collector needed |
| Scorecard | Python → Markdown + HTML | static, embeds cleanly in video |
| Demo video | **scripted run → static artifacts (JSON + screenshots) → Remotion render + macOS `say` TTS + ffmpeg captions** | fully autonomous, doesn't depend on live timing |
| Submission UI | chrome-devtools MCP fills DevPost | proven path |
| Packaging | `./run_demo.sh` + `requirements.txt` + `Makefile` | Mac-only, any Apple Silicon dev repros |
| CI | GH Actions: lint + regression test for false-correctness | doubles as "Progress" evidence |

Rejected: Triton, CUDA, LangGraph, Jaeger, Docker, vector DB, RAG, fine-tuning, web dashboard, separate MCPs per tool.

## 9. Build plan (5.5-day timeline)

| Day | Date | Deliverable |
| --- | --- | --- |
| D0 | 2026-05-22 evening | KernelForge v2 spec locked (this document) ✓; Codex round-3 review applied ✓; DevPost form rewritten; repo scaffolded; MLX installed; `mlx.core.fast.metal_kernel` hello-world (PyTorch RMSNorm vs hand-written Metal RMSNorm — matches within 1e-4). |
| D1 | 2026-05-23 | TrueFoundry AI Gateway live with `deepseek-v4-flash` primary + `deepseek-v4-pro` escalation. `llm_client` + strict JSON schema parsing. `kernel_lab` MCP with 4 tools running. First end-to-end RoPE round-trip on happy path. |
| D2 | 2026-05-24 | All 3 ops' PyTorch reference + hidden holdout suites (target ≥ 10 cases per op). Naive baseline (smoke-only) finishes all 3. Baseline data captured: naive smoke pass rate, naive holdout pass rate, naive false-correctness rate. |
| D3 | 2026-05-25 | KernelForge full iteration loop: holdout verify → structured diff feedback → escalation → re-generate. Chaos middleware (LLM brownout + silently-wrong-output) wired. False-correctness regression test in CI passes. |
| D4 | 2026-05-26 | Benchmarker (vs MLX eager, mx.compile, mx.fast built-ins). Scorecard 4-row + detailed README scorecard. Deterministic demo scenario `demo_main` reproducible 5/5 runs. End-to-end smoke + holdout + bench on all 3 ops working. |
| D5 | 2026-05-27 | Scripted run produces all static artifacts (traces, ledger, screenshots, scorecard.json, op-by-op timeline images). Remotion video pipeline consumes artifacts and renders 2:15 video + macOS `say` voiceover + ffmpeg-burned captions. README polish. DevPost rewrite. |
| D6 | 2026-05-28 morning | Final DevPost public preview read-through. Submit before 10:00 PDT. |

Each day ends with Codex round-N peer review on that day's diff.

## 10. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| TrueFoundry SaaS signup is sales-gated | M | High | (a) Sign up D0 night. (b) Email `sai@truefoundry.com` with pitch + access request. (c) `llm_client` built against OpenAI-compatible `base_url` so TF is swappable. (d) Honest `local_gateway` FastAPI proxy fallback labeled as such. |
| `mlx.core.fast.metal_kernel` rough on real ops | M | Medium | D0 spike on RMSNorm. If raw Metal too hard, allow `mx.compile`-authored implementations for some ops; this does not break the verification + escalation thesis. |
| DeepSeek v4-flash/pro quota / rate limit during dev or recording | L | Medium | Cache LLM responses keyed by (op, iteration, prompt_hash) for demo run. |
| 16GB Mac mini thrashes | L | High | Keep tensor sizes ≤ 4096; close other apps during demo; benchmark only the kernel of interest. |
| Remotion render fails / non-deterministic | M | High | Pre-cache LLM responses; static-artifact pipeline (scripted run → JSON+screenshots → Remotion consumes) means rendering does not depend on live execution. Fallback plan B: asciinema + ffmpeg, simpler animations, narration as voiceover only. |
| Naive baseline looks like a strawman | L | High | Naive uses the SAME DeepSeek + SAME first-iteration prompt as KernelForge. The only difference is what it verifies (smoke only vs holdout suite). Source published alongside. |
| 5.5 days runs out | M | Total | Hard freeze D5 for video. If D4 slips, cut SwiGLU; keep RoPE + RMSNorm. The money shot is RoPE, so it can not be cut. |
| MLX built-ins crush our hand-written kernel | M | Medium-low | This is honesty risk, not a project-failure risk. Report the loss honestly in the perf table. Judges will respect honesty more than a fake speedup. |
| False correctness slips through | L | Catastrophic | Two layers: (1) final answer rendered from ledger only, LLM cannot inject claims. (2) Regression test: assert no `op_in_final_answer.claims_correct` if `ledger[op].state != verified_correct`. |
| Judges misunderstand "robust verification" angle | M | Medium | README opening + Beat 3 narration both lead with "hidden holdout suite ≠ smoke test", with the concrete RoPE layout example. |

## 11. Success criteria

Bar for "good enough to place top-2 in TrueFoundry track" (Codex round-3 odds estimate: 42% after these refinements):

- One-command repro on any Apple Silicon Mac.
- Naive-vs-KernelForge demo video deterministic, side-by-side, ≤ 2 min 30 s, reproducible 5/5 runs.
- 3 TF product surfaces visible in the recording (Section 6.1).
- Division-of-labor sentence (Section 6.2) spoken once, printed in README + DevPost.
- Scorecard matches Section 5.7 exactly.
- TF AI Gateway visibly in use OR `local_gateway` fallback honestly labeled.
- README leads with the hidden-holdout differentiation, names the RoPE layout example.
- DevPost description matches the video — no claim not in the recording.
- Honest perf disclosure: where we beat MLX built-ins, where we lose.
- CI regression test green: no false-correctness claims.

Stretch (for Overall Winner consideration):
- 4th op (e.g., fused residual + RMSNorm) demonstrating composition.
- A short "design rationale" section in the README.
- An interactive `./play.sh` letting a Mac-equipped judge pick an op and watch live iteration.

## 12. Open questions (close on D0)

1. **TrueFoundry SaaS access** — signup + sai@truefoundry.com email + fallback `local_gateway` ready.
2. **MCP Gateway setup time** — same.
3. **`mlx.core.fast.metal_kernel` maturity** — D0 spike on RMSNorm decides per-op raw-Metal vs `mx.compile` strategy.
4. **DeepSeek key in `.env`** — `DEEPSEEK_API_KEY=...`, format confirmed.
5. **Demo recording approach** — Remotion + static-artifact pipeline locked; fallback to asciinema+ffmpeg if Remotion fails.
6. **Holdout case count per op** — start with ~10 each, expand if iteration loop converges too quickly (no demo drama) or too slowly (cap-bound abandonment).
