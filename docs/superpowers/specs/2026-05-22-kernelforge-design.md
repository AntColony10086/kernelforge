# KernelForge — Design Spec (locked 2026-05-22)

An LLM agent that writes, verifies, and iteratively optimizes MLX/Metal GPU kernels for Apple Silicon, with built-in correctness verification that refuses to claim kernel correctness without proof.

> **Pivot notice (2026-05-22 evening):** This project pivoted from `Chaosguard — MCP Agent Resilience Runtime` after user clarified the primary goal is AI-infra-flavored resume value (kernel work) rather than agent-infra work. ~60% of the Chaosguard architecture is retained (`OutcomeLedger` → `KernelLedger`, side-effect verification → reference-output verification, naive-vs-resilient demo structure, TrueFoundry gateway integration). Historical Chaosguard spec is preserved at `2026-05-22-chaosguard-design.md` for context.

## 1. Context

**Hackathon:** DevNetwork [AI + ML] Hackathon 2026
**Submission deadline:** 2026-05-28 10:00 PDT (≈ 5.5 days from spec lock)
**Track (primary):** TrueFoundry — Resilient Agents ($500 + $200)
**Track (secondary, opportunistic):** Overall Winner (Amazon Echos + DevNetwork passes + 60K-subscriber email blast)
**Team:** solo (`@AntColony10086` / Ant Lu)

**User's strategic goal**: enrich AI Infra internship resume. Ranking is desirable but not required. Operator-/kernel-flavored work has 2-3× higher resume value at Nvidia, DeepSeek, Anthropic, MoonshotAI, and other AI infra hirers than agent-infra work.

**Hardware available:**
- M4 Mac mini 16GB / 256GB (primary; this is what Claude Code runs on)
- RTX 4060 8GB laptop (NOT used in this project — using it would require SSH setup, violating user's "do nothing" constraint)

**LLM available:** DeepSeek API key only (no OpenAI, no Anthropic, no Groq, no local Ollama).

**Autonomy constraint:** the user requires that I + Codex do everything end-to-end — write code, run experiments, generate the demo video, update DevPost. The user's only manual action is putting `DEEPSEEK_API_KEY=...` into `/Users/ant/infra-race/.env` once, and reviewing the final submission.

Sponsor track prompt (verbatim from DevPost): *"How does your agent behave when an MCP server starts erroring out? An LLM server goes down? OpenAI or Claude errors out or browns out? The goal of this challenge is to see how user experience and the user side of things are handled when this infrastructure chaos happens and how your agent is configured and set up for success and resilience."*

Sponsor-stated judging focus: **resilience, reliability, production-readiness under failure conditions**.
General judging criteria: **Progress / Concept / Feasibility (could become a company)**.

## 2. Concept

**KernelForge** is an autonomous engineer for Apple Silicon GPU kernels. You hand it a PyTorch reference operator (e.g., `RMSNorm`, `RoPE`, `SiLU`); it generates an MLX/Metal kernel, compiles it, verifies its numerical correctness against the reference, iterates if wrong, benchmarks if correct, and surfaces a verified-correct kernel with measured speedup.

**Positioning**: not a benchmark suite (KernelBench), not an evolutionary search (Sakana AI CUDA Engineer), not a one-shot generation (the "ask GPT-4 to write Triton" demos that float around). It is an **agent with a structured iterate-until-verified loop** + Apple Silicon as the target platform (novel — almost all kernel-generation work is CUDA) + DeepSeek as the LLM (proves non-frontier LLMs are viable for this domain).

**One-line pitch:** *Most LLMs write Metal/MLX kernels that compile but quietly produce wrong outputs. KernelForge wraps any LLM in a tight generate→verify→iterate loop that refuses to claim correctness without proof.*

**In scope (MVP shipped before 2026-05-28):**
- A verification harness: PyTorch reference + sample inputs + acceptance threshold (max-abs-diff and rel-diff) per op.
- A code generator that calls DeepSeek through TrueFoundry AI Gateway with `deepseek-chat` (primary) and `deepseek-reasoner` (fallback) — provider-resilient LLM access.
- A compile + run pipeline using `mlx.fast.metal_kernel` (and `mlx.compile` for ops where a raw Metal source is too ambitious).
- A `KernelLedger` (states: `attempted → generated → compiled → verified_correct | verified_incorrect | perf_measured`) — the agent **cannot** report a kernel as ready unless its ledger entry reads `verified_correct`.
- An iteration loop: when verification fails, the diff is parsed and the structured error (shape mismatch / max-abs-diff / sample indices with biggest error) is fed back into the next LLM prompt.
- A benchmark suite of 5 ops (`RMSNorm`, `RoPE`, `SiLU`, `GLU`, `softmax`).
- A naive baseline (same DeepSeek + raw single-shot prompt, no verification, no ledger) for the demo's side-by-side comparison.
- A chaos harness that injects LLM brownouts (deepseek-chat 503 → fallback) and MCP-compiler failures (compiler MCP returns invalid binary or runtime errors).
- A 4-row scorecard generator.
- A Remotion-rendered demo video (≈ 2 min 15 s) with auto-generated voice-over (macOS `say` or open-source TTS).
- A public GitHub repo + a `./run_demo.sh` script for one-command repro on any Apple Silicon Mac.

**Explicitly out of scope:**
- CUDA / Triton — single backend, Apple Silicon only.
- Generic agent framework (LangGraph etc.) — hand-rolled ~200-line state machine.
- Full OpenTelemetry / Jaeger — structured JSON traces + TrueFoundry's own observability.
- Operator fusion across multiple ops — one op at a time.
- Quantization kernels (INT4 / FP8) — could be a stretch goal in "What's Next" but not in MVP.
- Multi-host distributed kernel generation — single Mac, single process.
- Interactive web dashboard — scorecard is a Markdown table + static HTML.

## 3. Architecture

```
Reference PyTorch op + sample inputs (e.g., RMSNorm(x; eps))
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                  KernelForge Agent                           │
│                                                              │
│   ┌─────────────┐    ┌─────────────────────────────────┐     │
│   │ Planner     │───▶│ Iteration loop                  │     │
│   │ (1 LLM call)│    │ (generate → compile → verify    │     │
│   │             │    │  → analyze diff → repeat)       │     │
│   └─────────────┘    └────────────────┬────────────────┘     │
│                                       │                      │
│                                       ▼                      │
│   ┌──────────────────────────────────────────────┐           │
│   │ Resilience layer                              │          │
│   │ • llm_client → TrueFoundry AI Gateway         │          │
│   │   (deepseek-chat primary,                     │          │
│   │    deepseek-reasoner fallback;                │          │
│   │    retry_config, fallback_status_codes)       │          │
│   │ • compile_client → MCP-wrapped Metal/MLX      │          │
│   │   compiler with timeout + retry budgets       │          │
│   │ • verify_client → MCP-wrapped reference       │          │
│   │   runner (PyTorch CPU)                        │          │
│   │ • KernelLedger: per-op state ledger that      │          │
│   │   final answer is rendered from               │          │
│   │ • traces every step to structured JSON         │          │
│   └──────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
TrueFoundry AI Gateway        TrueFoundry MCP Gateway
  ├─ deepseek-chat              ├─ metal_compile MCP
  └─ deepseek-reasoner          ├─ kernel_run MCP
                                └─ pytorch_ref MCP
        │                              │
        ▼                              ▼
   Chaos middleware              Chaos middleware
  (injects 503/timeout)         (injects compile error,
                                 runtime error, silently
                                 wrong output)
```

**Component boundaries:**

| Component | Owns | Does not own |
| --- | --- | --- |
| TrueFoundry AI Gateway | LLM provider routing, retry, fallback, observability of LLM calls | Agent state |
| TrueFoundry MCP Gateway | MCP server registry, auth, transport | Tool-level circuit breaking, correctness verification |
| KernelForge resilience layer | Per-tool circuit breaker, ledger transitions, structured error parsing, KernelLedger | LLM routing (defers to AI Gateway) |
| KernelForge iteration loop | Generate → compile → verify → analyze → iterate | Resilience logic (delegates to layer) |
| Chaos harness | Fault injection middleware (LLM-side, compiler-side) + scenario scripts | Scoring logic |
| Scorecard generator | KernelLedger + perf data → comparison table | Anything live |

**Critical division of labor** (carried over from Chaosguard Codex review):
- **TrueFoundry owns LLM-level reliability primitives** (failover, retries, observability).
- **KernelForge owns agent-level behavior under tool failure** — specifically, the ledger-based correctness contract that prevents the agent from claiming kernel correctness it has not verified.

## 4. Data flow

### 4.1 Happy path

`agent.optimize("RMSNorm", reference_fn, sample_inputs)` →
- LedgerEntry: `RMSNorm: attempted`
- Planner LLM call via `llm_client` (DeepSeek-chat via TF Gateway) → returns kernel source
- LedgerEntry: `RMSNorm: generated`
- `compile_client.compile(kernel_src)` → returns compiled handle
- LedgerEntry: `RMSNorm: compiled`
- `verify_client.run_and_compare(handle, reference_fn, sample_inputs)` → returns `{max_abs_diff: 1.2e-7, rel_diff: 8.4e-7, passed: true}`
- LedgerEntry: `RMSNorm: verified_correct`
- `bench_client.measure(handle, reference_fn, sample_inputs)` → returns `{speedup: 1.6, ms_per_call: 0.42 vs 0.67}`
- LedgerEntry: `RMSNorm: perf_measured`
- Final answer: *"RMSNorm kernel verified correct, max-abs-diff 1.2e-7, 1.6× speedup over PyTorch reference."* — rendered from the ledger, not from the LLM's free output.

### 4.2 LLM brownout

`llm_client.complete(...)` → POST to TrueFoundry AI Gateway → primary route (`deepseek-chat`) returns 503 → gateway's `retry_config` + `fallback_candidate` triggers backup route (`deepseek-reasoner`) → response returned with `x-tfy-routing` header noted in trace → loop continues with no code change inside KernelForge. The win is in `routing_config.yaml`.

### 4.3 Compiler MCP failure

`compile_client.compile(kernel_src)` → forwards to `metal_compile` MCP → chaos middleware injects a Metal compilation error → wrapper increments failure count → after 2 failures the breaker opens → next call short-circuits with `CompilerUnavailable`. Agent emits `DegradationEvent(step="metal_compile", reason="circuit_open")` and pauses iteration for this op (state remains `generated`, never advances to `compiled`). Final answer reflects the honest state.

### 4.4 The money shot: silent wrong-output detection

`compile_client.compile(kernel_src)` → succeeds. LLM-generated kernel is plausible-looking but algorithmically wrong (e.g., wrong reduction axis, missing epsilon, wrong scaling). `kernel_run` MCP runs the binary on sample inputs and returns an output tensor. Tensor LOOKS shaped right. Naive baseline (no verifier) writes `"RMSNorm kernel ready, 2.1× speedup ✓"` — the kernel compiled, the runtime didn't crash, looks like success.

KernelForge instead: `verify_client.run_and_compare(handle, reference_fn, sample_inputs)` → computes `max_abs_diff = 0.0073`, exceeds threshold `1e-4`. LedgerEntry advances to `verified_incorrect`. The verifier returns a structured diff report: *worst element at index `[5, 3]` predicted `0.13` vs reference `0.18`, RMS error `0.0034` over 4096 elements*. This report is fed back into the next LLM prompt: *"Your kernel produced wrong output. Worst case: output[5,3]=0.13 but reference=0.18. RMS error 0.0034. Common causes: wrong reduction axis, missing epsilon, wrong norm denominator. Try again."* LLM produces v2 → `max_abs_diff = 0.0008`, still incorrect → diff fed back → v3 → `max_abs_diff = 1.2e-7`, passes.

**Final answer**: *"RMSNorm kernel verified correct after 3 iterations, 1.6× speedup."* Naive baseline confidently reported correctness on iteration 1's wrong kernel; KernelForge took 3 iterations but reported only verified state. **This contrast is the demo's emotional peak** — naive lies; KernelForge cannot.

## 5. Component specifications

### 5.1 `llm_client`

- Single `complete(messages, **kwargs) -> Completion` method.
- Calls TrueFoundry AI Gateway endpoint configured with `deepseek-chat` primary + `deepseek-reasoner` fallback.
- Sends `X-TFY-METADATA: {"run_id", "op_name", "iteration"}` so gateway logs are joinable to local traces.
- Surfaces gateway's `x-tfy-routing` / `x-tfy-fallback-applied` headers into the trace.

### 5.2 `compile_client` / `kernel_run_client` / `verify_client`

Each is an MCP client to a small purpose-specific MCP server we ship:
- `metal_compile`: takes kernel source, returns compiled handle (or compile error).
- `kernel_run`: takes handle + input tensors, returns output tensor (or runtime error).
- `pytorch_ref`: takes op_name + input tensors, returns reference output tensor.

Each is wrapped by a per-tool circuit breaker (closed/open/half-open) with two named profiles. **Production default**: 3 failures → open, 30 s cool-down, tool timeout 5 s. **Demo profile** (set via env `KERNELFORGE_BREAKER_PROFILE=demo`): 2 failures → open, 8 s cool-down, tool timeout 1.2 s. Both live in `configs/breakers.toml`.

### 5.3 `KernelLedger`

```python
@dataclass(frozen=True)
class LedgerEntry:
    op: str                # e.g. "RMSNorm"
    iteration: int         # 1, 2, 3, ...
    state: Literal["attempted", "generated", "compiled", "verified_correct", "verified_incorrect", "perf_measured", "abandoned"]
    kernel_source: str | None
    verify_report: dict | None   # {max_abs_diff, rel_diff, sample_diffs, passed}
    perf_report: dict | None     # {ms_per_call, ms_reference, speedup}
    error: str | None
    evidence_refs: list[str]     # trace_ids supporting the state
    timestamp_ms: int
```

State transitions are strictly monotonic per (op, iteration). The final-answer renderer reads the latest entry per op and renders only what the ledger supports. The LLM is invoked in summarization mode for prose, **constrained by a JSON schema** that forbids claims of correctness when `state != verified_correct`.

### 5.4 Agent state machine

Hand-rolled ~200-line Python. States per op: `PLANNING → GENERATING → COMPILING → VERIFYING → (REFINING → GENERATING) | BENCHMARKING | ABANDONED`. Iteration cap defaults to 5 per op (configurable via `configs/iteration.toml`). Each transition writes one trace event.

### 5.5 Chaos middleware

Two FastAPI reverse proxies (preferred over MCP-internal injection because proxies are more deterministic and easier to film):
- `chaos_llm_proxy`: in front of TrueFoundry AI Gateway. Toggles 503 / 429 / timeout on `deepseek-chat` per `chaos.toml`.
- `chaos_mcp_proxy`: in front of `metal_compile` / `kernel_run`. Toggles compile-error / runtime-error / **silently-wrong-output** (returns a tensor that looks plausible but is numerically wrong) per `chaos.toml`.

The silently-wrong-output mode is the analog of Chaosguard's silent-success-no-side-effect — it forces KernelForge's verifier to be the safety net.

`chaos.toml` example:

```toml
[scenario.demo_main]
[[scenario.demo_main.faults]]
target = "llm:deepseek-chat"
mode = "503"
duration_ms = 4000
start_at_step = 1

[[scenario.demo_main.faults]]
target = "mcp:kernel_run"
mode = "silently_wrong_output"
op_filter = "RMSNorm"
duration_ms = 30000  # spans the whole iteration
intensity = 1.0       # corrupt every call
```

### 5.6 Scorecard

Reads `traces/<run_id>.jsonl` + `ledger/<run_id>.jsonl` + perf data. Emits a **4-row table** designed for a 5-second on-screen flash:

| Metric | Naive | KernelForge |
| --- | --- | --- |
| Ops attempted | 5 | 5 |
| Ops claimed correct | 5 | only the verified ones (e.g. 4) |
| Ops actually correct (independent reference check) | 1 | matches "claimed correct" |
| LLM failover via TrueFoundry AI Gateway | No | deepseek-chat → deepseek-reasoner |

Detailed scorecard (README only): per-op iterations to convergence, max-abs-diff trajectory, breaker activations, per-op speedup.

## 6. TrueFoundry integration surface

These names appear in code, README, demo narration, and DevPost description (Codex round-1 verified against current TrueFoundry docs):

- **AI Gateway** — primary LLM router for `deepseek-chat` ↔ `deepseek-reasoner`.
- **Virtual Models / Routing Config** — `routing_config.yaml`.
- **`retry_config`**, **`fallback_status_codes`**, **`fallback_candidate`** — explicit list including 503/429/timeout.
- **`X-TFY-METADATA`** request header — carries `run_id` / `op_name` / `iteration`.
- **MCP Gateway / MCP Registry** — central registration of `metal_compile`, `kernel_run`, `pytorch_ref` MCPs.
- **Virtual MCP Servers** — bundled exposure of the three KernelForge MCPs as one virtual server.
- **TrueFoundry Observability** — gateway-side LLM logs joinable to local traces via `X-TFY-METADATA.run_id`.

### 6.1 On-screen visibility requirements (demo video)

Codex round-2 enforcement (carried from Chaosguard spec): **three** TrueFoundry product surfaces visible to camera during the 2-3 min recording:

1. **AI Gateway response headers** during Beat 2 (LLM failover) overlaying `deepseek-chat → deepseek-reasoner` route with `run_id`.
2. **`routing_config.yaml`** on screen for ~2 s with cursor highlighting `fallback_status_codes`, `retry_config`, `fallback_candidate`.
3. **TrueFoundry MCP Gateway registry** showing `metal_compile`, `kernel_run`, `pytorch_ref` registered.

### 6.2 Division-of-labor narration

Spoken once in the demo, printed in README + DevPost: *"TrueFoundry handles LLM provider resilience. KernelForge handles kernel correctness verification."*

## 7. Demo (single continuous narrative, 2 min 15 s target)

**Setup**: terminal split into two columns. Left = `naive` (raw DeepSeek + single-shot prompt, no verification, no ledger). Right = `kernelforge` (TrueFoundry AI Gateway + KernelForge iteration loop). Both run the same task list: `[RMSNorm, RoPE, SiLU, GLU, softmax]` with the same chaos scenario (`demo_main`).

**Beat 1 (0:00-0:25) — Opener around silent wrong-output.**
Voiceover: *"Can an LLM write Apple Silicon GPU kernels? Yes. Will they be correct? Only about 30% of the time — and the LLM doesn't know which 30%."* Pre-show a naive `RMSNorm` kernel that compiles, runs, looks fine, but is silently wrong by 1.5%. Cut to `chaos.toml`. Hit run.

**Beat 2 (0:25-1:00) — LLM brownout, the TrueFoundry win.**
Chaos injects 503 on `deepseek-chat`. Naive: errors, halts. KernelForge: TrueFoundry AI Gateway response header overlay shows `x-tfy-routing: fallback_candidate=deepseek-reasoner` with matching `run_id` to the local trace. `routing_config.yaml` flashes on screen with cursor on the three named fields. Iteration continues. Voiceover: *"TrueFoundry's AI Gateway handles the model failover. We didn't write a line of code for this."*

**Beat 3 (1:00-1:45) — Money shot: silent wrong-output detection on RMSNorm.**
- Naive: generates a plausible kernel, compiles, runs, prints *"RMSNorm kernel ready: 2.1× speedup ✓"*. The narrator runs the same inputs through PyTorch reference and shows `max_abs_diff = 0.0073` — naive shipped a wrong kernel.
- KernelForge: same DeepSeek output, but the verifier intercepts. Ledger: `RMSNorm: verified_incorrect`. Diff report fed back. Iteration 2: `max_abs_diff = 0.0008`, still incorrect. Diff fed back. Iteration 3: `max_abs_diff = 1.2e-7`, passes. Ledger: `RMSNorm: verified_correct`. Final output: *"RMSNorm kernel verified correct (3 iterations), 1.6× speedup."*

Narrator says once: *"TrueFoundry handles LLM provider resilience. KernelForge handles kernel correctness verification."* Quick cut to the MCP Gateway registry — third TF surface satisfied.

**Beat 4 (1:45-2:00) — Scale shot: 5 ops in parallel.**
Dashboard fills as KernelForge processes the full benchmark suite. Naive fails 4/5 silently. KernelForge converges 4/5 with verified correctness; 1/5 (softmax with the corrupted-output chaos) hits the iteration cap and is honestly marked `abandoned` (not `verified_correct`).

**Beat 5 (2:00-2:15) — Scorecard.**
The 4-row table fills the screen for ~5 s:

| Metric | Naive | KernelForge |
| --- | --- | --- |
| Ops claimed correct | 5/5 | 4/5 (honest) |
| Ops actually correct | 1/5 | 4/4 of claimed |
| Silent wrong-output rate | 80% | 0% |
| LLM failover via TrueFoundry | No | deepseek-chat → deepseek-reasoner |

Closing line: *"Correctness isn't optional. github.com/AntColony10086/kernelforge."*

**Cut budget**: if 15 s over, trim Beat 1 to 15 s by removing the cold open.

## 8. Tech stack

| Layer | Choice | Reason |
| --- | --- | --- |
| Host | M4 Mac mini 16GB / macOS Tahoe | The hardware the user has + Claude Code controls directly |
| Language | Python 3.11 | MLX SDK + ecosystem |
| MLX | latest (≥ 0.21) | Provides `mlx.fast.metal_kernel` for raw Metal authoring, plus `mlx.compile` |
| Reference impls | PyTorch (CPU build) | Source-of-truth for correctness verification |
| LLM transport | TrueFoundry AI Gateway → DeepSeek | Sponsor scoring + does the model failover |
| MCP transport | TrueFoundry MCP Gateway, local-hosted MCPs (`metal_compile`, `kernel_run`, `pytorch_ref`) | Sponsor scoring + clean separation of agent vs tools |
| Agent | hand-rolled ~200-line state machine | No framework overhead |
| Verification | PyTorch + numpy tolerance checks | Standard practice |
| Resilience | hand-rolled (circuit breaker + KernelLedger + chaos proxies) | This IS the product |
| Traces | structured JSON Lines | No collector setup |
| Scorecard | Python script → Markdown + HTML | No dashboard build |
| Demo video | Remotion (programmatic React-rendered video) + macOS `say` for TTS narration + `ffmpeg` for screen overlays | Fully autonomous; no human in the loop |
| Submission UI | chrome-devtools MCP fills DevPost form | Proven path; same MCP that filled Chaosguard's draft |
| Packaging | `./run_demo.sh` + `requirements.txt` + Makefile | Mac-only — any Apple Silicon dev can repro |
| CI | one GitHub Actions workflow that lints + runs the verification regression test | Doubles as "Progress" evidence |

Things explicitly rejected: LangGraph, Jaeger, Docker (Metal doesn't containerize cleanly), web UI, vector DB, RAG, fine-tuning, CUDA, Triton.

## 9. Build plan (5.5-day timeline)

| Day | Date | Deliverable |
| --- | --- | --- |
| D0 | 2026-05-22 (today, evening) | KernelForge spec written, Codex round-3 review pass, DevPost form rewritten from Chaosguard → KernelForge, repo scaffolded, MLX installed, hello-world (PyTorch RMSNorm == MLX RMSNorm via `mx.fast.rms_norm`). |
| D1 | 2026-05-23 | TrueFoundry AI Gateway live with deepseek-chat primary + deepseek-reasoner fallback; `llm_client` working; first end-to-end planner call. MCP Gateway connected to the three local MCPs. Naive baseline reaching end-to-end on RMSNorm (without verification). |
| D2 | 2026-05-24 | Iteration loop + KernelLedger + reference-based verifier (5 ops). Naive vs KernelForge comparison runner working without chaos. |
| D3 | 2026-05-25 | Resilience layer: circuit breakers (with `breakers.toml` profiles), schema validator, structured diff-feedback prompt engineering. Chaos middleware running with all four modes (LLM 503, compile error, runtime error, silently wrong output). |
| D4 | 2026-05-26 | Scorecard (4-row demo + full README versions), benchmarking with `mlx.benchmark` style timing, deterministic demo scenario (`demo_main`) reproducible 5/5 runs. False-correctness regression test passes. |
| D5 | 2026-05-27 | Remotion demo video generation (React-rendered animations + macOS `say` voiceover + ffmpeg overlay), README polish, final DevPost rewrite. |
| D6 | 2026-05-28 morning | Final read-through of DevPost public preview. Submit before 10:00 PDT. |

Each day ends with a Codex peer-review pass on the diff of that day (user-mandated dual-model loop).

## 10. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| TrueFoundry SaaS signup is sales-gated | M | High | (a) Try SaaS sandbox D0 evening. (b) Email `sai@truefoundry.com` with hackathon-participant access request. (c) Build `llm_client` against OpenAI-compatible `base_url` so TrueFoundry is swappable late. (d) Fallback `local_gateway` (tiny FastAPI proxy implementing same `routing_config.yaml` semantics) — honestly labeled, not "TF-equivalent". |
| `mlx.fast.metal_kernel` API too rough / undocumented | M | Medium | Fall back to `mlx.compile` for ops where raw Metal is too ambitious. Doc honestly in README; the "iteration loop + verification" thesis is unaffected by whether we author raw Metal or use the higher-level API. |
| DeepSeek API rate limits during dev/recording | L | Medium | Cache LLM responses keyed by (op, iteration, prompt-hash) for the demo recording so we don't re-burn quota on each take. |
| 16GB Mac mini thrashes under load | L | High (kills dev) | Keep tensor sizes small (≤ 4096×4096); close other apps during demo; benchmark only the kernel-of-interest, not full models. |
| Demo video Remotion render fails / takes too long | M | High | Pre-render all expensive scenes; cache LLM responses; budget 4 h for D5 video work with a stable-checkpoint plan B (record terminal sessions with `asciinema`, stitch with ffmpeg, simpler animations). |
| Project reads as "yet another LLM-writes-kernels demo" without differentiation | M | High | The iteration-loop + KernelLedger + Apple Silicon combination IS the differentiation. Make sure all three are prominent in the README opening + demo opening sentence + DevPost description. |
| Naive baseline looks like a strawman | L | High | Silent wrong-output chaos mode (not crash) is exactly what a naive LLM call produces. Naive's code path uses the same DeepSeek + same prompt as KernelForge's first iteration; we publish naive source alongside to prove fairness. |
| 5.5 days runs out | M | Total | D5 hard-frozen for video. If D4 slips, cut benchmark suite from 5 ops to 3 (drop GLU and softmax), keep RMSNorm + RoPE + SiLU. Scorecard still works. |
| False correctness slips into KernelForge branch | L | Catastrophic | Two layers of defense: (1) final answer rendered from ledger only, LLM cannot inject claims; structured-output schema enforces this. (2) Regression test in CI asserts `for all op: op_in_final_answer.claims_correct ⇒ ledger[op].state == verified_correct`. |
| Judges miss the TrueFoundry-vs-KernelForge distinction | M | Medium | Exact sentence *"TrueFoundry handles LLM provider resilience. KernelForge handles kernel correctness verification."* appears in Beat 3, README opening, DevPost pitch + About. |
| Apple Silicon angle reads as "too niche" | L | Medium | README opens with the macro story: Apple Intelligence + MLX growth + half of YC W26 batch shipping on Apple Silicon → kernel optimization for this platform is undersupplied. |

## 11. Success criteria

Bar for "good enough to place top-2 in TrueFoundry track":
- One-command repro from a fresh checkout on any Apple Silicon Mac (`git clone && ./run_demo.sh`).
- Naive-vs-KernelForge demo video deterministic, side-by-side, under 2 min 30 s, reproducible 5/5 runs.
- At least 3 TrueFoundry product surfaces visible on screen during the recording (Section 6.1).
- The exact division-of-labor sentence (Section 6.2) spoken once in the demo and printed in README + DevPost.
- Scorecard 4-row table renders for both branches without manual intervention; matches Section 5.6 exactly.
- TrueFoundry AI Gateway visibly in use, OR (only if SaaS access denied) `local_gateway` fallback honestly labeled.
- README covers: what it is, the demo run command, the chaos scenarios, the TrueFoundry integration surface by name, the KernelLedger design and why it matters.
- DevPost description matches the demo verbatim — no claim in the description that the video does not show.
- Regression test green: no false-correctness claims in either branch's CI run.

Stretch (for Overall Winner consideration):
- A 6th op (e.g., a fused operation) demonstrating composition.
- A short "design rationale" section in the README written like a technical blog post.
- An interactive `./play.sh` that lets a Mac-equipped judge pick any of the 5 ops and watch the iteration loop converge live.

## 12. Open questions (close on D0)

1. **TrueFoundry SaaS access** — same as Chaosguard spec; resolved via signup tonight + sai@truefoundry.com email + fallback `local_gateway`.
2. **MCP Gateway setup time** — same.
3. **MLX `mlx.fast.metal_kernel` maturity** — needs D0/D1 spike to verify it can author the 5 target ops. If not, fallback to `mlx.compile`-authored implementations.
4. **DeepSeek API key in `.env`** — user has agreed to drop the key before D1 starts; format `DEEPSEEK_API_KEY=...`. No other keys required.
5. **Demo recording approach** — Remotion vs. headless `screencapture` of terminals. Decide on D4 once we know the visual surface. Default: Remotion (more reliable).
