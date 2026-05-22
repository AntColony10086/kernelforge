# KernelForge

**Verified MLX/Metal kernel generation for Apple Silicon.**

KernelForge wraps DeepSeek in a hidden-holdout verification harness that refuses to claim kernel correctness without proof, and routes cheap → expensive LLMs via TrueFoundry AI Gateway as failures escalate.

> Built for the DevNetwork [AI + ML] Hackathon 2026, TrueFoundry Resilient Agents track.

## What it does

Given a PyTorch reference operator (`RoPE`, `RMSNorm`, `SwiGLU`), KernelForge generates an MLX/Metal kernel, runs it against a **hidden holdout suite** (10+ cases per op varying shape, stride, dtype, eps, edge magnitudes), iterates with structured diff feedback until verified or capped, and reports honest perf vs MLX eager / `mx.compile` / `mx.fast` built-ins.

## Why this is different

- **Apple Silicon target.** Nearly all LLM-kernel-generation work is CUDA. MLX/Metal is undersupplied.
- **Hidden holdout verification.** The LLM never sees the holdout inputs; it only gets structured diffs on failure, so it cannot overfit.
- **Cost-aware LLM routing.** Cheap `deepseek-v4-flash` on the happy path; escalate to `deepseek-v4-pro` only after a real failure.
- **`KernelLedger` correctness contract.** The final answer is rendered from the ledger, not from the LLM. The agent cannot claim correctness outside a `verified_correct` ledger state.

> **TrueFoundry handles LLM provider resilience. KernelForge handles kernel correctness verification.**

## Quickstart

Requires Apple Silicon Mac, Python 3.11+, Node 20+, Bun, ffmpeg.

```bash
git clone https://github.com/AntColony10086/kernelforge.git
cd kernelforge
cp .env.example .env  # then put your DEEPSEEK_API_KEY in
./run_demo.sh
```

This runs the naive baseline and KernelForge over RoPE, RMSNorm, SwiGLU under the deterministic `demo_main` chaos scenario, then renders the demo video.

## Architecture

See `docs/superpowers/specs/2026-05-22-kernelforge-design.md` for the full design — architecture, holdouts, ledger states, demo plan, and risk mitigations.

```
PyTorch reference op + hidden holdout suite
        │
        ▼
KernelForge Agent
  ├─ Planner (1 LLM call via TrueFoundry AI Gateway)
  ├─ Iteration loop (generate → compile → smoke → holdout verify → refine)
  ├─ Resilience layer (circuit breakers, schema validator, KernelLedger)
  └─ kernel_lab MCP (compile / run / verify / bench)
```

## TrueFoundry surface

- **AI Gateway** routes `deepseek-v4-flash` ↔ `deepseek-v4-pro` per `configs/routing_config.yaml`.
- **MCP Gateway** registers `kernel_lab` (compile/run/verify/bench).
- **`X-TFY-METADATA`** carries `run_id` / `op` / `iteration` / `escalate=pro`.
- If TrueFoundry SaaS access is not available, falls back to the honestly-labeled `local_gateway` proxy at `127.0.0.1:8765`.

## Repo layout

```
kernelforge/         agent state machine, ledger, holdouts, llm_client, prompts, final_answer
kernel_lab/          MCP server + compile/run/verify/bench tools
references/          PyTorch reference impls for the 3 target ops
chaos/               deterministic fault injection middleware
baselines/           naive smoke-only baseline (the strawman)
scorecard/           4-row demo scorecard + detailed README scorecard
local_gateway/       TrueFoundry AI Gateway fallback (honestly labeled)
configs/             routing/breakers/iteration/chaos
demo/                scripted recorder + Remotion video project + artifacts
tests/               pytest suite (34+ tests, false-correctness regression guard)
```

## Hidden holdout suite (the differentiation)

Each op has ~10 holdout cases that vary shape, stride, dtype, eps, and edge magnitudes. The kernel only graduates to `verified_correct` when EVERY holdout passes within tolerance `(max_abs_diff <= 1e-4, max_rel_diff <= 1e-3)`. Example RoPE holdouts:

- `shape=[1,8,64]` float32 — basic smoke.
- `shape=[2,32,128]` float32 — **catches split-half vs interleaved layout bugs** (the money-shot case).
- `shape=[4,16,256]` float16 — dtype precision.
- `shape=[1,256,64]` — large position-id, catches frequency-schedule bugs.
- `shape=[1,8,64] base=500000.0` — non-default base frequency.

The LLM **never sees these inputs**. After a failure, only `(case_name, max_abs_diff, suspected_cause_hints)` is fed back into the next prompt.

## Scorecard (demo run)

See `demo/artifacts/scorecard_demo.md` and `demo/artifacts/scorecard_readme.md` (populated after `./run_demo.sh`).

## Why not LangGraph / Sakana / etc.

- **LangGraph** would replace the ~200 LOC hand-rolled state machine with a heavier framework. Net: more deps, less observable, slower to debug. Skipped.
- **Sakana AI CUDA Engineer** runs an evolutionary loop over CUDA kernels. We use structured diff feedback + LLM escalation instead (cheaper, faster, Mac-native).
- **KernelBench (Stanford)** is a benchmark, not an agent. Useful for evaluation; we built the agent.

## License

MIT (see `LICENSE`).
