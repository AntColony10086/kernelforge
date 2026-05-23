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
    total = len(outcomes) or 1

    # Canonical escalation order (cheap → expensive), NOT alphabetical:
    # deepseek-v4-flash is the happy-path route; deepseek-coder is the
    # escalation target. Render in the order the agent actually traverses.
    _CANONICAL_ROUTE_ORDER = [
        "deepseek-v4-flash",
        "deepseek-coder",
        "deepseek-v4-pro",  # historical fallback name
    ]
    all_routes = {r for o in outcomes for r in o.kf_llm_routes}
    ordered = [r for r in _CANONICAL_ROUTE_ORDER if r in all_routes] + sorted(all_routes - set(_CANONICAL_ROUTE_ORDER))
    routes_str = " -> ".join(ordered) if len(ordered) > 1 else (ordered[0] if ordered else "n/a")

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
    lines = [
        "## Detailed scorecard",
        "",
        "| Op | KF claim | KF iters | LLM route | Speedup vs MLX eager | Speedup vs mx.fast |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for o in outcomes:
        sp_eager = o.kf_speedups.get("mx_eager", "n/a")
        sp_fast = next((v for k, v in o.kf_speedups.items() if k.startswith("mx_fast")), "n/a")
        claim = "verified" if o.kf_claimed_correct else "—"
        routes = " -> ".join(o.kf_llm_routes) if o.kf_llm_routes else "n/a"
        sp_eager_s = f"{sp_eager:.2f}x" if isinstance(sp_eager, (int, float)) else sp_eager
        sp_fast_s = f"{sp_fast:.2f}x" if isinstance(sp_fast, (int, float)) else sp_fast
        lines.append(f"| {o.op} | {claim} | {o.kf_iterations} | {routes} | {sp_eager_s} | {sp_fast_s} |")
    return "\n".join(lines)
