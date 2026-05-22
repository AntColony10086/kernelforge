import json
from pathlib import Path

from scorecard.generate import compute_outcomes
from scorecard.render import render_demo_scorecard, render_readme_scorecard


def test_demo_scorecard_4_rows(tmp_path: Path):
    naive = tmp_path / "naive.jsonl"
    kf = tmp_path / "kf.jsonl"
    naive.write_text(
        json.dumps({"op": "rope", "state": "verified_correct", "claimed_correct": True, "iteration": 1, "llm_route": "deepseek-v4-flash"})
        + "\n"
    )
    kf.write_text(
        json.dumps(
            {
                "op": "rope",
                "state": "verified_correct",
                "iteration": 2,
                "llm_route": "deepseek-v4-pro",
                "perf_report": {"speedups": {"mx_eager": 1.1, "mx_fast_rope": 0.85}},
            }
        )
        + "\n"
    )
    out = compute_outcomes(naive, kf, ground_truth={"rope": False})  # naive's claim is a lie
    md = render_demo_scorecard(out)
    lines = [line for line in md.split("\n") if line.startswith("|")]
    assert len(lines) == 6  # header + separator + 4 metric rows


def test_readme_scorecard_contains_per_op_speedups(tmp_path: Path):
    naive = tmp_path / "naive.jsonl"
    kf = tmp_path / "kf.jsonl"
    naive.write_text(json.dumps({"op": "rmsnorm", "claimed_correct": True, "iteration": 1, "llm_route": "deepseek-v4-flash"}) + "\n")
    kf.write_text(
        json.dumps(
            {
                "op": "rmsnorm",
                "state": "perf_measured",
                "iteration": 1,
                "llm_route": "deepseek-v4-flash",
                "perf_report": {"speedups": {"mx_eager": 1.5, "mx_fast_rms_norm": 0.75}},
            }
        )
        + "\n"
    )
    out = compute_outcomes(naive, kf, ground_truth={"rmsnorm": True})
    md = render_readme_scorecard(out)
    assert "1.50x" in md
    assert "0.75x" in md
    assert "rmsnorm" in md
