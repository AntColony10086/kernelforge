from kernelforge.holdouts import HOLDOUTS, cases_for


def test_holdouts_cover_core_ops():
    core = {"rope", "rmsnorm", "swiglu"}
    assert core.issubset(set(HOLDOUTS.keys()))


def test_holdouts_at_least_four_cases_per_op():
    for op in HOLDOUTS:
        assert len(cases_for(op)) >= 4, f"{op} has fewer than 4 holdouts"


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
            if op == "rope":
                out = c.reference_fn(inputs["x"], base=inputs["base"])
            elif op == "rmsnorm":
                out = c.reference_fn(inputs["x"], inputs["weight"], inputs["eps"])
            elif op == "swiglu":
                out = c.reference_fn(inputs["gate"], inputs["up"])
            assert out is not None
