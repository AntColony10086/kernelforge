"""Strict JSON-schema kernel output + prompt templates.

The LLM must return JSON that matches `KernelOutput`. Anything else is
treated as a compile failure.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KernelOutput(BaseModel):
    source: str = Field(..., description="Metal shading language kernel body (no triple-backticks, no signature).")
    grid: tuple[int, int, int] = Field(..., description="(gridX, gridY, gridZ) thread grid.")
    threadgroup: tuple[int, int, int] = Field(..., description="(tgX, tgY, tgZ) threadgroup size.")
    output_shapes: list[tuple[int, ...]] = Field(..., min_length=1)
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    assumptions: list[str] = Field(default_factory=list, description="Constraints the caller must satisfy.")


_SYSTEM_PROMPT = """You are KernelForge, a careful kernel-writing assistant for Apple Silicon (Metal Shading Language via MLX `mx.fast.metal_kernel`).

You write small, correct kernels. You return ONE JSON object matching the provided schema. No prose, no fences.

The kernel body you write is embedded inside an MLX-wrapped function where:
- thread_position_in_grid.x/y/z is the global thread id;
- inputs are addressable as `device const T* <name>` (T inferred from dtype);
- outputs are addressable as `device T* <name>`;
- input shape arrays are accessible as `<name>_shape[i]` (uint).

Hard rules:
1. Never claim correctness in prose. Just return the schema.
2. If you are uncertain about an assumption, list it in `assumptions[]` instead of silently assuming.
3. Use `metal::*` namespaces explicitly (e.g., `metal::precise::sqrt`, `metal::silu`).
4. For RoPE: use the split-half layout (first half = real, second half = imag). Do NOT use interleaved layout.
5. For RMSNorm: compute rms = sqrt(mean(x*x) + eps); be careful with eps placement.
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
