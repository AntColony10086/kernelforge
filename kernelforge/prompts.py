"""Strict JSON-schema kernel output + prompt templates.

The LLM must return JSON that matches `KernelOutput`. Anything else is
treated as a compile failure.
"""
from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class KernelOutput(BaseModel):
    # Aliases tolerate the DeepSeek family's tendency to emit "kernel", "kernel_body",
    # "function_name" etc. instead of "source". We still prefer the canonical names in
    # the prompt; aliases are defense in depth.
    model_config = {"populate_by_name": True, "extra": "ignore"}

    source: str = Field(
        ...,
        description="Metal shading language kernel body (no triple-backticks, no signature).",
        validation_alias=AliasChoices("source", "kernel", "kernel_body", "kernel_source", "code", "metal_source"),
    )
    grid: tuple[int, int, int] = Field(
        ...,
        description="(gridX, gridY, gridZ) thread grid.",
        validation_alias=AliasChoices("grid", "grid_size", "thread_grid"),
    )
    threadgroup: tuple[int, int, int] = Field(
        ...,
        description="(tgX, tgY, tgZ) threadgroup size.",
        validation_alias=AliasChoices("threadgroup", "threadgroup_size", "thread_group", "block_size"),
    )
    output_shapes: list[tuple[int, ...]] = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("output_shapes", "output_shape", "out_shapes"),
    )
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        "float32",
        validation_alias=AliasChoices("dtype", "data_type", "output_dtype"),
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Constraints the caller must satisfy.",
        validation_alias=AliasChoices("assumptions", "notes", "caller_assumptions"),
    )


_SYSTEM_PROMPT = """You are KernelForge, a careful kernel-writing assistant for Apple Silicon (Metal Shading Language via MLX `mx.fast.metal_kernel`).

You write small, correct kernels. You MUST return exactly ONE JSON object using the field names below. No prose, no markdown fences, no other keys.

REQUIRED JSON SCHEMA — your response MUST be valid JSON shaped like this example (replace values, keep keys):

{
  "source": "uint tid = thread_position_in_grid.x; ...",
  "grid": [1024, 1, 1],
  "threadgroup": [64, 1, 1],
  "output_shapes": [[1024]],
  "dtype": "float32",
  "assumptions": ["head_dim is even"]
}

Field names are case-sensitive. Do NOT use 'kernel', 'kernel_body', 'function_name', or any alias — the parser ONLY accepts the exact keys above.

The kernel body you put in `source` is embedded inside an MLX-wrapped function where:
- thread_position_in_grid.x/y/z is the global thread id;
- inputs are addressable as `device const T* <name>` (T inferred from dtype);
- outputs are addressable as `device T* <name>`;
- ONLY INPUT shape arrays are auto-bound. For input `x` you can read `x_shape[0]`, `x_shape[1]`, ... (uint). There is NO `out_shape`. If you need output dimensions, derive them from the inputs (typically output shape equals input shape).
- The available headers are already included (e.g., `<metal_stdlib>`). Do NOT write `#include`, `using namespace`, or a function signature — just the kernel body statements.

Hard rules:
1. Never claim correctness in prose. Just return the schema object.
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
