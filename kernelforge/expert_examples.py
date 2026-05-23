"""Few-shot expert examples for the KernelForge LLM prompts.

For each broad op category, we provide ONE good example and ONE bad
example so the LLM sees both the target style and a common failure
pattern. The LLM is told NOT to copy the source verbatim, but to use
it for API conventions (`x_shape[i]`, `metal::precise::*`, etc.) and
common pitfalls.

Example library is intentionally small (3 pairs total). DeepSeek
generalises style well — more examples means longer prompts without
proportional accuracy gain.
"""
from __future__ import annotations

from typing import Literal

OpCategory = Literal["activation", "reduction", "geometric", "elementwise", "linalg"]


# -- Activation (single-input element-wise) ----------------------------------

_GOOD_SILU = '''Good example — SiLU activation (single input, element-wise):

{
  "source": "uint tid = thread_position_in_grid.x;\\nuint total = x_shape[0] * x_shape[1] * x_shape[2];\\nif (tid < total) {\\n    float v = float(x[tid]);\\n    float s = 1.0f / (1.0f + metal::precise::exp(-v));\\n    out[tid] = (decltype(out[tid]))(v * s);\\n}\\n",
  "grid": [262144, 1, 1],
  "threadgroup": [256, 1, 1],
  "output_shapes": [[B, S, D]],
  "dtype": "float32",
  "assumptions": ["grid covers total elements; bounds check inside kernel"]
}

Why this is good:
- Always promotes to fp32 BEFORE calling exp() — avoids fp16 sigmoid saturation.
- Bounds-checks `tid < total` instead of trusting the grid exactly.
- Uses `metal::precise::*` for math precision.
- Casts the result back to the output dtype at the end.
'''

_BAD_SILU = '''Bad example — SiLU written in fp16, no promotion:

{
  "source": "uint tid = thread_position_in_grid.x;\\nhalf v = x[tid];           // fp16 input\\nhalf s = 1.0h / (1.0h + metal::exp(-v));   // BUG: should promote to fp32 before exp\\nout[tid] = v * s;\\n",
  ...
}

Why this is wrong:
- For fp16 inputs in the saturated range (|x| > 5), `metal::exp(-v)` underflows to 0 in fp16, sigmoid becomes 1.0 exactly, output becomes v — silent NaN-free corruption.
- No bounds check — if grid > total elements, reads past array end.
'''


# -- Reduction (over last dim) -----------------------------------------------

_GOOD_RMSNORM = '''Good example — RMSNorm (reduction over last dim, then per-row normalisation):

{
  "source": "uint row = thread_position_in_grid.x;\\nuint num_rows = x_shape[0] * x_shape[1];\\nif (row >= num_rows) return;\\nuint D = x_shape[2];\\nuint base = row * D;\\nfloat sum_sq = 0.0f;\\nfor (uint i = 0; i < D; i++) {\\n    float v = float(x[base + i]);\\n    sum_sq += v * v;\\n}\\nfloat rms = metal::precise::sqrt(sum_sq / float(D) + 1e-6f);\\nfor (uint i = 0; i < D; i++) {\\n    out[base + i] = (decltype(out[base + i]))((float(x[base + i]) / rms) * float(weight[i]));\\n}\\n",
  "grid": [B_times_S, 1, 1],
  "threadgroup": [1, 1, 1],
  "output_shapes": [[B, S, D]],
  "dtype": "float32",
  "assumptions": ["one thread per row, simple correctness-first impl"]
}

Why this is good:
- Reduces in fp32 even when input is fp16 — prevents reduction-precision loss.
- `eps` is INSIDE the sqrt (`sum_sq / D + eps`), not outside — prevents division underflow.
- Correctness-first; can be optimised later with threadgroup reductions.
'''

_BAD_RMSNORM = '''Bad example — RMSNorm with eps outside sqrt + missing fp32 accumulation:

{
  "source": "uint row = thread_position_in_grid.x;\\nuint D = x_shape[2];\\nhalf sum_sq = 0.0h;                       // BUG: fp16 reduction loses precision\\nfor (uint i = 0; i < D; i++) {\\n    half v = x[row * D + i];\\n    sum_sq += v * v;\\n}\\nhalf rms = metal::sqrt(sum_sq / half(D)) + 1e-6h;   // BUG: eps OUTSIDE sqrt, division can underflow first\\nfor (uint i = 0; i < D; i++) {\\n    out[row * D + i] = (x[row * D + i] / rms) * weight[i];\\n}\\n",
  ...
}

Why this is wrong:
- fp16 sum_sq overflows for D > ~256 or magnitudes > ~1.
- eps OUTSIDE the sqrt means `sum_sq / D` can be 0 before eps is added, division by 0.
- No bounds check on `row`.
'''


# -- Geometric (RoPE) — THE money-shot category ------------------------------

_GOOD_ROPE = '''Good example — RoPE with SPLIT-HALF layout (Llama / DeepSeek convention):

{
  "source": "uint tid = thread_position_in_grid.x;\\nuint B = x_shape[0]; uint S = x_shape[1]; uint D = x_shape[2];\\nuint half_D = D / 2;\\nuint total = B * S * half_D;\\nif (tid >= total) return;\\nuint b = tid / (S * half_D);\\nuint s = (tid / half_D) % S;\\nuint d = tid % half_D;\\nfloat inv_freq = metal::precise::pow(10000.0f, -float(2 * d) / float(D));\\nfloat theta = float(s) * inv_freq;\\nfloat c = metal::precise::cos(theta);\\nfloat si = metal::precise::sin(theta);\\nuint row_base = b * S * D + s * D;\\nfloat xr = float(x[row_base + d]);\\nfloat xi = float(x[row_base + half_D + d]);\\nout[row_base + d]          = (decltype(out[0]))(xr * c - xi * si);\\nout[row_base + half_D + d] = (decltype(out[0]))(xr * si + xi * c);\\n",
  "grid": [B_times_S_times_halfD, 1, 1],
  "threadgroup": [128, 1, 1],
  "output_shapes": [[B, S, D]],
  "dtype": "float32",
  "assumptions": ["split-half layout: first half is real, second half is imag (Llama convention)"]
}

Why this is good:
- SPLIT-HALF layout: indexes `[d]` and `[half_D + d]` — matches PyTorch reference exactly.
- Per-thread one (real, imag) pair, parallel across batch, seq, half_D.
- All angle math in fp32 to preserve precision.
'''

_BAD_ROPE = '''Bad example — RoPE with INTERLEAVED layout (common LLM mistake):

{
  "source": "uint tid = thread_position_in_grid.x;\\nuint pair_idx = tid;\\nuint base = pair_idx * 2;     // BUG: indexes (x0, x1), (x2, x3), ... — interleaved, not split-half\\nfloat xr = float(x[base]);\\nfloat xi = float(x[base + 1]);\\nout[base]     = xr * c - xi * si;\\nout[base + 1] = xr * si + xi * c;\\n",
  ...
}

Why this is wrong:
- The PyTorch / Llama / DeepSeek convention is SPLIT-HALF (first half is x_real, second half is x_imag), NOT interleaved (x0, x1 adjacent).
- Interleaved layout COINCIDENTALLY matches the small smoke-test shape `[1, 1, 64]` by accident but produces 100% wrong output on `[2, 32, 128]`.
- This is the exact "smoke passes, holdout fails" failure mode the hidden holdout suite is built to catch.
'''


# -- Category mapping --------------------------------------------------------

_OP_TO_CATEGORY: dict[str, OpCategory] = {
    "rope": "geometric",
    "rmsnorm": "reduction",
    "layernorm": "reduction",
    "swiglu": "activation",
    "softmax": "reduction",
    "gelu": "activation",
    "silu": "activation",
    "tanh": "activation",
    "relu": "activation",
    "sigmoid": "activation",
    "exp": "activation",
    "log": "activation",
    "sqrt": "activation",
    "abs": "activation",
    "sum_last": "reduction",
    "max_last": "reduction",
    "mean_last": "reduction",
    "elementwise_add": "elementwise",
    "elementwise_mul": "elementwise",
    "matmul": "linalg",
}


_EXAMPLES_BY_CATEGORY: dict[OpCategory, tuple[str, str]] = {
    "activation": (_GOOD_SILU, _BAD_SILU),
    "elementwise": (_GOOD_SILU, _BAD_SILU),  # close enough to activation pattern
    "reduction": (_GOOD_RMSNORM, _BAD_RMSNORM),
    "geometric": (_GOOD_ROPE, _BAD_ROPE),
    "linalg": (_GOOD_RMSNORM, _BAD_RMSNORM),  # matmul uses reduction patterns
}


def examples_for(op: str) -> tuple[str, str]:
    """Return (good_example, bad_example) strings for the op's category."""
    category = _OP_TO_CATEGORY.get(op, "activation")
    return _EXAMPLES_BY_CATEGORY[category]
