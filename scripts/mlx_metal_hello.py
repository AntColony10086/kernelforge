"""Spike: confirm mlx.core.fast.metal_kernel works on this machine.

Runs a trivial 'identity' Metal kernel and compares against the
PyTorch identity-equivalent. Prints diagnostic info on failure.
"""
import sys

import mlx.core as mx


SRC = """
uint tid = thread_position_in_grid.x;
if (tid < inp_shape[0]) {
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

    (out,) = kernel(
        inputs=[inp],
        grid=(n, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )
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
