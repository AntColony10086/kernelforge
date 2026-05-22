from kernel_lab.compile_tool import clear_cache, compile_kernel


IDENTITY_SRC = """
uint tid = thread_position_in_grid.x;
if (tid < inp_shape[0]) {
    out[tid] = inp[tid];
}
"""


def test_compile_identity_returns_handle():
    clear_cache()
    handle = compile_kernel(
        name="identity_test_1",
        source=IDENTITY_SRC,
        grid=(64, 1, 1),
        threadgroup=(64, 1, 1),
        input_names=["inp"],
        output_names=["out"],
    )
    assert isinstance(handle, str)
    assert len(handle) == 64  # sha256 hex


def test_compile_cache_hits_same_handle():
    clear_cache()
    h1 = compile_kernel(
        name="identity_test_2",
        source=IDENTITY_SRC,
        grid=(64, 1, 1),
        threadgroup=(64, 1, 1),
        input_names=["inp"],
        output_names=["out"],
    )
    h2 = compile_kernel(
        name="identity_test_2",
        source=IDENTITY_SRC,
        grid=(64, 1, 1),
        threadgroup=(64, 1, 1),
        input_names=["inp"],
        output_names=["out"],
    )
    assert h1 == h2
