"""kernel_lab MCP server: exposes compile/run/verify/bench as MCP tools.

NOTE: this is a thin shim. The in-process Python API
(`kernel_lab.compile_tool.compile_kernel`, etc.) is what KernelForge
calls directly to avoid tensor-marshaling overhead across the MCP boundary.

The MCP server exists for sponsor-track signal (TrueFoundry MCP Gateway
registers `kernel_lab` as the only MCP server) and for the demo
recording (the MCP Gateway registry view shows it).
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from kernel_lab.compile_tool import compile_kernel

mcp = FastMCP("kernel_lab")


@mcp.tool()
def compile(
    name: str,
    source: str,
    grid: tuple,
    threadgroup: tuple,
    input_names: list[str],
    output_names: list[str],
) -> dict:
    """Compile a Metal kernel from source. Returns a cached handle."""
    handle = compile_kernel(
        name=name,
        source=source,
        grid=grid,
        threadgroup=threadgroup,
        input_names=input_names,
        output_names=output_names,
    )
    return {"handle": handle}


@mcp.tool()
def run(handle: str, inputs: list, grid: tuple, threadgroup: tuple, output_shapes: list, output_dtype: str = "float32") -> dict:
    """Run a compiled kernel. NOTE: tensor marshaling across MCP is not
    implemented for the demo; KernelForge calls the in-process Python API
    directly."""
    return {"note": "use the in-process kernel_lab.run_tool.run_kernel API"}


@mcp.tool()
def verify(handle: str, op: str, grid: tuple, threadgroup: tuple) -> dict:
    """Verify a kernel against the hidden holdout suite for `op`. NOTE:
    in-process call only; see kernel_lab.verify_tool.verify_kernel."""
    return {"note": "use the in-process kernel_lab.verify_tool.verify_kernel API"}


@mcp.tool()
def bench(handle: str, op: str, grid: tuple, threadgroup: tuple) -> dict:
    """Benchmark a kernel against MLX baselines. NOTE: in-process call
    only; see kernel_lab.bench_tool.bench_kernel."""
    return {"note": "use the in-process kernel_lab.bench_tool.bench_kernel API"}


if __name__ == "__main__":
    mcp.run()
