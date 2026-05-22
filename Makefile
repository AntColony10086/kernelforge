.PHONY: install test lint format demo clean

install:
	uv pip install -e ".[dev]"

test:
	uv run pytest -q

lint:
	uv run ruff check kernelforge kernel_lab references chaos baselines scorecard local_gateway tests

format:
	uv run ruff format kernelforge kernel_lab references chaos baselines scorecard local_gateway tests

demo:
	uv run python -m demo.record
	cd demo/remotion && bun run build

clean:
	rm -rf .venv .pytest_cache __pycache__ */__pycache__ kernelforge.egg-info
