#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "ERROR: missing .env — copy .env.example and add DEEPSEEK_API_KEY."
  exit 1
fi

# shellcheck disable=SC2046
export $(grep -v '^#' .env | xargs)

if [ -z "${TFY_GATEWAY_BASE_URL:-}" ]; then
  echo "==> TrueFoundry SaaS not configured. Starting local_gateway fallback on :8765..."
  uv run uvicorn local_gateway.server:app --port 8765 --log-level warning &
  GATEWAY_PID=$!
  # Make sure we tear it down on exit.
  trap 'kill $GATEWAY_PID 2>/dev/null || true' EXIT
  sleep 2
fi

echo "==> Recording demo scenarios (naive + KernelForge over 3 ops)..."
uv run python -m demo.record

if [ -d demo/remotion ] && [ -f demo/remotion/package.json ]; then
  echo "==> Generating voiceover..."
  uv run python demo/voiceover.py || echo "WARN: voiceover failed (continuing without audio)"

  echo "==> Rendering Remotion video..."
  (cd demo/remotion && bun install --silent && bun run remotion render KernelForgeDemo out/demo.mp4) || \
    echo "WARN: Remotion render failed — see logs"

  if [ -f demo/remotion/out/demo.mp4 ]; then
    echo "==> Demo video: demo/remotion/out/demo.mp4"
  fi
fi

echo
echo "==> Scorecard:"
cat demo/artifacts/scorecard_demo.md
echo
echo "==> Detailed scorecard:"
cat demo/artifacts/scorecard_readme.md
