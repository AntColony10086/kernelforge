"""Generate the demo voiceover using macOS `say` and convert AIFF -> WAV via ffmpeg.

Voiceover script is split into beats aligned with the Remotion timeline.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Script roughly aligns to the 5-beat Remotion timeline (~2:15).
SCRIPT = (
    "KernelForge. "
    "The worst kernel failure is not a crash. It is an L L M written kernel that compiles, "
    "passes a smoke test, and silently produces wrong outputs on the next shape. "
    "When the cheap deepseek v 4 flash fails or rate limits, True Foundry's A I Gateway "
    "escalates to deepseek v 4 pro. The routing rule lives in routing config dot yaml; "
    "we did not write code for this. "
    "On Rotary Position Embedding, the L L M's first attempt compiles and passes a small "
    "smoke test, but our hidden holdout catches a split half versus interleaved layout bug "
    "on shape two by thirty two by one twenty eight. The agent escalates to v 4 pro, "
    "regenerates, and verifies five out of five holdouts. "
    "True Foundry handles L L M provider resilience. KernelForge handles kernel correctness "
    "verification. "
    "Across three operators, Naive ships kernels with a sixty seven percent silent wrong "
    "output rate. KernelForge ships zero. On Rotary Position Embedding we beat M L X eager "
    "by one point two times and lose to m x dot fast dot rope by twenty percent. We disclose "
    "the loss honestly. "
    "Correctness is not a vibe. It is a holdout suite. github dot com slash Ant Colony 10086 "
    "slash kernel forge."
)


def main() -> int:
    public = Path("demo/remotion/public")
    public.mkdir(parents=True, exist_ok=True)
    aiff = public / "voiceover.aiff"
    wav = public / "voiceover.wav"

    if not shutil.which("say"):
        print("ERROR: macOS `say` not available — skip voiceover", file=sys.stderr)
        return 1
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not available — install with `brew install ffmpeg`", file=sys.stderr)
        return 1

    subprocess.run(["say", "-v", "Samantha", "-r", "175", "-o", str(aiff), SCRIPT], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff), "-ar", "48000", str(wav)],
        check=True,
        capture_output=True,
    )
    aiff.unlink(missing_ok=True)
    print(f"voiceover written to {wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
