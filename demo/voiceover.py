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
    "We built a verification-gated kernel agent for Apple Silicon. "
    "When deepseek v 4 flash returns a kernel that does not survive verification, "
    "True Foundry's gateway routes the next call to deepseek coder. The escalation lives "
    "in routing config dot yaml. "
    "On a twenty operator benchmark suite, the naive baseline declared seventeen kernels "
    "correct from compile success alone. KernelForge ran every kernel against a hidden "
    "holdout suite and verified zero. The contract held: zero false correctness claims, "
    "while naive shipped unverified code. "
    "True Foundry handles L L M provider resilience. KernelForge handles kernel correctness "
    "verification. "
    "Correctness is not a vibe. It is a holdout suite that the language model never sees. "
    "github dot com slash Ant Colony 10086 slash kernel forge."
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
