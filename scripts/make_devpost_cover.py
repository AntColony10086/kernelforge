"""Generate a 3:2 cover image for DevPost's Project Media gallery.

Output: ~/Desktop/kernelforge_devpost_cover.png (1500x1000, <5MB, 3:2)

Style: dark navy background, electric-green + amber accents, monospaced
typography, split-screen contrast between Naive and KernelForge.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# -- Dimensions / palette ----------------------------------------------------

W, H = 1500, 1000  # 3:2
PAD = 60
BG = (11, 11, 16)            # near-black navy
PANEL_BG = (20, 22, 30)
DIV = (50, 55, 70)
FG = (234, 234, 240)
DIM = (160, 160, 175)
RED = (248, 107, 107)
GREEN = (82, 210, 115)
BLUE = (108, 184, 255)
AMBER = (255, 180, 90)

MONO_PATHS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/Library/Fonts/JetBrainsMono-Regular.ttf",
]
SANS_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _font(paths: list[str], size: int) -> ImageFont.FreeTypeFont:
    for p in paths:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text(draw: ImageDraw.ImageDraw, xy, text, font, fill, *, anchor="la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _measure(font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int]:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _hline(draw, y, x0, x1, color, width=1):
    draw.line([(x0, y), (x1, y)], fill=color, width=width)


def _rounded(draw, xy, radius, **kwargs):
    draw.rounded_rectangle(xy, radius=radius, **kwargs)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    f_title = _font(MONO_PATHS, 64)
    f_subtitle = _font(SANS_PATHS, 26)
    f_section = _font(SANS_PATHS, 30)
    f_metric_big = _font(MONO_PATHS, 84)
    f_metric_sub = _font(SANS_PATHS, 18)
    f_label = _font(MONO_PATHS, 19)
    f_chip = _font(MONO_PATHS, 15)
    f_foot = _font(MONO_PATHS, 18)

    # ---- Header ------------------------------------------------------------
    _text(draw, (PAD, PAD), "KernelForge", f_title, FG)
    _text(draw, (PAD, PAD + 78), "Verified MLX/Metal Kernel Agent — refuses to ship what it cannot verify",
          f_subtitle, DIM)

    # Hackathon chip top-right
    chip_text = "DevNetwork [AI+ML] 2026 · TrueFoundry Resilient Agents"
    cw, ch = _measure(f_chip, chip_text)
    chip_x1 = W - PAD
    chip_x0 = chip_x1 - cw - 24
    chip_y0 = PAD + 6
    chip_y1 = chip_y0 + ch + 14
    _rounded(draw, (chip_x0, chip_y0, chip_x1, chip_y1), radius=10,
             fill=PANEL_BG, outline=BLUE, width=1)
    _text(draw, ((chip_x0 + chip_x1) // 2, (chip_y0 + chip_y1) // 2),
          chip_text, f_chip, BLUE, anchor="mm")

    # ---- Split panels ------------------------------------------------------
    panel_top = 220
    panel_bottom = H - 130
    gutter = 24
    left_x0, left_x1 = PAD, W // 2 - gutter // 2
    right_x0, right_x1 = W // 2 + gutter // 2, W - PAD

    def panel(x0, x1, accent, header, claim, claim_sub, second_line, second_color):
        _rounded(draw, (x0, panel_top, x1, panel_bottom), radius=18,
                 fill=PANEL_BG, outline=accent, width=2)
        # Header bar
        _hline(draw, panel_top + 60, x0 + 24, x1 - 24, accent, width=1)
        _text(draw, ((x0 + x1) // 2, panel_top + 28),
              header, f_section, accent, anchor="mm")
        # Big metric
        _text(draw, ((x0 + x1) // 2, panel_top + 170),
              claim, f_metric_big, FG, anchor="mm")
        _text(draw, ((x0 + x1) // 2, panel_top + 230),
              claim_sub, f_metric_sub, DIM, anchor="mm")
        # Second line
        _text(draw, ((x0 + x1) // 2, panel_top + 320),
              second_line, f_section, second_color, anchor="mm")

    panel(left_x0, left_x1, RED, "Naive baseline",
          "17/20", "kernels claimed correct (compile only)",
          "smoke ≠ verification", AMBER)
    panel(right_x0, right_x1, GREEN, "KernelForge",
          "0", "false-success claims (the contract)",
          "20 ops · ~80 hidden holdout cases", GREEN)

    # Sub-row inside each panel: ops detail
    inner_top = panel_top + 380
    # Left panel detail
    _text(draw, (left_x0 + 36, inner_top),
          "→ 1 smoke test\n→ trusts compile success\n→ no holdout suite", f_label, DIM)
    # Right panel detail
    _text(draw, (right_x0 + 36, inner_top),
          "→ generate → compile → smoke\n→ run hidden holdout suite\n→ KernelLedger refuses lie\n→ Flash → coder on failure", f_label, DIM)

    # ---- vs divider --------------------------------------------------------
    vs_y = (panel_top + panel_bottom) // 2
    _text(draw, (W // 2, vs_y), "vs", f_section, DIM, anchor="mm")

    # ---- Bottom strip ------------------------------------------------------
    foot_y = H - 60
    foot_chips = [
        ("20 ops", GREEN),
        ("~80 hidden holdout cases", GREEN),
        ("0 false-success claims", GREEN),
        ("Apple Silicon · MLX/Metal", BLUE),
        ("TrueFoundry AI Gateway", AMBER),
    ]
    # Calculate total width
    chip_pad_x = 14
    chip_pad_y = 8
    chip_gap = 14
    chip_sizes = [(c, _measure(f_foot, c)) for c, _ in foot_chips]
    total_w = sum(w + 2 * chip_pad_x for (_, (w, _h)) in chip_sizes) + chip_gap * (len(chip_sizes) - 1)
    x = (W - total_w) // 2
    for ((text, color), (tw, th)) in zip(foot_chips, [s for _, s in chip_sizes]):
        x0 = x
        x1 = x + tw + 2 * chip_pad_x
        y0 = foot_y - th // 2 - chip_pad_y
        y1 = foot_y + th // 2 + chip_pad_y
        _rounded(draw, (x0, y0, x1, y1), radius=10,
                 fill=PANEL_BG, outline=color, width=1)
        _text(draw, ((x0 + x1) // 2, foot_y), text, f_foot, color, anchor="mm")
        x = x1 + chip_gap

    # ---- Subtle background: faint chip-grid motif --------------------------
    # Draw very faint dots in a grid to suggest GPU compute lanes
    dot_color = (30, 33, 45)
    for gy in range(panel_bottom + 16, foot_y - 60, 16):
        for gx in range(PAD + 20, W - PAD - 20, 16):
            draw.point((gx, gy), fill=dot_color)

    # ---- Save --------------------------------------------------------------
    out = Path(os.path.expanduser("~/Desktop/kernelforge_devpost_cover.png"))
    img.save(out, format="PNG", optimize=True)
    size_kb = out.stat().st_size // 1024
    print(f"saved: {out}  ({W}x{H}, {size_kb} KB)")


if __name__ == "__main__":
    main()
