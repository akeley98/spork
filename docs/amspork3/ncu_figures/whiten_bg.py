#!/usr/bin/env python3
"""Replace the light gray (#E6E6E6) nsight compute background with pure white.

Each pixel is modeled as a blend  t * ink + (1 - t) * bg  of the background with
one "ink": the blue/orange bar colors or black (text, gridlines). The pixel is
re-blended over white instead,  t * ink + (1 - t) * white,  so solid bar colors and
black text are unchanged and only the background share of each pixel is lightened.
Pixels that fit no ink well (e.g. bar/text boundaries) are left alone.

Usage: whiten_bg.py IN.png OUT.png
"""
import sys

import numpy as np
from PIL import Image

BG = np.array([230, 230, 230], dtype=np.float64)
WHITE = np.array([255, 255, 255], dtype=np.float64)
INKS = np.array([
    [0, 0, 0],        # text, gridlines, axis labels
    [99, 127, 255],   # blue bars (#637FFF)
    [223, 119, 99],   # orange bars (#DF7763)
    [255, 255, 255],  # lighter-than-bg grays (screenshot separator line) -> white
], dtype=np.float64)
MAX_RESIDUAL = 6.0  # RGB distance from the bg->ink segment to count as a blend


def whiten(rgb):
    px = rgb.reshape(-1, 3).astype(np.float64)
    best_t = np.zeros(len(px))
    best_ink = np.zeros(len(px), dtype=int)
    best_res = np.full(len(px), np.inf)
    for i, ink in enumerate(INKS):
        d = ink - BG
        t = np.clip((px - BG) @ d / (d @ d), 0.0, 1.0)
        res = np.linalg.norm(px - (BG + t[:, None] * d), axis=1)
        better = res < best_res
        best_t[better], best_ink[better], best_res[better] = t[better], i, res[better]
    fit = best_res <= MAX_RESIDUAL
    t = best_t[:, None]
    out = px.copy()
    out[fit] = (t * INKS[best_ink] + (1.0 - t) * WHITE)[fit]
    print(f"remapped {fit.mean():.1%} of pixels; left {(~fit).sum()} unmatched", file=sys.stderr)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8).reshape(rgb.shape)


def main():
    src, dst = sys.argv[1:3]
    img = Image.open(src).convert("RGBA")
    arr = np.array(img)
    arr[..., :3] = whiten(arr[..., :3])
    Image.fromarray(arr).save(dst)


if __name__ == "__main__":
    main()
