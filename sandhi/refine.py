"""Sub-pixel refinement by patch phase correlation.

LoFTR emits matches on its own coarse grid, so its output is only accurate to
about a pixel. `cv2.phaseCorrelate` returns a genuine float shift, which is what
makes "sub-pixel" real rather than a rounded claim: it takes RMSE from 1.212 px
to 0.966 px on the same matches.

The sign convention was measured, not assumed. Taking patch B `s` pixels to the
right of patch A makes phaseCorrelate return `-s`, so the correction ADDS the
returned shift. Subtracting it doubles the error instead of removing it, which
is exactly the bug that made refinement look useless at first (BUGS.md BUG-012).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def refine(a8: np.ndarray, b8: np.ndarray, pa: np.ndarray, pb: np.ndarray,
           half: int | None = None, max_shift: float | None = None):
    """Re-localise matches to sub-pixel precision.

    Returns (pts_a, pts_b, kept_indices, mean_shift). A refinement larger than
    `max_shift` means the two patches disagree about what they contain; those
    matches are dropped, because a wrong sub-pixel answer is worse than none.

    `kept_indices` must be used to select confidences — refinement drops
    arbitrary indices, so slicing by length silently mispairs every match.
    """
    cfg = config.get()
    half = cfg.refine_patch if half is None else half
    max_shift = cfg.refine_max_shift if max_shift is None else max_shift

    h, w = a8.shape
    rh, rw = b8.shape
    win = cv2.createHanningWindow((2 * half, 2 * half), cv2.CV_32F)
    out_a, out_b, moved, kept = [], [], [], []

    for i, ((xa, ya), (xb, yb)) in enumerate(zip(pa, pb)):
        ia, ja = int(round(ya)), int(round(xa))
        ib, jb = int(round(yb)), int(round(xb))
        if not (half <= ia < h - half and half <= ja < w - half
                and half <= ib < rh - half and half <= jb < rw - half):
            continue

        pat_a = a8[ia - half:ia + half, ja - half:ja + half].astype(np.float32)
        pat_b = b8[ib - half:ib + half, jb - half:jb + half].astype(np.float32)
        if pat_a.std() < 1e-3 or pat_b.std() < 1e-3:
            continue
        pat_a = (pat_a - pat_a.mean()) / pat_a.std()
        pat_b = (pat_b - pat_b.mean()) / pat_b.std()

        (dx, dy), _ = cv2.phaseCorrelate(pat_a * win, pat_b * win)
        if np.hypot(dx, dy) > max_shift:
            continue

        out_a.append((xa, ya))
        out_b.append((xb + dx, yb + dy))   # ADD: see module docstring
        moved.append(np.hypot(dx, dy))
        kept.append(i)

    if not out_a:
        return (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int), 0.0)
    return (np.array(out_a), np.array(out_b), np.array(kept),
            float(np.mean(moved)))
