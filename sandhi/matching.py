"""Dense matching: LoFTR, tiled for coverage, with coarse alignment first.

Lunar terrain is texture-poor, so detector-based methods starve — SIFT scatters
by 53 px across windows and is unusable. LoFTR is semi-dense and survives, using
the public MegaDepth `outdoor` weights (never trained on lunar imagery).

Two structural details matter more than the matcher choice:

* **Coarse alignment first.** Tiled matching only searches a small pad around
  each tile, so it silently fails on a grossly displaced pair. OHRC sits ~437 px
  from its Kaguya reference — far outside a 16 px pad — and without this stage
  the case yields 7 matches instead of 1443.
* **Tiling, not selection.** Bucketed selection cannot raise coverage: it only
  redistributes matches among cells that already hold some, and empty cells stay
  empty. Coverage has to be forced where matches are produced.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config

_MODEL = None


def loftr_model():
    """Lazily construct LoFTR. Downloads ~44 MB of weights on first use."""
    global _MODEL
    if _MODEL is None:
        from kornia.feature import LoFTR
        _MODEL = LoFTR(pretrained="outdoor").eval()
    return _MODEL


def match(a8: np.ndarray, b8: np.ndarray, conf_thresh: float | None = None):
    """Whole-image LoFTR match. Returns (pts_a, pts_b, confidences)."""
    thresh = config.get().loftr_conf if conf_thresh is None else conf_thresh

    def prep(img):
        return torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]

    with torch.inference_mode():
        out = loftr_model()({"image0": prep(a8), "image1": prep(b8)})
    keep = out["confidence"] >= thresh
    return (out["keypoints0"][keep].numpy(),
            out["keypoints1"][keep].numpy(),
            out["confidence"][keep].numpy())


def coarse_offset(a8: np.ndarray, b8: np.ndarray):
    """Gross translation from a whole-image match. Returns (dx, dy, n)."""
    cfg = config.get()
    pa, pb, _ = match(a8, b8)
    if len(pa) < cfg.min_matches:
        return 0, 0, len(pa)
    dx = float(np.median(pb[:, 0] - pa[:, 0]))
    dy = float(np.median(pb[:, 1] - pa[:, 1]))
    return int(round(dx)), int(round(dy)), len(pa)


def match_tiled(a8: np.ndarray, b8: np.ndarray, tiles: int | None = None,
                pad: int | None = None):
    """Match tile by tile so every region gets its own attempt."""
    cfg = config.get()
    tiles = cfg.tiles if tiles is None else tiles
    pad = cfg.tile_pad if pad is None else pad

    h, w = a8.shape
    # Adapt the tile count to the image shape. A narrow strip would otherwise be
    # cut below min_tile_px and the match count collapses.
    tiles = max(1, min(tiles, min(h, w) // cfg.min_tile_px))
    step_y, step_x = h / tiles, w / tiles

    all_a, all_b, all_c = [], [], []
    for ty in range(tiles):
        for tx in range(tiles):
            y0, x0 = int(ty * step_y), int(tx * step_x)
            y1, x1 = int((ty + 1) * step_y), int((tx + 1) * step_x)
            # LoFTR needs dimensions divisible by 8.
            ah, aw = ((y1 - y0) // 8) * 8, ((x1 - x0) // 8) * 8
            if ah < 32 or aw < 32:
                continue

            ry0, rx0 = max(0, y0 - pad), max(0, x0 - pad)
            rh = ((min(h, y0 + ah + pad) - ry0) // 8) * 8
            rw = ((min(w, x0 + aw + pad) - rx0) // 8) * 8
            if rh < 32 or rw < 32:
                continue

            pa, pb, conf = match(a8[y0:y0 + ah, x0:x0 + aw],
                                 b8[ry0:ry0 + rh, rx0:rx0 + rw])
            if len(pa) == 0:
                continue
            all_a.append(pa + np.array([x0, y0]))
            all_b.append(pb + np.array([rx0, ry0]))
            all_c.append(conf)

    if not all_a:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    return np.vstack(all_a), np.vstack(all_b), np.concatenate(all_c)


def match_aligned(a8: np.ndarray, b8: np.ndarray, tiles: int | None = None):
    """Coarse-align, then tile. Returns (pts_a, pts_b, conf, (dx, dy), n_coarse).

    The reference is rolled into rough alignment so each tile's small search pad
    looks at the right ground, and the shift is undone on the returned points.
    """
    dx, dy, n_coarse = coarse_offset(a8, b8)
    b_aligned = np.roll(b8, (-dy, -dx), axis=(0, 1)) if (dx or dy) else b8
    pa, pb, conf = match_tiled(a8, b_aligned, tiles=tiles)
    if len(pb):
        pb = pb + np.array([dx, dy], dtype=float)
    return pa, pb, conf, (dx, dy), n_coarse
