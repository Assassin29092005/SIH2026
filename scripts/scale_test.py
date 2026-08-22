"""Scale invariance: match across a resolution ratio, with and without Stage B.

The problem statement's second axis. Chandrayaan-2 OHRC is ~0.25 m/px while
reference basemaps run to ~100 m/px, so a real pipeline faces ratios in the
hundreds. Here we simulate the ratio by downsampling one side of a pair whose
true correspondence we already know exactly.

This is a controlled degradation of REAL imagery, not synthetic data: both
images are genuine Kaguya observations, and only the sampling changes. The
ground truth stays exact -- if the reference is decimated by k, then source
pixel (x, y) corresponds to reference pixel (x/k, y/k), so a match's error in
source pixels is hypot(xb*k - xa, yb*k - ya).

    python scripts/scale_test.py --size 1024
"""

import argparse

import cv2
import numpy as np

from ablation import (
    POOL_ELEVATIONS,
    best_azimuth_at,
    dedupe,
    match_sift,
    match_uniformity,
    normalise,
    to_uint8,
)
from photometric import render
from triple_io import find_tiles, open_triple

SCALE_FACTORS = (1, 2, 4, 8, 16)
INLIER_PX = 3.0


def decimate(img: np.ndarray, k: int) -> np.ndarray:
    """Downsample by an integer factor with area averaging (mimics a coarser sensor)."""
    if k == 1:
        return img
    h, w = img.shape[0] // k, img.shape[1] // k
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def score_scaled(pa: np.ndarray, pb: np.ndarray, k: int, shape: tuple) -> dict:
    """Score matches where pb lives in an image decimated by k."""
    if len(pa) == 0:
        return {"matches": 0, "correct": 0, "rate": 0.0, "rmse": float("nan"),
                "coverage": 0.0}
    err = np.hypot(pb[:, 0] * k - pa[:, 0], pb[:, 1] * k - pa[:, 1])
    ok = err <= INLIER_PX * k  # tolerance scales with the coarser pixel
    correct = int(ok.sum())
    rmse = float(np.sqrt((err[ok] ** 2).mean())) if correct else float("nan")
    coverage, _ = match_uniformity(pa[ok], shape)
    return {"matches": len(err), "correct": correct,
            "rate": correct / len(err), "rmse": rmse, "coverage": coverage}


def intersect_masks(valid_fine: np.ndarray, valid_coarse: np.ndarray, k: int):
    """Make two validity masks at different resolutions cover the same ground.

    Critical, not cosmetic. `to_uint8` stretches by percentiles over the valid
    pixels, so if the two masks cover different terrain the two images get
    different intensity mappings and their descriptors stop being comparable.
    Independent masks dropped this pipeline from 47/67 correct to 0/17.
    See BUGS.md BUG-008.
    """
    if k == 1:
        both = valid_fine & valid_coarse
        return both, both

    # Coarse mask -> fine grid by pixel replication.
    up = np.repeat(np.repeat(valid_coarse, k, axis=0), k, axis=1)
    h, w = valid_fine.shape
    up = up[:h, :w]
    fine = valid_fine & up

    # Fine mask -> coarse grid; a coarse pixel is valid only if mostly valid.
    down = decimate(valid_fine.astype(np.float32), k) > 0.5
    ch, cw = valid_coarse.shape
    coarse = valid_coarse & down[:ch, :cw]
    return fine, coarse


def stage_b_scaled(dem, scale, lat, source, reference, k):
    """Normalise both sides at their own resolution, then match across the ratio."""
    dem_k = decimate(dem, k)
    scale_k = scale * k

    all_a, all_b = [], []
    for el in POOL_ELEVATIONS:
        az_s, _ = best_azimuth_at(dem, scale, lat, source, el)
        az_r, _ = best_azimuth_at(dem_k, scale_k, lat, reference, el)

        ns, vs = normalise(source, render(dem, scale, lat, az_s, el, shadows=True))
        nr, vr = normalise(reference, render(dem_k, scale_k, lat, az_r, el, shadows=True))
        vs, vr = intersect_masks(vs, vr, k)

        a8, b8 = to_uint8(ns, vs), to_uint8(nr, vr)
        a8[~vs] = 0
        b8[~vr] = 0

        pa, pb = match_sift(a8, b8)
        if len(pa):
            all_a.append(pa)
            all_b.append(pb)

    if not all_a:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return dedupe(np.vstack(all_a), np.vstack(all_b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default=None)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--windows", type=int, default=3)
    args = parser.parse_args()

    tiles = find_tiles()
    if not tiles:
        print("No downloaded triples. Run: python scripts/kaguya.py fetch")
        return 1

    triple = open_triple(args.tile or tiles[0])
    size = args.size
    spots = [(2000, 2000), (5888, 5888), (8000, 3000)][: args.windows]

    print(f"tile {triple.tile}   window {size}x{size}   {len(spots)} windows")
    print("source = morning at full resolution, reference = evening decimated\n")
    print(f"{'ratio':>6} {'m/px':>7} | {'raw ok':>6} {'raw n':>6} {'rate':>5} "
          f"| {'B ok':>5} {'B n':>5} {'rate':>5} {'RMSE':>7} {'cov':>5}")

    for k in SCALE_FACTORS:
        raw_t = {"correct": 0, "matches": 0}
        stb_t = {"correct": 0, "matches": 0}
        rmses, covs = [], []

        for row, col in spots:
            bands = triple.read_window(row, col, size)
            dem = np.nan_to_num(bands["dem"], nan=float(np.nanmean(bands["dem"])))
            source = bands["morning"]
            reference = decimate(bands["evening"], k)
            lat, _ = triple.pixel_latlon(row + size / 2, col + size / 2)
            scale = abs(triple.transform.a)

            r = score_scaled(*match_sift(to_uint8(source), to_uint8(reference)),
                             k, (size, size))
            b = score_scaled(*stage_b_scaled(dem, scale, lat, source, reference, k),
                             k, (size, size))

            for tgt, s in ((raw_t, r), (stb_t, b)):
                tgt["correct"] += s["correct"]
                tgt["matches"] += s["matches"]
            if not np.isnan(b["rmse"]):
                rmses.append(b["rmse"])
            covs.append(b["coverage"])

        rr = raw_t["correct"] / max(raw_t["matches"], 1)
        br = stb_t["correct"] / max(stb_t["matches"], 1)
        rmse = np.mean(rmses) if rmses else float("nan")
        print(f"{k:>5}x {scale*k:>7.2f} | {raw_t['correct']:>6} {raw_t['matches']:>6} "
              f"{rr*100:>4.0f}% | {stb_t['correct']:>5} {stb_t['matches']:>5} "
              f"{br*100:>4.0f}% {rmse:>7.3f} {np.mean(covs):>5.2f}")

    print("\nRMSE is in COARSE reference pixels; inlier tolerance scales with k.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
