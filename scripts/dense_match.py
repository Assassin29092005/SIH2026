"""Stage C: does a dense matcher beat SIFT, and does it still need Stage B?

Compares four combinations on identical windows, scored against the identity
ground truth:

    SIFT   raw          the classical baseline that fails outright
    SIFT   + Stage B    what we have so far
    LoFTR  raw          does a learned dense matcher survive illumination alone?
    LoFTR  + Stage B    the full pipeline

The second question matters as much as the first. If LoFTR alone handles
cross-illumination, Stage B is redundant and the project should pivot. If it
does not, Stage B is load-bearing and the combination is the contribution.

LoFTR weights are the public MegaDepth-trained 'outdoor' checkpoint from
kornia. It has never seen a lunar image, so this also measures the domain gap.

    python scripts/dense_match.py --size 512
"""

import argparse

import numpy as np
import torch
from kornia.feature import LoFTR

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

INLIER_PX = 3.0
LOFTR_CONF = 0.5

_MODEL = None


def loftr_model() -> LoFTR:
    global _MODEL
    if _MODEL is None:
        _MODEL = LoFTR(pretrained="outdoor").eval()
    return _MODEL


def match_loftr(a8: np.ndarray, b8: np.ndarray, conf: float = LOFTR_CONF):
    """Semi-dense matches from LoFTR. Returns (pts_a, pts_b)."""
    def prep(img):
        t = torch.from_numpy(img.astype(np.float32) / 255.0)
        return t[None, None]  # (1, 1, H, W)

    with torch.inference_mode():
        out = loftr_model()({"image0": prep(a8), "image1": prep(b8)})

    keep = out["confidence"] >= conf
    pa = out["keypoints0"][keep].numpy()
    pb = out["keypoints1"][keep].numpy()
    return pa, pb


def stage_b_pair(dem, scale, lat, morning, evening, el):
    """Normalised 8-bit pair at one elevation, plus the shared validity mask."""
    az_m, _ = best_azimuth_at(dem, scale, lat, morning, el)
    az_e, _ = best_azimuth_at(dem, scale, lat, evening, el)

    nm, vm = normalise(morning, render(dem, scale, lat, az_m, el, shadows=True))
    ne, ve = normalise(evening, render(dem, scale, lat, az_e, el, shadows=True))
    valid = vm & ve

    a8, b8 = to_uint8(nm, valid), to_uint8(ne, valid)
    a8[~valid] = 0
    b8[~valid] = 0
    return a8, b8


def pooled(matcher, dem, scale, lat, morning, evening):
    """Run `matcher` on the Stage-B pair at each elevation and pool the results."""
    all_a, all_b = [], []
    for el in POOL_ELEVATIONS:
        pa, pb = matcher(*stage_b_pair(dem, scale, lat, morning, evening, el))
        if len(pa):
            all_a.append(pa)
            all_b.append(pb)
    if not all_a:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return dedupe(np.vstack(all_a), np.vstack(all_b))


def evaluate(pa, pb, shape) -> dict:
    """Score against the identity ground truth."""
    if len(pa) == 0:
        return {"n": 0, "ok": 0, "rate": 0.0, "rmse": float("nan"), "cov": 0.0}
    err = np.hypot(pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1])
    good = err <= INLIER_PX
    ok = int(good.sum())
    rmse = float(np.sqrt((err[good] ** 2).mean())) if ok else float("nan")
    cov, _ = match_uniformity(pa[good], shape)
    return {"n": len(err), "ok": ok, "rate": ok / len(err), "rmse": rmse, "cov": cov}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--windows", type=int, default=3)
    args = parser.parse_args()

    tiles = find_tiles()
    if not tiles:
        print("No downloaded triples. Run: python scripts/kaguya.py fetch")
        return 1

    triple = open_triple(args.tile or tiles[0])
    size = args.size
    spots = [(2000, 2000), (5888, 5888), (8000, 3000), (4000, 7000)][: args.windows]

    print(f"tile {triple.tile}   {size}x{size}   {len(spots)} windows")
    print("LoFTR: kornia 'outdoor' weights (MegaDepth), never trained on lunar data\n")

    combos = ["SIFT  raw", "SIFT  + Stage B", "LoFTR raw", "LoFTR + Stage B"]
    totals = {c: {"n": 0, "ok": 0, "rmse": [], "cov": []} for c in combos}

    for row, col in spots:
        bands = triple.read_window(row, col, size)
        dem = np.nan_to_num(bands["dem"], nan=float(np.nanmean(bands["dem"])))
        morning, evening = bands["morning"], bands["evening"]
        lat, _ = triple.pixel_latlon(row + size / 2, col + size / 2)
        scale = abs(triple.transform.a)

        m8, e8 = to_uint8(morning), to_uint8(evening)
        results = {
            "SIFT  raw": evaluate(*match_sift(m8, e8), (size, size)),
            "SIFT  + Stage B": evaluate(
                *pooled(match_sift, dem, scale, lat, morning, evening), (size, size)),
            "LoFTR raw": evaluate(*match_loftr(m8, e8), (size, size)),
            "LoFTR + Stage B": evaluate(
                *pooled(match_loftr, dem, scale, lat, morning, evening), (size, size)),
        }
        for name, r in results.items():
            totals[name]["n"] += r["n"]
            totals[name]["ok"] += r["ok"]
            if not np.isnan(r["rmse"]):
                totals[name]["rmse"].append(r["rmse"])
            totals[name]["cov"].append(r["cov"])

    print(f"{'method':<18} {'correct':>8} {'matches':>8} {'rate':>6} {'RMSE':>8} {'cov':>6}")
    print("-" * 60)
    for name in combos:
        t = totals[name]
        rate = t["ok"] / max(t["n"], 1)
        rmse = np.mean(t["rmse"]) if t["rmse"] else float("nan")
        print(f"{name:<18} {t['ok']:>8} {t['n']:>8} {rate*100:>5.0f}% "
              f"{rmse:>8.3f} {np.mean(t['cov']):>6.2f}")

    print("\nRMSE in source pixels, over matches within 3 px. cov = grid coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
