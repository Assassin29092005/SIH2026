"""Scale handling: match across a resolution ratio by working at a common GSD.

The earlier `scale_test.py` normalised the fine image at fine resolution and the
coarse image at coarse resolution, then tried to match the two. That fails, and
measurably so: a Stage-B product normalised at full resolution and then
decimated correlates with one normalised directly at the coarse resolution at
only 0.56 (k=2), 0.27 (k=4) and -0.09 (k=8). A coarse DEM cancels different
shading than a fine DEM does, so the two normalised products are not comparable.

Same principle as the same-elevation rule in `ablation.py`: **the normalisation
parameters must match across the pair.** Resolution is one of those parameters.

So the pipeline is:

    1. Estimate (or read) the resolution ratio k.
    2. Decimate the fine image AND the DEM to the coarse GSD.
    3. Normalise BOTH sides at that common GSD.
    4. Match; correspondence lives in the coarse frame.

Accuracy is reported in COARSE pixels, deliberately. When the reference is k
times coarser, sub-source-pixel accuracy is not recoverable from it -- the
information is not there. Sub-coarse-pixel is the honest target, and it maps
back to k source pixels.

    python scripts/scale_pipeline.py --size 1024
    python scripts/scale_pipeline.py --estimate
"""

import argparse

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
from dense_match import match_loftr
from photometric import render
from scale_test import decimate
from triple_io import find_tiles, open_triple

SCALE_FACTORS = (1, 2, 4, 8, 16)
CANDIDATE_SCALES = (1, 2, 3, 4, 6, 8, 12, 16)
INLIER_COARSE_PX = 3.0


def normalise_at(image, dem, map_scale, lat, elevation):
    """Stage-B normalise one image against a DEM already at the same GSD."""
    az, _ = best_azimuth_at(dem, map_scale, lat, image, elevation)
    norm, valid = normalise(image, render(dem, map_scale, lat, az, elevation, shadows=True))
    return norm, valid


def match_common_gsd(source, reference, dem, base_scale, lat, k, matcher,
                     elevations=POOL_ELEVATIONS):
    """Match a fine source against a k-times coarser reference.

    Both sides are brought to the coarse GSD before any normalisation happens.
    Returned coordinates are in the COARSE frame, where ground truth is the
    identity.
    """
    src_k = decimate(source.astype(np.float32), k)
    dem_k = decimate(dem, k)
    scale_k = base_scale * k

    # Crop to a common shape; decimation can leave a one-pixel difference.
    h = min(src_k.shape[0], reference.shape[0])
    w = min(src_k.shape[1], reference.shape[1])
    src_k, ref_k = src_k[:h, :w], reference[:h, :w]
    dem_k = dem_k[:h, :w]

    all_a, all_b = [], []
    for el in elevations:
        na, va = normalise_at(src_k, dem_k, scale_k, lat, el)
        nb, vb = normalise_at(ref_k, dem_k, scale_k, lat, el)
        valid = va & vb

        a8, b8 = to_uint8(na, valid), to_uint8(nb, valid)
        a8[~valid] = 0
        b8[~valid] = 0

        pa, pb = matcher(a8, b8)
        if len(pa):
            all_a.append(pa)
            all_b.append(pb)

    if not all_a:
        return np.zeros((0, 2)), np.zeros((0, 2)), (h, w)
    pa, pb = dedupe(np.vstack(all_a), np.vstack(all_b))
    return pa, pb, (h, w)


def score(pa, pb, shape) -> dict:
    """Score in coarse pixels against the identity ground truth."""
    if len(pa) == 0:
        return {"n": 0, "ok": 0, "rate": 0.0, "rmse": float("nan"), "cov": 0.0}
    err = np.hypot(pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1])
    good = err <= INLIER_COARSE_PX
    ok = int(good.sum())
    rmse = float(np.sqrt((err[good] ** 2).mean())) if ok else float("nan")
    cov, _ = match_uniformity(pa[good], shape)
    return {"n": len(err), "ok": ok, "rate": ok / len(err), "rmse": rmse, "cov": cov}


def estimate_scale(source, reference, dem, base_scale, lat,
                   candidates=CANDIDATE_SCALES) -> tuple[int, dict]:
    """Recover an unknown resolution ratio by trying each candidate.

    The correct k should yield far more correct matches than any wrong k, so
    match count alone is a usable signal without knowing the answer.
    """
    best_k, best = candidates[0], {"ok": -1}
    results = {}
    for k in candidates:
        if min(reference.shape) * k > min(source.shape) * 1.05:
            continue  # k implies a reference larger than the source
        pa, pb, shape = match_common_gsd(source, reference, dem, base_scale, lat,
                                         k, match_loftr, elevations=(5.0,))
        s = score(pa, pb, shape)
        results[k] = s
        if s["ok"] > best["ok"]:
            best_k, best = k, s
    return best_k, results


def run_ratios(triple, spots, size, matcher, matcher_name) -> None:
    print(f"\n{matcher_name} at common GSD")
    print(f"{'ratio':>6} {'coarse m/px':>12} | {'correct':>8} {'matches':>8} "
          f"{'rate':>6} {'RMSE':>8} {'cov':>6}")
    print("-" * 66)

    base_scale = abs(triple.transform.a)
    for k in SCALE_FACTORS:
        tot = {"ok": 0, "n": 0}
        rmses, covs = [], []
        for row, col in spots:
            bands = triple.read_window(row, col, size)
            dem = np.nan_to_num(bands["dem"], nan=float(np.nanmean(bands["dem"])))
            lat, _ = triple.pixel_latlon(row + size / 2, col + size / 2)
            reference = decimate(bands["evening"].astype(np.float32), k)

            pa, pb, shape = match_common_gsd(bands["morning"], reference, dem,
                                             base_scale, lat, k, matcher)
            s = score(pa, pb, shape)
            tot["ok"] += s["ok"]
            tot["n"] += s["n"]
            if not np.isnan(s["rmse"]):
                rmses.append(s["rmse"])
            covs.append(s["cov"])

        rate = tot["ok"] / max(tot["n"], 1)
        rmse = np.mean(rmses) if rmses else float("nan")
        print(f"{k:>5}x {base_scale*k:>12.2f} | {tot['ok']:>8} {tot['n']:>8} "
              f"{rate*100:>5.0f}% {rmse:>8.3f} {np.mean(covs):>6.2f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default=None)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--sift", action="store_true", help="also run the SIFT baseline")
    parser.add_argument("--estimate", action="store_true", help="recover an unknown ratio")
    args = parser.parse_args()

    tiles = find_tiles()
    if not tiles:
        print("No downloaded triples. Run: python scripts/kaguya.py fetch")
        return 1

    triple = open_triple(args.tile or tiles[0])
    spots = [(2000, 2000), (5888, 5888), (8000, 3000)][: args.windows]
    size = args.size
    print(f"tile {triple.tile}   {size}x{size}   {len(spots)} windows")
    print("RMSE and inlier threshold are in COARSE reference pixels.")

    if args.estimate:
        row, col = spots[0]
        bands = triple.read_window(row, col, size)
        dem = np.nan_to_num(bands["dem"], nan=float(np.nanmean(bands["dem"])))
        lat, _ = triple.pixel_latlon(row + size / 2, col + size / 2)
        base_scale = abs(triple.transform.a)

        for true_k in (2, 4, 8):
            reference = decimate(bands["evening"].astype(np.float32), true_k)
            got, results = estimate_scale(bands["morning"], reference, dem,
                                          base_scale, lat)
            table = "  ".join(f"{k}:{s['ok']}" for k, s in sorted(results.items()))
            verdict = "OK" if got == true_k else "WRONG"
            print(f"\n  true k={true_k:>2}  recovered k={got:>2}  {verdict}")
            print(f"    correct matches per candidate -> {table}")
        return 0

    if args.sift:
        run_ratios(triple, spots, size, match_sift, "SIFT")
    run_ratios(triple, spots, size, match_loftr, "LoFTR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Recovering the ratio from metadata, which is what the real task needs.
#
# Blind estimation by match count or matcher confidence is CONFOUNDED here, and
# measurably so: both sides get normalised against the same decimated DEM, so
# both inherit identical DEM-derived structure that the matcher locks onto
# whatever k is tried. Measured across k=1..12 against a true k of 4 and 8,
# match counts stay flat (708/664/712/727) and mean confidence never leaves
# 0.90-0.97. See BUGS.md BUG-010.
#
# It does not matter, because every product in this project declares its own
# ground sample distance:
#
#   OHRC       PDS4 label   pixel_resolution = 0.26 m
#   TMC-2      GeoTIFF      geotransform, ~5.05 m at the equator
#   Kaguya TC  PDS3 label   MAP_SCALE = 7.403 m/px
#   LROC NAC   ODE metadata resolution field
#
# So k is read, not guessed.

MOON_RADIUS_M = 1737400.0


def gsd_metres(dataset) -> float:
    """Ground sample distance in metres for an open rasterio dataset.

    Handles both projected products (metres already) and geographic ones like
    the TMC-2 GeoTIFFs, whose transform is in DEGREES on a lunar sphere.
    """
    res = abs(dataset.transform.a)
    crs = dataset.crs
    if crs is not None and getattr(crs, "is_geographic", False):
        return float(np.radians(res) * MOON_RADIUS_M)
    return float(res)


def scale_ratio(fine_gsd_m: float, coarse_gsd_m: float) -> float:
    """How many fine pixels span one coarse pixel."""
    if fine_gsd_m <= 0:
        raise ValueError(f"bad fine GSD {fine_gsd_m}")
    return coarse_gsd_m / fine_gsd_m
