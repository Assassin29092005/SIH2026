"""The headline experiment: does Stage B make a cross-illumination pair matchable?

Morning and evening images of identical terrain anti-correlate (-0.560 measured),
and classical descriptors fail on them. Stage B divides each image by a DEM
render of its own illumination geometry, cancelling the topographic shading and
leaving albedo-like structure that should match.

Ground truth here is exact and needs no estimation: the three products share one
map grid, so the true correspondence is the IDENTITY. A match from (x1,y1) to
(x2,y2) therefore has error hypot(x2-x1, y2-y1) directly.

Metrics are the ones the problem statement asks for -- RMSE, inlier count,
inlier ratio -- plus match uniformity, which it also asks for and which most
implementations skip.

    python scripts/ablation.py --size 512
"""

import argparse

import cv2
import numpy as np

from estimate_sun import correlate
from photometric import render
from triple_io import find_tiles, open_triple

# Below this rendered reflectance a pixel is shadowed or grazing, and dividing
# by it amplifies noise instead of removing shading.
SHADOW_FLOOR = 0.08

INLIER_THRESHOLDS_PX = (1.0, 3.0, 5.0)


def to_uint8(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Percentile stretch to 8-bit, which is what OpenCV's SIFT wants."""
    valid = img[mask] if mask is not None else img[np.isfinite(img)]
    if valid.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def normalise(image: np.ndarray, rendered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide out the modelled topographic shading. Returns (normalised, valid_mask)."""
    valid = rendered >= SHADOW_FLOOR
    ratio = np.zeros(image.shape, np.float64)
    ratio[valid] = image[valid].astype(np.float64) / rendered[valid]
    return ratio, valid


# Candidate sun elevations. `STANDARD_GEOMETRY = (30, 0, 30)` in the label
# describes the photometric CORRECTION applied, not the acquisition geometry --
# these mosaics are shot near the terminator and the correction rescales
# brightness without removing shadows baked in at low sun. Assuming 30 deg
# incidence produced a nearly flat render (cv 0.045) and a dead ablation.
# See BUGS.md BUG-007.
ELEVATION_CANDIDATES = (5, 10, 15, 20, 30, 45, 60)


def best_geometry(dem, scale, lat, target, az_step=15) -> tuple[float, float, float]:
    """Sun (azimuth, elevation) whose render correlates best with `target`.

    Two stages, because ray-marched shadows dominate the cost. Shading alone
    ranks candidates well enough to find the neighbourhood, so the coarse pass
    skips shadows entirely and only the shortlist pays for them.
    """
    coarse = []
    for el in ELEVATION_CANDIDATES:
        for az in range(0, 360, az_step):
            r = correlate(render(dem, scale, lat, az, el, shadows=False), target)
            coarse.append((r, float(az), float(el)))

    coarse.sort(reverse=True)
    best_az, best_el, best_r = coarse[0][1], coarse[0][2], -2.0
    for _, az, el in coarse[:6]:
        r = correlate(render(dem, scale, lat, az, el, shadows=True), target)
        if r > best_r:
            best_az, best_el, best_r = az, el, r
    return best_az, best_el, best_r


def match_sift(a8: np.ndarray, b8: np.ndarray, ratio_test: float = 0.75):
    """SIFT + Lowe ratio test. Returns (pts_a, pts_b) as float arrays."""
    sift = cv2.SIFT_create()
    ka, da = sift.detectAndCompute(a8, None)
    kb, db = sift.detectAndCompute(b8, None)
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))

    matcher = cv2.BFMatcher()
    pairs = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio_test * n.distance]
    if not good:
        return np.zeros((0, 2)), np.zeros((0, 2))

    pa = np.array([ka[m.queryIdx].pt for m in good])
    pb = np.array([kb[m.trainIdx].pt for m in good])
    return pa, pb


def match_uniformity(pts: np.ndarray, shape: tuple, grid: int = 8) -> tuple[float, float]:
    """Grid coverage fraction and normalised entropy of the match distribution."""
    if len(pts) == 0:
        return 0.0, 0.0
    h, w = shape
    gy = np.clip((pts[:, 1] / h * grid).astype(int), 0, grid - 1)
    gx = np.clip((pts[:, 0] / w * grid).astype(int), 0, grid - 1)
    counts = np.bincount(gy * grid + gx, minlength=grid * grid).astype(float)
    coverage = float((counts > 0).mean())
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(grid * grid))
    return coverage, entropy


def score(pa: np.ndarray, pb: np.ndarray, shape: tuple) -> dict:
    """Score matches against the identity ground truth."""
    if len(pa) == 0:
        return {"matches": 0, "rmse": float("nan"), "inliers": {t: 0 for t in INLIER_THRESHOLDS_PX},
                "ratio": {t: 0.0 for t in INLIER_THRESHOLDS_PX}, "coverage": 0.0, "entropy": 0.0}

    err = np.hypot(pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1])
    inliers = {t: int((err <= t).sum()) for t in INLIER_THRESHOLDS_PX}
    ratios = {t: inliers[t] / len(err) for t in INLIER_THRESHOLDS_PX}

    # RMSE over the 3 px inlier set: RMSE over all matches is dominated by
    # wild outliers and says nothing about localisation accuracy.
    good = err[err <= 3.0]
    rmse = float(np.sqrt((good**2).mean())) if good.size else float("nan")

    coverage, entropy = match_uniformity(pa[err <= 3.0], shape)
    return {"matches": len(err), "rmse": rmse, "inliers": inliers,
            "ratio": ratios, "coverage": coverage, "entropy": entropy}


def report(name: str, s: dict) -> None:
    inl = "  ".join(f"<={t:g}px: {s['inliers'][t]:>4} ({s['ratio'][t]*100:5.1f}%)"
                    for t in INLIER_THRESHOLDS_PX)
    print(f"  {name:<26} matches={s['matches']:>5}   {inl}")
    print(f"  {'':<26} RMSE(<=3px)={s['rmse']:.3f}px   "
          f"coverage={s['coverage']:.2f}  entropy={s['entropy']:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default=None)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    tiles = find_tiles()
    if not tiles:
        print("No downloaded triples. Run: python scripts/kaguya.py fetch")
        return 1

    triple = open_triple(args.tile or tiles[0])
    size = args.size
    row = (triple.height - size) // 2
    col = (triple.width - size) // 2
    bands = triple.read_window(row, col, size)

    dem = np.nan_to_num(bands["dem"], nan=float(np.nanmean(bands["dem"])))
    morning, evening = bands["morning"], bands["evening"]
    lat, _ = triple.pixel_latlon(row + size / 2, col + size / 2)
    scale = abs(triple.transform.a)

    print(f"tile {triple.tile}  window ({row},{col}) {size}x{size}  lat {lat:.3f}\n")

    az_m, el_m, r_m = best_geometry(dem, scale, lat, morning)
    az_e, el_e, r_e = best_geometry(dem, scale, lat, evening)
    print(f"recovered sun  morning: az={az_m:.0f} el={el_m:.0f}  r={r_m:+.3f}")
    print(f"               evening: az={az_e:.0f} el={el_e:.0f}  r={r_e:+.3f}")

    render_m = render(dem, scale, lat, az_m, el_m, shadows=True)
    render_e = render(dem, scale, lat, az_e, el_e, shadows=True)

    norm_m, valid_m = normalise(morning, render_m)
    norm_e, valid_e = normalise(evening, render_e)
    valid = valid_m & valid_e
    print(f"valid (unshadowed in both): {100*valid.mean():.1f}% of window\n")

    raw_corr = correlate(morning[valid], evening[valid])
    norm_corr = correlate(norm_m[valid], norm_e[valid])
    print("CORRELATION")
    print(f"  raw morning vs evening        {raw_corr:+.3f}")
    print(f"  Stage-B normalised            {norm_corr:+.3f}")
    print(f"  swing                         {norm_corr - raw_corr:+.3f}\n")

    m8, e8 = to_uint8(morning), to_uint8(evening)
    nm8, ne8 = to_uint8(norm_m, valid), to_uint8(norm_e, valid)
    # Suppress shadowed pixels so SIFT does not key on the mask edges.
    nm8[~valid] = 0
    ne8[~valid] = 0

    print("SIFT MATCHING vs identity ground truth")
    raw_score = score(*match_sift(m8, e8), (size, size))
    report("raw", raw_score)
    nrm_score = score(*match_sift(nm8, ne8), (size, size))
    report("Stage-B normalised", nrm_score)

    base = raw_score["inliers"][3.0]
    got = nrm_score["inliers"][3.0]
    print()
    if base == 0 and got == 0:
        print("VERDICT: neither pass matched. Try a larger window or another tile.")
    elif base == 0:
        print(f"VERDICT: raw found 0 correct matches, Stage B found {got}.")
    else:
        print(f"VERDICT: correct matches {base} -> {got}  ({got/base:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
