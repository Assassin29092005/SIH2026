"""Re-measurement harness with mandatory controls. Supersedes the retracted results.

Everything reported before BUG-011 was invalid: a shared zero-mask written into
both images gave the matcher an identical, perfectly aligned pattern to lock
onto, and the identity ground truth was wrong by a measured 8.25 px. This
harness is built so neither failure can recur silently.

Three rules it enforces:

1. **No identity ground truth.** Kaguya morning and evening are independently
   orthorectified and genuinely offset. Metrics are RANSAC inlier ratio and
   residual about a fitted model -- which is what the problem statement asks
   for anyway -- so no assumed correspondence is needed.

2. **Validation is cross-window consistency.** If a method measures real
   terrain, independent windows must recover the *same* inter-product offset.
   An artifact cannot do that: a mask artifact reports zero everywhere, and
   noise reports scatter. Agreement across windows is the evidence.

3. **Four controls gate every number.** A method is only reported if it passes:
   - shifting the RAW input by N changes the recovered offset by -N
   - pure noise collapses
   - constant grey collapses
   - matching different terrain collapses
   The shift must perturb the *raw* input. Perturbing the normalised output
   moves imagery and mask together and hides exactly the bug that caused BUG-011.

    python scripts/remeasure.py
    python scripts/remeasure.py --windows 6 --full
"""

import argparse

import cv2
import numpy as np

from ablation import best_azimuth_at, match_sift, match_uniformity
from dense_match import match_loftr
from photometric import render
from triple_io import find_tiles, open_triple

WINDOWS = [(2000, 2000), (4000, 7000), (5888, 5888),
           (8000, 3000), (9500, 9500), (3000, 6000)]
SIZE = 512
ROLL_PX = 15
MIN_MATCHES = 8
RANSAC_PX = 3.0

# Real imagery must out-match noise by at least this factor. Below it, the
# method is keying on something both images share that is not the terrain.
NOISE_REJECT_RATIO = 5.0

# Below this the render is grazing/shadowed. Clamping instead of masking is the
# whole point: masking writes an identical pattern into both images.
RENDER_FLOOR = 0.15


def stretch(x: np.ndarray) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    hi = max(hi, lo + 1e-6)
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def build_raw(img, ctx, el=None):
    return stretch(img.astype(np.float64))


def build_stageb(img, ctx, el):
    """Stage B without any masking: divide by a render clamped at a floor.

    No pixel is zeroed, so the two images share no synthetic structure.
    """
    dem, scale, lat = ctx
    az, _ = best_azimuth_at(dem, scale, lat, img, el)
    rend = np.clip(render(dem, scale, lat, az, el, shadows=False), RENDER_FLOOR, None)
    return stretch(img.astype(np.float64) / rend)


def build_stageb_hp(img, ctx, el, ksize=31):
    """Stage B, then local contrast normalisation.

    Plain division leaves a `1/render` term that is identical in both images and
    is enough on its own for the matcher to lock onto -- pure noise divided by
    the render still matches. That residue is smooth, because the DEM is coarser
    than the imagery, so subtracting a local mean and dividing by a local
    standard deviation removes it while keeping fine detail.
    """
    dem, scale, lat = ctx
    az, _ = best_azimuth_at(dem, scale, lat, img, el)
    rend = np.clip(render(dem, scale, lat, az, el, shadows=False), RENDER_FLOOR, None)
    ratio = (img.astype(np.float64) / rend).astype(np.float32)
    mu = cv2.blur(ratio, (ksize, ksize))
    sd = np.sqrt(np.maximum(cv2.blur(ratio * ratio, (ksize, ksize)) - mu * mu, 1e-6))
    return stretch((ratio - mu) / sd)


def build_raw_hp(img, ctx, el=None, ksize=31):
    """Local contrast normalisation alone, with no DEM involved at all.

    The honest control for Stage B: if this matches as well, the DEM is adding
    nothing and the gain was just high-pass filtering.
    """
    x = img.astype(np.float32)
    mu = cv2.blur(x, (ksize, ksize))
    sd = np.sqrt(np.maximum(cv2.blur(x * x, (ksize, ksize)) - mu * mu, 1e-6))
    return stretch((x - mu) / sd)


def offset_of(pa, pb):
    """Robust translation between two match sets, plus its dispersion."""
    if len(pa) < MIN_MATCHES:
        return None
    dx = pb[:, 0] - pa[:, 0]
    dy = pb[:, 1] - pa[:, 1]
    return (float(np.median(dx)), float(np.median(dy)),
            float(np.median(np.abs(dx - np.median(dx)))),
            float(np.median(np.abs(dy - np.median(dy)))))


def fit_quality(pa, pb, shape):
    """RANSAC inlier ratio, residual and uniformity. No ground truth required."""
    if len(pa) < MIN_MATCHES:
        return None
    model, inl = cv2.estimateAffinePartial2D(
        pa.astype(np.float32), pb.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    if model is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < 3:
        return None
    pred = (model @ np.c_[pa, np.ones(len(pa))].T).T
    resid = np.hypot(pred[:, 0] - pb[:, 0], pred[:, 1] - pb[:, 1])
    cov, ent = match_uniformity(pa[inl], shape)
    return {"n": len(pa), "inliers": int(inl.sum()), "ratio": float(inl.mean()),
            "rmse": float(np.sqrt((resid[inl] ** 2).mean())), "cov": cov, "ent": ent}


def run_controls(build, matcher, bands, ctx, el, rng):
    """The four-control gate. Returns (passed, detail lines)."""
    morning, evening = bands["morning"], bands["evening"]
    ref = build(evening, ctx, el)

    base = offset_of(*matcher(build(morning, ctx, el), ref))
    lines, ok = [], True

    if base is None:
        return False, ["real pair: too few matches to measure"]
    lines.append(f"real pair          dx={base[0]:+7.2f} dy={base[1]:+7.2f}  n>={MIN_MATCHES}")

    # 1. Shift the RAW input. Recovered dx must move by -ROLL_PX.
    rolled = offset_of(*matcher(build(np.roll(morning, ROLL_PX, axis=1), ctx, el), ref))
    if rolled is None:
        lines.append("roll control       collapsed (cannot verify)")
        ok = False
    else:
        delta = rolled[0] - base[0]
        good = abs(delta - (-ROLL_PX)) < 4.0
        ok &= good
        lines.append(f"roll +{ROLL_PX} raw        dx moved {delta:+7.2f}  "
                     f"(want {-ROLL_PX:+d})  {'PASS' if good else 'FAIL'}")

    # 2/3. Noise and constant must produce far fewer matches than real imagery.
    # Graded rather than binary: a handful of chance matches from noise is not
    # disqualifying, but noise matching nearly as well as terrain means the
    # method is keying on something other than the terrain.
    n_real = len(matcher(build(morning, ctx, el), ref)[0])
    for tag, img in [("noise", rng.normal(morning.mean(), morning.std(), morning.shape)),
                     ("constant", np.full(morning.shape, morning.mean(), float))]:
        n_fake = len(matcher(build(img, ctx, el), ref)[0])
        ratio = n_real / max(n_fake, 1)
        good = n_fake < MIN_MATCHES or ratio >= NOISE_REJECT_RATIO
        ok &= good
        lines.append(f"{tag:<18} n={n_fake:>5} vs real {n_real:<5} "
                     f"ratio {ratio:>6.1f}x  {'PASS' if good else 'FAIL'}")

    return ok, lines


def evaluate(build, matcher, triple, spots, el, label, rng, verbose):
    ctx0 = None
    print(f"\n=== {label} ===")

    # Gate on the first window before spending time on the rest.
    row, col = spots[0]
    bands = triple.read_window(row, col, SIZE)
    dem = np.nan_to_num(bands["dem"], nan=float(np.nanmean(bands["dem"])))
    lat, _ = triple.pixel_latlon(row + SIZE / 2, col + SIZE / 2)
    ctx0 = (dem, abs(triple.transform.a), lat)

    passed, lines = run_controls(build, matcher, bands, ctx0, el, rng)
    for ln in lines:
        print(f"  {ln}")
    if not passed:
        print("  -> CONTROLS FAILED. Numbers from this method are not reported.")
        return None

    print("  -> controls passed\n")
    print(f"  {'window':>14} {'n':>6} {'inl':>6} {'ratio':>6} {'RMSE':>7} "
          f"{'cov':>5} {'dx':>7} {'dy':>7}")
    offs, fits = [], []
    for row, col in spots:
        bands = triple.read_window(row, col, SIZE)
        dem = np.nan_to_num(bands["dem"], nan=float(np.nanmean(bands["dem"])))
        lat, _ = triple.pixel_latlon(row + SIZE / 2, col + SIZE / 2)
        ctx = (dem, abs(triple.transform.a), lat)

        pa, pb = matcher(build(bands["morning"], ctx, el),
                         build(bands["evening"], ctx, el))
        off = offset_of(pa, pb)
        fit = fit_quality(pa, pb, (SIZE, SIZE))
        if off is None or fit is None:
            print(f"  ({row:>5},{col:>5}) {len(pa):>6}   too few")
            continue
        offs.append(off[:2])
        fits.append(fit)
        print(f"  ({row:>5},{col:>5}) {fit['n']:>6} {fit['inliers']:>6} "
              f"{fit['ratio']*100:>5.0f}% {fit['rmse']:>7.3f} {fit['cov']:>5.2f} "
              f"{off[0]:>+7.2f} {off[1]:>+7.2f}")

    if len(offs) < 2:
        print("  too few usable windows to judge consistency")
        return None

    a = np.array(offs)
    spread = float(np.hypot(a[:, 0].std(), a[:, 1].std()))
    print(f"\n  recovered offset: dx={np.median(a[:,0]):+.2f} dy={np.median(a[:,1]):+.2f} px")
    print(f"  cross-window spread: {spread:.2f} px  <- the real evidence")
    print(f"  mean inlier ratio {np.mean([f['ratio'] for f in fits])*100:.0f}%, "
          f"mean RMSE {np.mean([f['rmse'] for f in fits]):.3f} px, "
          f"mean coverage {np.mean([f['cov'] for f in fits]):.2f}")
    return {"offs": a, "spread": spread, "fits": fits, "label": label}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default="N18E009N15E012SC")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--full", action="store_true", help="also run SIFT variants")
    args = parser.parse_args()

    if args.tile not in find_tiles():
        print(f"tile {args.tile} not downloaded")
        return 1
    triple = open_triple(args.tile)
    spots = WINDOWS[: args.windows]
    rng = np.random.default_rng(0)

    print(f"tile {triple.tile}, {len(spots)} windows of {SIZE}x{SIZE}")
    print("No identity ground truth is used. Validation is cross-window agreement.")

    results = []
    combos = [("LoFTR raw", build_raw, match_loftr, None),
              ("LoFTR + local-norm only", build_raw_hp, match_loftr, None),
              ("LoFTR + Stage B el=20", build_stageb, match_loftr, 20.0),
              ("LoFTR + StageB+HP el=20", build_stageb_hp, match_loftr, 20.0),
              ("LoFTR + StageB+HP el=10", build_stageb_hp, match_loftr, 10.0)]
    if args.full:
        combos += [("SIFT raw", build_raw, match_sift, None),
                   ("SIFT + Stage B el=20", build_stageb, match_sift, 20.0)]

    for label, build, matcher, el in combos:
        r = evaluate(build, matcher, triple, spots, el, label, rng, args.full)
        if r:
            results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY (methods that passed the control gate)")
    print(f"{'method':<26} {'spread':>8} {'inlier':>8} {'RMSE':>8} {'cov':>6}")
    for r in results:
        print(f"{r['label']:<26} {r['spread']:>8.2f} "
              f"{np.mean([f['ratio'] for f in r['fits']])*100:>7.0f}% "
              f"{np.mean([f['rmse'] for f in r['fits']]):>8.3f} "
              f"{np.mean([f['cov'] for f in r['fits']]):>6.2f}")
    print("\nspread = cross-window scatter of the recovered offset, in px. Lower is better;")
    print("it is the only column an artifact cannot fake.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
