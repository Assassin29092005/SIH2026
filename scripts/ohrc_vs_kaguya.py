"""Register a real Chandrayaan-2 OHRC strip against a lunar reference image.

This is the actual task the problem statement describes, on real data, end to
end: a Chandrayaan-2 optical image matched to an independent lunar reference,
across sensor, illumination and a ~28x native scale ratio.

It differs from every earlier experiment in one crucial way: **there is no
identity ground truth.** The Kaguya triple gave exact correspondence for free
because its three products share one map grid. OHRC and Kaguya are independent
missions with independent geometry, so correctness has to be judged the way the
problem statement asks -- inlier count, inlier ratio and RMSE about a fitted
model -- rather than against a known answer.

That turns the residual into a measurement rather than a score: the systematic
part of the displacement is Chandrayaan-2's geolocation offset relative to the
Kaguya reference frame, and the scatter about it is our registration precision.

    python scripts/ohrc_project.py --gsd 7.403 --out data/interim/ohrc_7m.tif
    python scripts/ohrc_vs_kaguya.py
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from ablation import match_uniformity
from dense_match import match_loftr
from remeasure import build_raw_hp, offset_of
from triple_io import find_tiles, open_triple

ROOT = Path(__file__).resolve().parent.parent
MOON_RADIUS_M = 1737400.0
RANSAC_PX = 3.0


def load_projected(path: Path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.bounds


def kaguya_window(triple, bounds):
    """Read the Kaguya triple over a lon/lat bounding box, at native GSD."""
    out = {}
    for kind, label in triple.paths.items():
        with rasterio.open(label) as src:
            # Kaguya transforms are in projected metres; convert the lon/lat box.
            r = MOON_RADIUS_M
            win = from_bounds(
                np.radians(bounds.left) * r, np.radians(bounds.bottom) * r,
                np.radians(bounds.right) * r, np.radians(bounds.top) * r,
                transform=src.transform,
            )
            band = src.read(1, window=win, boundless=True, fill_value=0)
            if kind == "dem":
                band = band.astype(np.float32)
                band[band == -32768] = np.nan
            out[kind] = band
    return out


def match_pair(source, reference, dem=None, gsd_m=None, lat=None):
    """Match with the validated DEM-free normalisation.

    Stage-B division is deliberately NOT used: it failed the control gate,
    passing noise through more readily than terrain (BUGS.md BUG-011). Local
    contrast normalisation uses no DEM, so it cannot inject shared structure,
    and it is also the better performer.
    """
    return match_loftr(build_raw_hp(source, None, None),
                       build_raw_hp(reference, None, None))


def controls(source, reference, rng, roll_px=15):
    """The four-control gate, on the real OHRC/Kaguya pair."""
    base = offset_of(*match_pair(source, reference))
    lines, ok = [], True
    if base is None:
        return False, ["real pair: too few matches"]
    n_real = len(match_pair(source, reference)[0])
    lines.append(f"real pair       dx={base[0]:+7.2f} dy={base[1]:+7.2f}  n={n_real}")

    rolled = offset_of(*match_pair(np.roll(source, roll_px, axis=1), reference))
    if rolled is None:
        lines.append("roll control    collapsed  FAIL")
        ok = False
    else:
        delta = rolled[0] - base[0]
        good = abs(delta + roll_px) < 4.0
        ok &= good
        lines.append(f"roll +{roll_px} raw     dx moved {delta:+7.2f} (want {-roll_px:+d})"
                     f"  {'PASS' if good else 'FAIL'}")

    for tag, img in [("noise", rng.normal(source[source > 0].mean(),
                                          source[source > 0].std(), source.shape)),
                     ("constant", np.full(source.shape,
                                          float(source[source > 0].mean())))]:
        n_fake = len(match_pair(img.astype(np.float32), reference)[0])
        ratio = n_real / max(n_fake, 1)
        good = n_fake < 8 or ratio >= 5.0
        ok &= good
        lines.append(f"{tag:<15} n={n_fake:>5} ratio {ratio:>6.1f}x  "
                     f"{'PASS' if good else 'FAIL'}")
    return ok, lines


def report(pa, pb, shape, gsd_m, label) -> dict:
    """Fit a model and report the metrics the problem statement asks for."""
    print(f"\n{label}")
    if len(pa) < 8:
        print(f"  only {len(pa)} matches - too few to fit a model")
        return {}

    dx, dy = pb[:, 0] - pa[:, 0], pb[:, 1] - pa[:, 1]
    print(f"  matches: {len(pa)}")
    print(f"  median displacement: dx={np.median(dx):+.2f} dy={np.median(dy):+.2f} px "
          f"({np.hypot(np.median(dx), np.median(dy))*gsd_m:.1f} m on the ground)")

    # Affine is the right model class here: same projection and GSD, so any
    # real transform is a small shift plus residual geolocation distortion.
    model, inliers = cv2.estimateAffinePartial2D(
        pa.astype(np.float32), pb.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX,
    )
    if model is None:
        print("  RANSAC failed to fit a model")
        return {}

    inliers = inliers.ravel().astype(bool)
    n_in = int(inliers.sum())
    pred = (model @ np.c_[pa, np.ones(len(pa))].T).T
    resid = np.hypot(pred[:, 0] - pb[:, 0], pred[:, 1] - pb[:, 1])
    rmse = float(np.sqrt((resid[inliers] ** 2).mean())) if n_in else float("nan")

    scale = float(np.hypot(model[0, 0], model[0, 1]))
    rot = float(np.degrees(np.arctan2(model[0, 1], model[0, 0])))
    cov, ent = match_uniformity(pa[inliers], shape)

    print(f"  inliers: {n_in}/{len(pa)} ({100*n_in/len(pa):.1f}%)")
    print(f"  RMSE about the model: {rmse:.3f} px = {rmse*gsd_m:.2f} m")
    print(f"  fitted scale {scale:.4f}, rotation {rot:+.3f} deg")
    print(f"  shift: {model[0,2]:+.2f}, {model[1,2]:+.2f} px "
          f"= {np.hypot(model[0,2], model[1,2])*gsd_m:.1f} m")
    print(f"  uniformity: coverage {cov:.2f}, entropy {ent:.2f}")
    return {"n": len(pa), "inliers": n_in, "rmse": rmse, "scale": scale,
            "rot": rot, "cov": cov}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projected", default="data/interim/ohrc_7m.tif")
    parser.add_argument("--gsd", type=float, default=7.403)
    parser.add_argument("--rows", type=int, default=1024,
                        help="rows of the strip to use per chunk")
    args = parser.parse_args()

    path = ROOT / args.projected
    if not path.exists():
        print(f"missing {path}")
        print("run: python scripts/ohrc_project.py --gsd 7.403 --out data/interim/ohrc_7m.tif")
        return 1

    ohrc, transform, bounds = load_projected(path)
    print(f"OHRC projected: {ohrc.shape[1]} x {ohrc.shape[0]} at {args.gsd} m/px")
    print(f"  lon {bounds.left:.4f}..{bounds.right:.4f}  lat {bounds.bottom:.4f}..{bounds.top:.4f}")

    tiles = find_tiles()
    triple = None
    for name in tiles:
        t = open_triple(name)
        with rasterio.open(t.paths["morning"]) as src:
            r = MOON_RADIUS_M
            lon0 = np.degrees(src.bounds.left / r)
            lon1 = np.degrees(src.bounds.right / r)
            lat0 = np.degrees(src.bounds.bottom / r)
            lat1 = np.degrees(src.bounds.top / r)
        if lon0 <= bounds.left and lon1 >= bounds.right and lat0 <= bounds.bottom:
            triple = t
            break
    if triple is None:
        print("no downloaded Kaguya tile covers this strip")
        print("run: python scripts/kaguya.py fetch --bbox 23.3 23.6 0.2 1.1")
        return 1

    print(f"reference tile: {triple.tile}")
    bands = kaguya_window(triple, bounds)
    print(f"  Kaguya window: {bands['morning'].shape}")

    h = min(ohrc.shape[0], bands["morning"].shape[0])
    w = min(ohrc.shape[1], bands["morning"].shape[1])
    lat = (bounds.bottom + bounds.top) / 2

    # The strip is long and thin; process it in chunks so LoFTR sees roughly
    # square inputs, which is what it was trained on.
    rng = np.random.default_rng(0)
    # Gate before reporting anything.
    r0 = slice(1024, 2048)
    ok, lines = controls(ohrc[r0, :w].astype(np.float32),
                         bands["morning"][r0, :w].astype(np.float32), rng)
    print("")
    print("control gate:")
    for ln in lines:
        print(f"  {ln}")
    if not ok:
        print("  -> CONTROLS FAILED. No results reported.")
        return 1
    print("  -> controls passed")

    results = []
    for row0 in range(0, h - args.rows + 1, args.rows):
        sl = slice(row0, row0 + args.rows)
        src_chunk = ohrc[sl, :w].astype(np.float32)
        if (src_chunk > 0).mean() < 0.5:
            continue
        dem = bands["dem"][sl, :w]
        dem = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
        for ref_name in ("morning", "evening"):
            ref_chunk = bands[ref_name][sl, :w].astype(np.float32)
            pa, pb = match_pair(src_chunk, ref_chunk)
            r = report(pa, pb, (args.rows, w),
                       args.gsd, f"rows {row0}-{row0+args.rows} vs Kaguya {ref_name}")
            if r:
                results.append(r)

    if results:
        print("\n" + "=" * 62)
        tot_in = sum(r["inliers"] for r in results)
        tot_n = sum(r["n"] for r in results)
        print(f"OHRC vs Kaguya, {len(results)} chunks")
        print(f"  inliers {tot_in}/{tot_n} ({100*tot_in/max(tot_n,1):.1f}%)")
        print(f"  mean RMSE {np.mean([r['rmse'] for r in results]):.3f} px "
              f"= {np.mean([r['rmse'] for r in results])*args.gsd:.2f} m")
        print(f"  mean coverage {np.mean([r['cov'] for r in results]):.2f}")
    else:
        print("\nno chunk produced a usable model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
