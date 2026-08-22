"""Viewpoint variation: fit the transform the geometry actually requires.

The problem statement names three challenges. Illumination and scale are
handled elsewhere; this is the third. Different camera positions shift, scale,
rotate and **perspective-distort** the same scene, and a similarity or affine
model cannot represent perspective at all -- it will absorb the error into a
worse fit rather than failing loudly.

Model selection is by held-out residual, not by in-sample fit. A homography has
8 degrees of freedom against affine's 4, so it *always* fits the training points
at least as well; picking on in-sample error would select it every time and
prove nothing. Splitting the matches and scoring on the unseen half is what
makes the comparison honest.

Test data is genuinely oblique, not synthesised: LROC NAC images of one site
from different orbits, selected through ODE's emission-angle index. The pair
used here spans 1.2 deg (near-nadir) to 58.9 deg emission -- a 57.8 deg
difference in viewing geometry, on real imagery.

    python scripts/viewpoint.py --self-check
    python scripts/viewpoint.py --nac
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from register import match_with_conf, refine_subpixel, uniformity
from remeasure import build_raw_hp
from triple_io import find_tiles, open_triple

ROOT = Path(__file__).resolve().parent.parent
NAC_DIR = ROOT / "data" / "raw" / "nac"
RANSAC_PX = 3.0
MIN_FOR_SPLIT = 16

# A more complex model is only accepted if it beats the simpler one by this
# fraction on held-out error. Without it the selector picks homography on
# orthorectified nadir pairs where no perspective exists, winning by 5-9% --
# noise, not geometry. Parsimony is the default; perspective must be earned.
COMPLEXITY_MARGIN = 0.15


MODELS = {
    "similarity": dict(dof=4),   # shift, rotate, uniform scale
    "affine": dict(dof=6),       # + shear, anisotropic scale
    "homography": dict(dof=8),   # + perspective
}


def fit_model(kind, pa, pb):
    """Fit one model class. Returns (matrix, inlier_mask) or None."""
    if kind == "similarity":
        m, inl = cv2.estimateAffinePartial2D(
            pa.astype(np.float32), pb.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    elif kind == "affine":
        m, inl = cv2.estimateAffine2D(
            pa.astype(np.float32), pb.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    elif kind == "homography":
        m, inl = cv2.findHomography(
            pa.astype(np.float32), pb.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    else:
        raise ValueError(kind)
    if m is None or inl is None:
        return None
    return m, inl.ravel().astype(bool)


def apply_model(kind, m, pts):
    if kind == "homography":
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), m)
        return p.reshape(-1, 2)
    return (m @ np.c_[pts, np.ones(len(pts))].T).T


def residual(kind, m, pa, pb):
    pred = apply_model(kind, m, pa)
    return np.hypot(pred[:, 0] - pb[:, 0], pred[:, 1] - pb[:, 1])


def select_model(pa, pb, seed=0):
    """Compare model classes on HELD-OUT residual.

    A higher-DOF model always wins in-sample, so in-sample comparison would
    select homography unconditionally and tell us nothing about whether
    perspective is actually present.
    """
    n = len(pa)
    if n < MIN_FOR_SPLIT:
        return None

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    half = n // 2
    train, test = idx[:half], idx[half:]

    out = {}
    for kind in MODELS:
        got = fit_model(kind, pa[train], pb[train])
        if got is None:
            continue
        m, _ = got
        r_test = residual(kind, m, pa[test], pb[test])
        # Median, not mean: a handful of bad matches should not decide which
        # geometry the scene has.
        out[kind] = {"held_out_median": float(np.median(r_test)),
                     "held_out_rmse": float(np.sqrt((r_test ** 2).mean())),
                     "dof": MODELS[kind]["dof"]}

        full = fit_model(kind, pa, pb)
        if full:
            mf, inl = full
            out[kind]["inlier_ratio"] = float(inl.mean())
            out[kind]["in_sample_rmse"] = float(
                np.sqrt((residual(kind, mf, pa, pb)[inl] ** 2).mean()))
            out[kind]["model"] = mf
            out[kind]["inliers"] = inl
    if not out:
        return None

    # Walk simplest to most complex, upgrading only on a decisive improvement.
    order = [k for k in MODELS if k in out]
    best = order[0]
    for kind in order[1:]:
        gain = (out[best]["held_out_median"] - out[kind]["held_out_median"])
        rel = gain / max(out[best]["held_out_median"], 1e-9)
        out[kind]["gain_over_simpler"] = float(rel)
        if rel >= COMPLEXITY_MARGIN:
            best = kind
    return {"models": out, "best": best}


def report(sel, label):
    print(f"\n{label}")
    if sel is None:
        print("  too few matches to split")
        return None
    print(f"  {'model':<12} {'dof':>4} {'held-out med':>13} {'held-out RMSE':>14} "
          f"{'in-sample':>10} {'inliers':>8} {'gain':>7}")
    for kind in MODELS:
        if kind not in sel["models"]:
            continue
        m = sel["models"][kind]
        mark = "  <- selected" if kind == sel["best"] else ""
        gain = m.get("gain_over_simpler")
        gtxt = f"{gain*100:>+6.0f}%" if gain is not None else "     -"
        print(f"  {kind:<12} {m['dof']:>4} {m['held_out_median']:>13.3f} "
              f"{m['held_out_rmse']:>14.3f} {m.get('in_sample_rmse', float('nan')):>10.3f} "
              f"{m.get('inlier_ratio', 0)*100:>7.0f}% {gtxt}{mark}")
    return sel["best"]


def match_and_select(src, ref, label, refine=True):
    a8, b8 = build_raw_hp(src, None, None), build_raw_hp(ref, None, None)
    pa, pb, conf = match_with_conf(a8, b8)
    if len(pa) == 0:
        print(f"\n{label}\n  no matches")
        return None
    if refine:
        pa, pb, kept, _ = refine_subpixel(a8, b8, pa, pb)
    if len(pa) < MIN_FOR_SPLIT:
        print(f"\n{label}\n  only {len(pa)} matches after refinement")
        return None
    sel = select_model(pa, pb)
    best = report(sel, f"{label}  ({len(pa)} matches)")
    if sel and best:
        inl = sel["models"][best]["inliers"]
        cov, ent = uniformity(pa[inl], src.shape)
        print(f"  coverage {cov:.2f}, entropy {ent:.2f}")
    return sel


def locate_overlap(a, b, k=16):
    """Find where two long NAC strips overlap, by sliding at low resolution.

    NAC CDR labels carry no lat/lon -- geolocation needs SPICE -- so the
    overlapping rows must be searched for rather than looked up. The shape of
    the match-count profile is the result: a real overlap produces a clear peak,
    while a flat profile means the best offset is indistinguishable from any
    other and the counts are noise.
    """
    A = cv2.resize(a, (a.shape[1] // k, a.shape[0] // k), interpolation=cv2.INTER_AREA)
    B = cv2.resize(b, (b.shape[1] // k, b.shape[0] // k), interpolation=cv2.INTER_AREA)
    w = (A.shape[1] // 8) * 8
    counts, best = [], (0, 0, 0)

    for ra in range(0, A.shape[0] - w, w):
        a8 = build_raw_hp(A[ra:ra + w, :w], None, None)
        for rb in range(0, B.shape[0] - w, w // 2):
            b8 = build_raw_hp(B[rb:rb + w, :w], None, None)
            n = len(match_with_conf(a8, b8)[0])
            counts.append(n)
            if n > best[0]:
                best = (n, ra * k, rb * k)
    return best, np.array(counts)


def nac_pair():
    """The genuinely oblique real pair: 1.2 deg vs 58.9 deg emission."""
    import rasterio

    files = sorted(NAC_DIR.glob("*.IMG"))
    if len(files) < 2:
        print(f"need two NAC .IMG files in {NAC_DIR}, found {len(files)}")
        return 1

    full = []
    for f in files:
        with rasterio.open(f) as s:
            print(f"  {f.name}: {s.width} x {s.height} {s.dtypes[0]}")
            full.append(s.read(1).astype(np.float32))

    print("")
    print("nadir (emission 1.2 deg) vs oblique (emission 58.9 deg), 57.8 deg apart")
    print("searching for the overlapping rows (NAC labels carry no geolocation)")
    (n, ra, rb), counts = locate_overlap(full[0], full[1])

    print("")
    print(f"  best: {n} matches at A row {ra}, B row {rb}")
    print(f"  match-count profile over all offsets: min {counts.min()}, "
          f"median {np.median(counts):.0f}, max {counts.max()}")
    peak = counts.max() / max(np.median(counts), 1)
    print(f"  peak / median = {peak:.2f}")

    # 2.0 is not a real peak: at that very offset, full-resolution matching
    # still yields 0 usable matches. A genuine overlap peaks far higher.
    if peak < 3.0:
        print("")
        print("  RESULT: no overlap detectable. The profile is flat, so the best")
        print("  offset is indistinguishable from every other one - these counts")
        print("  are noise. Matching FAILS across a 57.8 deg emission-angle gap.")
        print("  For scale: Kaguya pairs of the same terrain give 60-280 matches.")
        return 0

    side = min(1024, full[0].shape[1])
    match_and_select(full[0][ra:ra + side, :side], full[1][rb:rb + side, :side],
                     "NAC nadir vs oblique")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nac", action="store_true", help="run the oblique NAC pair")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.nac:
        return nac_pair()

    # Control: Kaguya morning/evening are both nadir orthorectified products, so
    # there is no perspective between them. Homography must NOT win here -- if it
    # does, the selector is just rewarding degrees of freedom.
    tiles = find_tiles()
    if not tiles:
        print("no Kaguya tiles downloaded")
        return 1
    triple = open_triple("N18E009N15E012SC")
    picked = []
    for row, col in [(2000, 2000), (5888, 5888), (8000, 3000)]:
        b = triple.read_window(row, col, 512)
        sel = match_and_select(b["morning"].astype(np.float32),
                               b["evening"].astype(np.float32),
                               f"Kaguya nadir-vs-nadir ({row},{col})")
        if sel:
            picked.append(sel["best"])

    print("\n" + "=" * 60)
    print(f"nadir-vs-nadir selections: {picked}")
    print("Both products are orthorectified nadir views, so no perspective exists")
    print("between them. A selector that picks homography here is rewarding")
    print("degrees of freedom rather than measuring geometry.")

    if args.self_check:
        assert picked, "no window produced a model selection"
        n_homog = picked.count("homography")
        assert n_homog <= len(picked) // 2, (
            f"homography selected in {n_homog}/{len(picked)} nadir pairs; "
            "held-out selection is not discriminating")
        print("\nself-check: selector does not over-pick homography on nadir pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
