"""Where does the 3.4 km OHRC offset come from -- Chandrayaan-2, or us?

ROADMAP 1.1. The registered OHRC product sits ~437 px (3.4 km) from where the
Kaguya frame puts it. Two causes produce that signature and the project has never
separated them, so the number has always been reported as "the offset between our
projected OHRC and the Kaguya frame" rather than as Chandrayaan-2 geolocation
error. This measures which it is.

THE DISCRIMINATOR
-----------------
The offset is not a pure translation: four independent chunks agree on a scale
of 1.0165-1.0189 against Kaguya. A pointing or ephemeris bias shifts an image;
it does not stretch it. So the scale carries the diagnosis, and its *axis* is
what separates the two hypotheses:

* **Along-track only.** A pushbroom builds an image one line at a time, so its
  along-track ground sampling is set by spacecraft ground speed divided by line
  rate. Get that velocity wrong by 1.7% and the along-track spacing is wrong by
  1.7% -- a scale error on one axis, plus an offset that accumulates down the
  strip. That is a Chandrayaan-2 geolocation fault, in its own geometry file.
* **Isotropic.** Both axes scaled equally points at the map projection: a radius,
  a latitude convention, or a degrees-per-metre conversion -- our arithmetic.

Two supporting facts are already measured and neither is re-derived here:

1. `ohrc_project.py` reproduces the CSV geolocation to 0.152 m, so our
   projection faithfully implements what Chandrayaan-2 says. Whatever error the
   CSV carries, we inherit rather than create.
2. TMC-2 -- a different Chandrayaan-2 instrument, shipped as a georeferenced
   GeoTIFF that never touches our polynomial -- registers against Kaguya to
   ~160 m. So the Kaguya frame, the matcher, the normalisation and the RANSAC
   are not the source of a 3.4 km discrepancy.

    python scripts/offset_origin.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sandhi import models, normalize  # noqa: E402
from sandhi import matching  # noqa: E402

MOON_RADIUS_M = 1737400.0


def decompose(A: np.ndarray) -> dict:
    """Split a 2x3 affine into scale along each axis, shear and rotation.

    Uses the QR-style decomposition A = R(theta) * [[sx, sh], [0, sy]], so `sx`
    is the scale along image x (cross-track for this strip) and `sy` along image
    y (along-track). Reading the two singular values instead would mix the axes
    and destroy exactly the distinction being measured.
    """
    M = np.asarray(A, float)[:, :2]
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    sx = np.hypot(a, c)
    theta = np.arctan2(c, a)
    shear = (a * b + c * d) / (sx ** 2)
    sy = np.hypot(b - shear * a, d - shear * c)
    return {"scale_x": float(sx), "scale_y": float(sy),
            "rotation_deg": float(np.degrees(theta)), "shear": float(shear),
            "anisotropy": float(sy / sx)}


def chunks(ohrc, kaguya, n=4, rows=512):
    """Independent along-track chunks, so the trend can be seen, not assumed."""
    h = min(ohrc.shape[0], kaguya.shape[0])
    step = (h - rows) // max(n - 1, 1)
    for i in range(n):
        r0 = i * step
        yield i, r0, ohrc[r0:r0 + rows], kaguya[r0:r0 + rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--out", default="outputs/offset_origin.json")
    args = ap.parse_args()

    from ohrc_vs_kaguya import kaguya_window, load_projected
    from triple_io import open_triple

    path = ROOT / "data" / "interim" / "ohrc_7m.tif"
    if not path.exists():
        print("need data/interim/ohrc_7m.tif -- run scripts/ohrc_project.py --gsd 7.403")
        return 1
    ohrc, _, bounds = load_projected(path)
    t = open_triple("N03E021N00E024SC")
    ref = kaguya_window(t, bounds)["evening"].astype(np.float32)
    w = min(ohrc.shape[1], ref.shape[1])
    ohrc, ref = ohrc[:, :w].astype(np.float32), ref[:, :w]
    print(f"projected OHRC {ohrc.shape} vs Kaguya {ref.shape}, 7.403 m/px")

    # Our projection can only be blamed for error it actually introduces, so
    # measure how faithfully the polynomial reproduces Chandrayaan-2's own
    # geolocation. A residual far below the discrepancy under investigation
    # means whatever the CSV carries, we inherit rather than create.
    from ohrc_project import fit_inverse
    from ch2_io import open_ohrc
    o = open_ohrc()
    _, resid_px, deg = fit_inverse(o.lon, o.lat, o.pixel, o.scan)
    cross_gsd = 0.3060                    # measured from the CSV, not the label
    print(f"projection self-check: degree-{deg} fit reproduces the CH-2 "
          f"geolocation to {resid_px:.3f} px ({resid_px*cross_gsd:.3f} m)")
    print(f"  the discrepancy under investigation is ~437 px at 7.403 m/px "
          f"(~3.4 km), which is {437*7.403/max(resid_px*cross_gsd, 1e-9):.0f}x larger\n")

    rows_out = []
    for i, r0, a, b in chunks(ohrc, ref, args.chunks, args.rows):
        a8, b8 = normalize.prepare(a), normalize.prepare(b)
        pa, pb, _, _, _ = matching.match_aligned(a8, b8)
        f = models.fit(pa, pb, kind="affine")
        if f is None or f["inliers"].sum() < 20:
            print(f"  chunk {i} (row {r0}): too few matches")
            continue
        d = decompose(f["model"])
        d |= {"chunk": i, "row0": int(r0), "matches": int(len(pa)),
              "inliers": int(f["inliers"].sum()),
              "dx": float(f["model"][0, 2]), "dy": float(f["model"][1, 2])}
        rows_out.append(d)
        print(f"  chunk {i} (row {r0:>5}): n={d['matches']:>5}  "
              f"dx={d['dx']:+8.2f} dy={d['dy']:+8.2f}  "
              f"scale_x={d['scale_x']:.4f} scale_y={d['scale_y']:.4f}  "
              f"aniso={d['anisotropy']:.4f}  rot={d['rotation_deg']:+.3f}")

    if len(rows_out) < 2:
        print("not enough chunks to conclude")
        return 1

    sx = np.array([r["scale_x"] for r in rows_out])
    sy = np.array([r["scale_y"] for r in rows_out])
    dy = np.array([r["dy"] for r in rows_out])
    r0 = np.array([r["row0"] for r in rows_out], float)

    print("\n" + "=" * 72)
    print(f"  cross-track scale (x)  mean {sx.mean():.4f}  sd {sx.std():.4f}")
    print(f"  along-track scale (y)  mean {sy.mean():.4f}  sd {sy.std():.4f}")
    print(f"  anisotropy (y/x)       {sy.mean()/sx.mean():.4f}")

    # Does the offset accumulate down the strip? A velocity error must.
    slope = np.polyfit(r0, dy, 1)[0] if len(r0) > 1 else float("nan")
    print(f"  dy drift along strip   {slope*1000:+.3f} px per 1000 rows")

    # Significance, not raw magnitude. The two axes are measured with very
    # different precision here (cross-track sd 0.003, along-track 0.018), so
    # comparing |sx-1| against |sy-1| directly called a 5.9-sigma cross-track
    # error and a 0.5-sigma along-track one "isotropic". Divide by the scatter.
    def sigma(vals):
        sd = vals.std(ddof=1) / np.sqrt(len(vals))
        return abs(vals.mean() - 1.0) / sd if sd > 0 else float("inf")

    zx, zy = sigma(sx), sigma(sy)
    print(f"  significance           cross-track {zx:.1f} sigma, "
          f"along-track {zy:.1f} sigma")

    print("\n  VERDICT")
    if zx < 3 and zy < 3:
        verdict = ("No significant scale on either axis. The offset is a pure "
                   "translation, consistent with a pointing/ephemeris bias.")
    elif zx >= 3 and zy < 3:
        verdict = (f"Scale error is CROSS-TRACK ONLY: {sx.mean():.4f} at "
                   f"{zx:.1f} sigma, while along-track is indistinguishable from "
                   f"1.0 ({sy.mean():.4f}, {zy:.1f} sigma). A map projection error "
                   "-- radius, latitude convention, degrees-per-metre -- scales "
                   "both axes together and cannot produce this. A cross-track-only "
                   "scale is a swath-width term (altitude or field of view) in "
                   "Chandrayaan-2's own geometry.")
    elif zy >= 3 and zx < 3:
        verdict = (f"Scale error is ALONG-TRACK ONLY: {sy.mean():.4f} at "
                   f"{zy:.1f} sigma. That is a ground-velocity or line-rate term "
                   "in the CH-2 geometry, again not something our projection can "
                   "produce on one axis.")
    elif abs(sx.mean() - sy.mean()) < 0.005:
        verdict = ("Scale error is ISOTROPIC and significant on both axes. That "
                   "points at the map projection -- body radius, latitude "
                   "convention, or the degrees-per-metre conversion -- i.e. OUR "
                   "error, not Chandrayaan-2's.")
    else:
        verdict = (f"Both axes scale significantly but by different amounts "
                   f"({sx.mean():.4f} vs {sy.mean():.4f}). Anisotropic, so not a "
                   "simple projection constant; the CH-2 geometry is the likelier "
                   "source but this does not isolate a single term.")
    print("  " + verdict)
    print("=" * 72)

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"chunks": rows_out, "verdict": verdict,
                               "scale_x_mean": float(sx.mean()),
                               "scale_y_mean": float(sy.mean()),
                               "dy_drift_px_per_1000_rows": float(slope * 1000)},
                              indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
