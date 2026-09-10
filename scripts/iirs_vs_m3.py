"""Register Chandrayaan-2 IIRS against Chandrayaan-1 M3. Spectrometer to spectrometer.

THE HYPOTHESIS THIS TESTS
-------------------------
IIRS does not register against Kaguya: correlation +0.033, 44 matches, 6 inliers
(`iirs_vs_kaguya.py`). Three controls there rule out the projection, the strip
shape and the 11.49x scale ratio, leaving the instrument pairing itself -- an
imaging spectrometer against a visible framing camera at 85 m/px.

M3 tests exactly that. It is also an imaging spectrometer, ~140 m/px against
IIRS's 85.08, so this pairing is spectrometer-to-spectrometer at about **1.6x**
instead of spectrometer-to-camera at 11.5x. The prediction is falsifiable in
both directions:

* **It matches** -> the instrument-pairing explanation holds, and IIRS registers
  once given an appropriate reference.
* **It fails too** -> that explanation is wrong and the problem is IIRS itself,
  which has to be said plainly.

M3 is indexed by PDS ODE and needs no login, unlike everything Chandrayaan-2.
Fetch it with `python scripts/m3.py fetch --id M3G20090204T233457`.

FORMAT DIFFERENCES THAT MATTER
------------------------------
Nothing carries over from the IIRS reader. M3 global mode is 85 bands over
461-2976 nm, BIL interleave, with `data ignore value = -999.0` that poisons the
normalisation if it is not masked. Its LOC backplane is float64 with 3 bands
(lon, lat, radius) where IIRS's is float32 with 4.

`flip = 2` appears in the M3 reflectance header but NOT in its LOC or OBS
headers, so the cube may be stored mirrored relative to its own geolocation.
Rather than trust a convention, `--flip both` registers each orientation and
reports which one matches -- an ambiguity turned into a measurement.

GSD IS MEASURED, NOT ASSUMED
----------------------------
M3 global mode is nominally 140 m/px, but the label does not state a map scale,
and OHRC taught this project what a rounded or nominal figure costs: its label
says 0.26 m where its own geometry gives 0.3060 x 0.3225 (BUGS.md BUG-020,
ROADMAP 1.1). So the scale here is computed from the LOC backplane by arc length
between adjacent pixel centres.

    python scripts/iirs_vs_m3.py
    python scripts/iirs_vs_m3.py --flip both --no-gate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sandhi import controls, pipeline  # noqa: E402

M3_DIR = ROOT / "data" / "raw" / "m3"
MOON_RADIUS_M = 1737400.0
IIRS_GSD_M = 85.08
M3_IGNORE = -999.0


def m3_paths(obs: str | None = None):
    dirs = sorted(p for p in M3_DIR.glob("M3G*") if p.is_dir())
    if not dirs:
        raise SystemExit("no M3 product. python scripts/m3.py fetch --id M3G20090204T233457")
    d = next((x for x in dirs if obs and obs in x.name), dirs[0])
    rfl = next(d.glob("*_RFL.IMG"))
    loc = next(d.glob("*_LOC.IMG"))
    return d, rfl, loc


def m3_wavelengths(rfl: Path) -> np.ndarray:
    t = rfl.with_suffix(".HDR").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", t, re.S)
    return np.array([float(x) for x in m.group(1).replace("\n", " ").split(",")
                     if x.strip()])


def ground_gsd(lat: np.ndarray, lon: np.ndarray) -> float:
    """Metres per pixel across track, by arc length between adjacent centres."""
    p1, p2 = np.radians(lat[:, :-1]), np.radians(lat[:, 1:])
    dl = np.radians(lon[:, 1:] - lon[:, :-1])
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    d = 2 * MOON_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return float(np.median(d))


def project(band, lat, lon, gsd_m, ignore=None, grid=None):
    """Resample a swath onto a north-up lat/lon grid.

    `grid` is the load-bearing argument. Projecting each swath onto its OWN
    bounding box puts the two images in different frames with different origins,
    so they never overlap however good the matcher is -- and the failure looks
    like "no correspondence" rather than "not aligned". Both sides must land on
    ONE grid, which is what the OHRC and TMC-2 paths get for free by reading the
    reference over the source's bounds.
    """
    from ohrc_project import apply_inverse, fit_inverse

    h, w = band.shape
    ss, ll = np.meshgrid(np.arange(w), np.arange(h))
    coef, resid, degree = fit_inverse(lon.ravel(), lat.ravel(),
                                      ss.ravel().astype(float),
                                      ll.ravel().astype(float))
    dpp = np.degrees(gsd_m / MOON_RADIUS_M)
    if grid is None:
        west, east = float(lon.min()), float(lon.max())
        south, north = float(lat.min()), float(lat.max())
    else:
        west, south, east, north = grid
    width = max(int(round((east - west) / dpp)), 8)
    height = max(int(round((north - south) / dpp)), 8)

    cols = (np.arange(width) + 0.5) * dpp + west
    rows = north - (np.arange(height) + 0.5) * dpp
    LON, LAT = np.meshgrid(cols, rows)
    px, sc = apply_inverse(coef, LON.ravel(), LAT.ravel())
    inside = ((px >= 0) & (px <= w - 1) & (sc >= 0) & (sc <= h - 1))
    pxi = np.clip(np.round(px).astype(int), 0, w - 1)
    sci = np.clip(np.round(sc).astype(int), 0, h - 1)
    out = band[sci, pxi].reshape(height, width).astype(np.float32)
    valid = inside.reshape(height, width)
    if ignore is not None:
        valid &= (out != ignore)
    out[~valid] = 0.0
    return out, BoundingBox(west, south, east, north), resid, degree, valid


def read_m3(rfl, loc, nm, lat_range, flip):
    with rasterio.open(loc) as s:
        lon = s.read(1)
        lat = s.read(2)
    lo_lat, hi_lat = lat_range
    mid = lon.shape[1] // 2
    keep = np.flatnonzero((lat[:, mid] >= lo_lat) & (lat[:, mid] <= hi_lat))
    if keep.size < 64:
        raise SystemExit(f"M3 has only {keep.size} lines in {lat_range}")
    sub = slice(int(keep.min()), int(keep.max()) + 1)

    wl = m3_wavelengths(rfl)
    idx = int(np.argmin(np.abs(wl - nm)))
    with rasterio.open(rfl) as s:
        band = s.read(idx + 1, window=Window(0, sub.start, s.width,
                                             sub.stop - sub.start))
    band = band.astype(np.float32)
    if flip:
        # `flip = 2` in the reflectance header, absent from LOC/OBS.
        band = band[:, ::-1]
    return band, lat[sub], lon[sub], wl[idx], idx + 1


def read_iirs(nm, lat_range):
    from iirs_vs_kaguya import CUBES, overlap_lines, product, vsi, wavelengths
    z, stem = product()
    wl = wavelengths(z, stem)
    idx = int(np.argmin(np.abs(wl - nm)))
    lo_lat, hi_lat = lat_range
    sub, lat, lon = overlap_lines(z, stem, (-180.0, 180.0, lo_lat, hi_lat))
    with rasterio.open(vsi(z, stem, CUBES["reflectance"])) as s:
        band = s.read(idx + 1, window=Window(0, sub.start, s.width,
                                             sub.stop - sub.start))
    return (np.nan_to_num(band.astype(np.float32)), lat[sub], lon[sub],
            wl[idx], idx + 1)



def m3_vs_kaguya(obs, nm, tile_lat, gate=True) -> dict:
    """Register M3 against Kaguya -- the leg that tells us whose geometry is wrong.

    IIRS fails against Kaguya AND against M3, while matching itself at 100%.
    Two readings survive that: IIRS's geolocation is wrong, or spectrometer
    imagery cannot be matched to anything external by this pipeline. They are
    distinguished by asking whether a DIFFERENT spectrometer, with independent
    geolocation, registers against the same trusted reference.

    * M3 matches Kaguya -> spectrometers are matchable, and IIRS is the outlier.
    * M3 fails too -> the pipeline cannot bridge spectrometer to camera, and the
      IIRS result is not about IIRS at all.
    """
    from iirs_vs_kaguya import kaguya_over

    d, rfl, loc = m3_paths(obs)
    band, lat, lon, wl_nm, bi = read_m3(rfl, loc, nm, tile_lat, flip=False)
    gsd = ground_gsd(lat, lon)
    print(f"M3 {d.name} band {bi} ({wl_nm:.1f} nm) {band.shape}, "
          f"measured GSD {gsd:.2f} m/px")

    src, bounds, resid, deg, valid = project(band, lat, lon, gsd,
                                             ignore=M3_IGNORE)
    print(f"  projected {src.shape}, degree-{deg} fit residual {resid:.2f} px, "
          f"{100*valid.mean():.0f}% covered")
    ref, tile = kaguya_over(bounds)
    if ref is None:
        print("  no Kaguya product covers this box")
        return {"registered": False, "reason": "no reference"}
    print(f"  Kaguya evening from {tile}: {ref.shape}  "
          f"(ratio {gsd/7.403:.2f}x)")

    r = pipeline.register(src, ref, src_gsd=gsd, ref_gsd=7.403, gsd_m=gsd)
    if r is None:
        print("  NO FIT")
        return {"registered": False, "m3_gsd": round(gsd, 2)}
    m = r["metrics"]
    out = {"pair": "M3 vs Kaguya", "registered": True,
           "wavelength_nm": float(wl_nm), "m3_gsd": round(gsd, 2),
           "match_count": m["match_count"],
           "inlier_ratio": round(m["inlier_ratio"], 4),
           "rmse_px": None if m["rmse_px"] is None or np.isnan(m["rmse_px"])
                      else round(m["rmse_px"], 4),
           "coverage": round(m["coverage"], 4),
           "entropy": round(m["entropy"], 4), "model": r["fit"]["kind"]}
    for k in ("match_count", "inlier_ratio", "rmse_px", "coverage", "entropy",
              "model"):
        print(f"  {k:<14} {out[k]}")
    if gate:
        g = controls.run(controls.default_match_fn(), r["src"], r["ref"])
        out["gate"] = bool(g.passed)
        out["gate_checks"] = {c["name"]: bool(c["passed"]) for c in g.checks}
        print("  CONTROL GATE")
        for c in g.checks:
            print(f"    {c['name']:<18} {'PASS' if c['passed'] else 'FAIL'}  "
                  f"{c.get('detail','')}")
        print(f"    -> {'PASS' if g.passed else 'FAIL'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavelength", type=float, default=1504.0)
    ap.add_argument("--flip", default="both", choices=["yes", "no", "both"])
    ap.add_argument("--obs", default=None, help="M3 observation id")
    ap.add_argument("--vs-kaguya", action="store_true",
                    help="register M3 against Kaguya instead of against IIRS")
    ap.add_argument("--tile-lat", nargs=2, type=float, default=[15.0, 18.0],
                    metavar=("S", "N"), help="latitude band for --vs-kaguya")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--out", default="outputs/iirs_vs_m3.json")
    args = ap.parse_args()

    if args.vs_kaguya:
        row = m3_vs_kaguya(args.obs, args.wavelength, tuple(args.tile_lat),
                           gate=not args.no_gate)
        out = ROOT / "outputs" / "m3_vs_kaguya.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
        return 0

    d, rfl, loc = m3_paths(args.obs)
    print(f"M3  {d.name}")

    # Where do the two strips share latitude?
    with rasterio.open(loc) as s:
        m3_lat = s.read(2)
    from iirs_vs_kaguya import product, vsi
    z, stem = product()
    with rasterio.open(vsi(z, stem, "_d_loc_d18_ard.img")) as s:
        ii_lat = s.read(2)
    lo = max(float(m3_lat.min()), float(ii_lat.min()))
    hi = min(float(m3_lat.max()), float(ii_lat.max()))
    print(f"  M3 lat {m3_lat.min():.2f}..{m3_lat.max():.2f}   "
          f"IIRS lat {ii_lat.min():.2f}..{ii_lat.max():.2f}")
    print(f"  shared latitude band: {lo:.2f}..{hi:.2f} ({hi-lo:.2f} deg)")
    if hi - lo < 0.5:
        raise SystemExit("strips barely share latitude")

    rows = []
    flips = {"yes": [True], "no": [False], "both": [False, True]}[args.flip]
    for flip in flips:
        m3b, m3lat, m3lon, m3nm, m3band = read_m3(rfl, loc, args.wavelength,
                                                  (lo, hi), flip)
        iib, iilat, iilon, iinm, iiband = read_iirs(args.wavelength, (lo, hi))
        m3_gsd = ground_gsd(m3lat, m3lon)
        ii_gsd = ground_gsd(iilat, iilon)
        print(f"\n[flip={'yes' if flip else 'no'}] "
              f"M3 band {m3band} {m3nm:.1f} nm {m3b.shape}, "
              f"IIRS band {iiband} {iinm:.1f} nm {iib.shape}")
        print(f"  measured GSD: M3 {m3_gsd:.2f} m/px, IIRS {ii_gsd:.2f} m/px "
              f"-> ratio {m3_gsd/ii_gsd:.2f}x")

        # ONE grid for both, over the ground they actually share, at the
        # coarser of the two scales.
        grid = (max(float(iilon.min()), float(m3lon.min())),
                max(float(iilat.min()), float(m3lat.min())),
                min(float(iilon.max()), float(m3lon.max())),
                min(float(iilat.max()), float(m3lat.max())))
        if grid[2] <= grid[0] or grid[3] <= grid[1]:
            print(f"  swaths share latitude but not ground: {grid}")
            rows.append({"flip": flip, "registered": False, "reason": "no overlap"})
            continue
        common = max(ii_gsd, m3_gsd)
        print(f"  common grid lon {grid[0]:.3f}..{grid[2]:.3f} "
              f"lat {grid[1]:.3f}..{grid[3]:.3f} at {common:.1f} m/px")

        src, sb, sres, _, svalid = project(iib, iilat, iilon, common, grid=grid)
        ref, rb, rres, _, rvalid = project(m3b, m3lat, m3lon, common,
                                           ignore=M3_IGNORE, grid=grid)
        both = svalid & rvalid
        print(f"  projected IIRS {src.shape} (fit {sres:.2f} px), "
              f"M3 {ref.shape} (fit {rres:.2f} px), "
              f"overlapping pixels {100*both.mean():.1f}%")
        if both.mean() < 0.02:
            print("  the two swaths barely cover the same ground")

        # Already on one grid at one scale, so no GSD arguments here.
        r = pipeline.register(src, ref, gsd_m=common)
        if r is None:
            print("  NO FIT")
            rows.append({"flip": flip, "registered": False,
                         "m3_gsd": round(m3_gsd, 2), "iirs_gsd": round(ii_gsd, 2)})
            continue
        m = r["metrics"]
        row = {"flip": flip, "registered": True,
               "wavelength_nm": float(m3nm),
               "m3_gsd": round(m3_gsd, 2), "iirs_gsd": round(ii_gsd, 2),
               "scale_ratio": round(m3_gsd / ii_gsd, 3),
               "match_count": m["match_count"],
               "inlier_ratio": round(m["inlier_ratio"], 4),
               "rmse_px": None if m["rmse_px"] is None or np.isnan(m["rmse_px"])
                          else round(m["rmse_px"], 4),
               "coverage": round(m["coverage"], 4),
               "entropy": round(m["entropy"], 4), "model": r["fit"]["kind"]}
        for k in ("match_count", "inlier_ratio", "rmse_px", "coverage",
                  "entropy", "model"):
            print(f"  {k:<14} {row[k]}")
        if not args.no_gate:
            g = controls.run(controls.default_match_fn(), r["src"], r["ref"])
            row["gate"] = bool(g.passed)
            row["gate_checks"] = {c["name"]: bool(c["passed"]) for c in g.checks}
            print("  CONTROL GATE")
            for c in g.checks:
                print(f"    {c['name']:<18} {'PASS' if c['passed'] else 'FAIL'}  "
                      f"{c.get('detail','')}")
            print(f"    -> {'PASS' if g.passed else 'FAIL'}")
        rows.append(row)

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
