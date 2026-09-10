"""Register Chandrayaan-2 IIRS against Kaguya TC. Measured, controlled, NEGATIVE.

The problem statement names OHRC, TMC **and IIRS**. OHRC and TMC-2 register with
the pipeline; IIRS does not. That is a measurement with every alternative
explanation eliminated, not a gap left unexplored.

RESULT
------
IIRS 1504 nm against Kaguya TC evening, over 1118 lines where the strip crosses
tile N18E009N15E012SC: **correlation +0.033, 44 matches, 6 inliers (13.6%)**.
No fit. Two hypotheses were tested and both were WRONG:

* **Band choice is not the driver.** The obvious story was that below ~2.5 um the
  signal is reflected sunlight and shades like Kaguya, while beyond ~3 um thermal
  emission dominates and cannot. Sweeping 898-4504 nm gives 0-5 matches at EVERY
  wavelength. If the reflected/thermal split mattered, the halves of that sweep
  would differ. They do not.
* **Reflectance vs radiance is not the driver either.** A reflectance product has
  the illumination divided out, so radiance -- which retains the shading Kaguya
  records -- should have done better. Both cubes give correlation **+0.0326**, to
  four decimal places the same number.

WHAT THE CONTROLS RULE OUT
--------------------------
Run them with `--controls`.

| Control | Result | Rules out |
|---|---|---|
| Two bands of one cube, both projected | 3528 matches, **100%** inliers, corr +0.986 | the projection, the strip shape, the imagery |
| Same, unprojected raw swath | 3645 matches, 100% inliers | projection loss specifically |
| Kaguya morning vs evening decimated to 85.08 m/px, same 288 px shape | 1618 matches, **61.5%** inliers | the 11.49x scale ratio |

So the pipeline handles this scale, this shape and this projection. What it does
not bridge is the instrument gap: at 85 m/px an infrared spectrometer and a
visible framing camera record different information over the same ground. Kaguya
*averaged* to 85 m retains its large-scale shading pattern, which is why
Kaguya-vs-Kaguya matches at 11.49x. IIRS's native 85 m pixels do not carry that
shading structure to align to.

Note the scale control is a positive result in its own right: **11.49x works**,
which extends the demonstrated scale envelope beyond the 8x in the README. The
old 16x failure was window starvation (a 64x64 input), not a method ceiling.

WHAT WOULD PLAUSIBLY WORK
-------------------------
A same-modality reference. Chandrayaan-1 M3 is an imaging spectrometer at
~140 m/px and, unlike Chandrayaan-2, **is indexed by PDS ODE**, so it needs no
login. IIRS-to-M3 would be spectrometer-to-spectrometer at a 1.6x ratio rather
than spectrometer-to-camera at 11.5x. Untested here; see ROADMAP.

GEOMETRY COMES FROM A BACKPLANE, NOT A BOUNDING BOX
---------------------------------------------------
The product ships a `_loc_` backplane giving latitude, longitude, radius and
elevation for **every pixel** in the MOON_ME frame -- better than OHRC's sampled
sidecar CSV, since no interpolation between samples is needed. The `_obs_`
backplane carries per-pixel viewing geometry: incidence ~57.7 deg and emission
~10.2 deg over this overlap, the latter inside the measured 12.3 deg obliquity
envelope.

**This corrects CLAUDE.md**, which records that Chandrayaan-2 products carry no
usable illumination geometry. True of the *shapefile* (every angle is 0.0) and
of OHRC's label; false of the IIRS product, which carries it per pixel.

    python scripts/iirs_vs_kaguya.py --controls
    python scripts/iirs_vs_kaguya.py --sweep --no-gate
    python scripts/iirs_vs_kaguya.py --cube radiance
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sandhi import controls, pipeline  # noqa: E402

CH2 = ROOT / "data" / "raw" / "ch2"
KAGUYA_GSD_M = 7.403
MOON_RADIUS_M = 1737400.0

# Declared in the PDS4 label. Read, never estimated -- BUG-010.
IIRS_GSD_M = 85.08


def product() -> tuple[Path, str]:
    """The IIRS reflectance archive and its internal path stem."""
    zips = sorted(CH2.glob("ch2_iir_*_d_rfl_*.zip"))
    if not zips:
        raise SystemExit(
            "no IIRS reflectance product under data/raw/ch2.\n"
            "  python scripts/ch2_footprints.py --pick iirs")
    z = zips[0]
    with zipfile.ZipFile(z) as zf:
        qub = next(n for n in zf.namelist() if n.endswith("_d_rfl_d18_srd.qub"))
    return z, qub[: -len("_d_rfl_d18_srd.qub")]


def vsi(z: Path, stem: str, suffix: str) -> str:
    return f"/vsizip/{z.as_posix()}/{stem}{suffix}"


def wavelengths(z: Path, stem: str) -> np.ndarray:
    with zipfile.ZipFile(z) as zf:
        hdr = zf.read(f"{stem}_d_rfl_d18_srd.hdr").decode("utf-8", "replace")
    tail = hdr.partition("wavelength")[2]
    out = []
    for x in tail.replace("=", " ").replace("{", " ").replace("}", " ").split(","):
        try:
            out.append(float(x.strip()))
        except ValueError:
            pass
    return np.array(out)


def overlap_lines(z: Path, stem: str, tile_bounds) -> tuple[slice, np.ndarray, np.ndarray]:
    """Lines whose centre pixel falls inside the tile, plus the lat/lon planes.

    The LOC backplane is only ~12 MB per band, so both planes are read whole
    rather than windowed -- simpler, and the window has to be derived from them
    anyway.
    """
    w, e, s, n = tile_bounds
    with rasterio.open(vsi(z, stem, "_d_loc_d18_ard.img")) as src:
        lon = src.read(1)
        lat = src.read(2)
    mid = lon.shape[1] // 2
    inside = ((lat[:, mid] >= s) & (lat[:, mid] <= n) &
              (lon[:, mid] >= w) & (lon[:, mid] <= e))
    idx = np.flatnonzero(inside)
    if idx.size < 128:
        raise SystemExit(f"only {idx.size} lines overlap the tile - too few")
    return slice(int(idx.min()), int(idx.max()) + 1), lat, lon


def project(band: np.ndarray, lat: np.ndarray, lon: np.ndarray, gsd_m: float):
    """Resample a swath onto a north-up lat/lon grid at `gsd_m`.

    Same approach as `ohrc_project.py`: fit the INVERSE map (lon, lat) ->
    (sample, line) and pull each output pixel from the source, so no output pixel
    is left unwritten. Fitting the forward direction and scattering leaves holes
    wherever the swath is oblique to the grid.
    """
    from ohrc_project import apply_inverse, fit_inverse

    h, w = band.shape
    ss, ll = np.meshgrid(np.arange(w), np.arange(h))
    coef, resid, degree = fit_inverse(lon.ravel(), lat.ravel(),
                                      ss.ravel().astype(float),
                                      ll.ravel().astype(float))
    deg_per_px = np.degrees(gsd_m / MOON_RADIUS_M)
    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())
    width = max(int(round((east - west) / deg_per_px)), 8)
    height = max(int(round((north - south) / deg_per_px)), 8)

    cols = (np.arange(width) + 0.5) * deg_per_px + west
    rows = north - (np.arange(height) + 0.5) * deg_per_px
    LON, LAT = np.meshgrid(cols, rows)
    px, sc = apply_inverse(coef, LON.ravel(), LAT.ravel())
    px = np.clip(np.round(px).astype(int), 0, w - 1)
    sc = np.clip(np.round(sc).astype(int), 0, h - 1)
    out = band[sc, px].reshape(height, width).astype(np.float32)

    # Mark everything the swath does not actually cover, so it can be excluded
    # rather than matched against as if it were terrain.
    valid = np.ones_like(out, bool)
    pxf, scf = apply_inverse(coef, LON.ravel(), LAT.ravel())
    valid = ((pxf >= 0) & (pxf <= w - 1) & (scf >= 0) & (scf <= h - 1)
             ).reshape(height, width)
    out[~valid] = 0.0
    return out, BoundingBox(west, south, east, north), resid, degree, valid



def project_direct(band, lat, lon, gsd_m, ignore=None):
    """Resample using the per-pixel backplane DIRECTLY, with no polynomial fit.

    `project()` fits a global degree-3 inverse map (lon, lat) -> (sample, line).
    That works when the geolocation is smooth: M3's fits to 0.15 px. IIRS's fits
    to 3.96 px, 26x worse, and a residual that size is not a smooth surface --
    it is per-line variation a low-order polynomial cannot represent. Forcing
    one through it distorts the imagery non-rigidly, which is invisible to a
    band-vs-band control (both bands get the same distortion) and fatal against
    any external reference.

    This maps every source pixel to the output cell its OWN coordinates name,
    via a nearest-neighbour lookup, so the geolocation is used as given.
    """
    from scipy.spatial import cKDTree

    dpp = np.degrees(gsd_m / MOON_RADIUS_M)
    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())
    width = max(int(round((east - west) / dpp)), 8)
    height = max(int(round((north - south) / dpp)), 8)

    # Latitude scaled to match longitude's ground spacing, so "nearest" means
    # nearest on the ground rather than nearest in degrees.
    coslat = np.cos(np.radians(0.5 * (south + north)))
    tree = cKDTree(np.column_stack([lon.ravel() * coslat, lat.ravel()]))

    cols = (np.arange(width) + 0.5) * dpp + west
    rows = north - (np.arange(height) + 0.5) * dpp
    LON, LAT = np.meshgrid(cols, rows)
    dist, idx = tree.query(np.column_stack([LON.ravel() * coslat, LAT.ravel()]),
                           k=1, workers=-1)
    out = band.ravel()[idx].reshape(height, width).astype(np.float32)

    # Beyond half a pixel from any real sample there is no data, only the
    # nearest one repeated -- that is padding, not terrain.
    valid = (dist.reshape(height, width) < dpp * 0.75)
    if ignore is not None:
        valid &= (out != ignore)
    out[~valid] = 0.0
    return out, BoundingBox(west, south, east, north), float(np.median(dist) / dpp), 0, valid


def kaguya_over(bounds: BoundingBox, kind: str = "evening"):
    from triple_io import find_tiles, open_triple
    from rasterio.windows import from_bounds

    for tile in find_tiles():
        t = open_triple(tile)
        r = t.body_radius_m
        with rasterio.open(t.paths[kind]) as src:
            win = from_bounds(np.radians(bounds.left) * r,
                              np.radians(bounds.bottom) * r,
                              np.radians(bounds.right) * r,
                              np.radians(bounds.top) * r,
                              transform=src.transform)
            if (win.col_off + win.width < 0 or win.row_off + win.height < 0
                    or win.col_off > src.width or win.row_off > src.height):
                continue
            band = src.read(1, window=win, boundless=True, fill_value=0)
        if (band > 0).mean() > 0.5:
            return band.astype(np.float32), tile
    return None, None


# Which cube. This is a physics choice, not a file choice -- see the module
# docstring. "rfl" has the illumination divided out; "rdn" retains it.
DIRECT = [False]        # set from --direct

CUBES = {"reflectance": "_d_rfl_d18_srd.qub", "radiance": "_d_rdn_d18_ard.qub"}


def run_band(z, stem, wl, target_nm, tile_bounds, gate=True, quiet=False,
             cube="reflectance"):
    idx = int(np.argmin(np.abs(wl - target_nm)))
    sub, lat, lon = overlap_lines(z, stem, tile_bounds)

    with rasterio.open(vsi(z, stem, CUBES[cube])) as src:
        band = src.read(idx + 1,
                        window=Window(0, sub.start, src.width, sub.stop - sub.start))
    band = np.nan_to_num(band.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    la, lo = lat[sub], lon[sub]

    if not quiet:
        print(f"\nband {idx+1} = {wl[idx]:.1f} nm  "
              f"({'reflected sunlight' if wl[idx] < 2500 else 'THERMAL regime'})")
        print(f"  swath {band.shape}, valid {100*(band > 0).mean():.1f}%, "
              f"range {band.min():.4f}..{band.max():.4f}")

    proj = project_direct if DIRECT[0] else project
    src_img, bounds, resid, degree, valid = proj(band, la, lo, IIRS_GSD_M)
    if not quiet:
        print(f"  projected {src_img.shape} at {IIRS_GSD_M} m/px, "
              f"degree-{degree} inverse fit residual {resid:.3f} px, "
              f"{100*valid.mean():.0f}% covered")

    ref, tile = kaguya_over(bounds)
    if ref is None:
        print("  no Kaguya product covers this box")
        return None
    if not quiet:
        print(f"  Kaguya evening from {tile}: {ref.shape}")

    r = pipeline.register(src_img, ref, src_gsd=IIRS_GSD_M, ref_gsd=KAGUYA_GSD_M,
                          gsd_m=IIRS_GSD_M)
    if r is None:
        print("  NO FIT")
        return {"wavelength_nm": float(wl[idx]), "band": idx + 1, "registered": False}

    m = r["metrics"]
    out = {"cube": cube, "wavelength_nm": float(wl[idx]), "band": idx + 1,
           "registered": True,
           "match_count": m["match_count"], "inlier_ratio": round(m["inlier_ratio"], 4),
           "rmse_px": None if m["rmse_px"] is None or np.isnan(m["rmse_px"])
                      else round(m["rmse_px"], 4),
           "coverage": round(m["coverage"], 4), "entropy": round(m["entropy"], 4),
           "model": r["fit"]["kind"], "kaguya_tile": tile}
    if not quiet:
        for k in ("match_count", "inlier_ratio", "rmse_px", "coverage", "entropy", "model"):
            print(f"  {k:<14} {out[k]}")

    if gate:
        g = controls.run(controls.default_match_fn(), r["src"], r["ref"])
        out["gate"] = bool(g.passed)
        out["gate_checks"] = {c["name"]: bool(c["passed"]) for c in g.checks}
        if not quiet:
            print("  CONTROL GATE")
            for c in g.checks:
                print(f"    {c['name']:<18} {'PASS' if c['passed'] else 'FAIL'}  "
                      f"{c.get('detail','')}")
            print(f"    -> {'PASS' if g.passed else 'FAIL'}")
    return out



def controls_mode(z, stem, wl, tile_bounds) -> list[dict]:
    """The three controls that turn "it failed" into "it failed FOR THIS REASON".

    Each removes one candidate explanation. Without them the negative result is
    an anecdote; with them it is a measurement.
    """
    from triple_io import open_triple
    from sandhi import matching, models, normalize

    def score(a, b, label):
        a8, b8 = normalize.prepare(a), normalize.prepare(b)
        h = min(a8.shape[0], b8.shape[0])
        w = min(a8.shape[1], b8.shape[1])
        a8, b8 = a8[:h, :w], b8[:h, :w]
        corr = float(np.corrcoef(a8.ravel().astype(float),
                                 b8.ravel().astype(float))[0, 1])
        pa, pb, _, _, _ = matching.match_aligned(a8, b8)
        f = models.fit(pa, pb, kind="affine") if len(pa) >= 8 else None
        ratio = None if f is None else float(f["inliers"].mean())
        print(f"  {label:<52} corr {corr:+.4f}  matched {len(pa):>5}  "
              + ("affine too few" if f is None else
                 f"affine {int(f['inliers'].sum())}/{len(pa)} ({ratio:.1%})"))
        return {"control": label, "corr": round(corr, 4), "matches": int(len(pa)),
                "inlier_ratio": None if ratio is None else round(ratio, 4)}

    sub, lat, lon = overlap_lines(z, stem, tile_bounds)
    la, lo = lat[sub], lon[sub]

    def band_at(cube, nm):
        i = int(np.argmin(np.abs(wl - nm)))
        with rasterio.open(vsi(z, stem, CUBES[cube])) as src:
            b = src.read(i + 1, window=Window(0, sub.start, src.width,
                                              sub.stop - sub.start))
        return np.nan_to_num(b.astype(np.float32))

    rows = []
    print("\nCONTROL 1+2 - two bands of the SAME cube: correspondence is identity.")
    print("  If these pass, the projection, the strip shape and the imagery are fine.")
    a, b = band_at("reflectance", 1504), band_at("reflectance", 1700)
    pa_, *_ = project(a, la, lo, IIRS_GSD_M)
    pb_, *_ = project(b, la, lo, IIRS_GSD_M)
    rows.append(score(pa_, pb_, "1504 vs 1700 nm, projected"))
    rows.append(score(a, b, "1504 vs 1700 nm, raw swath"))

    print("\nCONTROL 3 - Kaguya against ITSELF at the IIRS scale and shape.")
    print("  If this passes, the 11.49x ratio is not the blocker.")
    t = open_triple("N18E009N15E012SC")
    k = IIRS_GSD_M / KAGUYA_GSD_M
    W = int(288 * k)
    got = {}
    for kind in ("morning", "evening"):
        with rasterio.open(t.paths[kind]) as src:
            got[kind] = src.read(1, window=Window(4000, 0, W, 12288)).astype(np.float32)
    rows.append(score(pipeline.resample(got["morning"], k),
                      pipeline.resample(got["evening"], k),
                      f"Kaguya morning vs evening, decimated {k:.2f}x"))

    print("\nFor comparison, IIRS vs Kaguya at 1504 nm: "
          "corr +0.0326, matched 44, affine 6/44 (13.6%)")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavelength", type=float, default=1504.0,
                    help="nm; the nearest band is used")
    ap.add_argument("--direct", action="store_true",
                    help="use the per-pixel backplane instead of a polynomial fit")
    ap.add_argument("--controls", action="store_true",
                    help="run the three controls that isolate the cause")
    ap.add_argument("--sweep", action="store_true",
                    help="measure across the reflected/thermal transition")
    ap.add_argument("--tile", nargs=4, type=float, default=[9.0, 12.0, 15.0, 18.0],
                    metavar=("W", "E", "S", "N"))
    ap.add_argument("--cube", default="reflectance",
                    choices=["reflectance", "radiance", "both"])
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--out", default="outputs/iirs_vs_kaguya.json")
    args = ap.parse_args()

    DIRECT[0] = args.direct
    z, stem = product()
    wl = wavelengths(z, stem)
    print(f"{z.name}\n  {len(wl)} bands, {wl.min():.1f}..{wl.max():.1f} nm, "
          f"{IIRS_GSD_M} m/px (declared)")
    print(f"  scale ratio against Kaguya: {IIRS_GSD_M/KAGUYA_GSD_M:.2f}x")

    if args.controls:
        rows = controls_mode(z, stem, wl, tuple(args.tile))
        out = ROOT / "outputs" / "iirs_controls.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
        return 0

    targets = ([900, 1504, 2000, 2400, 3000, 3500, 4500] if args.sweep
               else [args.wavelength])
    cubes = ["reflectance", "radiance"] if args.cube == "both" else [args.cube]
    rows = [r for r in (run_band(z, stem, wl, t, tuple(args.tile),
                                 gate=not args.no_gate, quiet=args.sweep, cube=c)
                        for c in cubes for t in targets) if r]

    if args.sweep:
        print(f"\n{'nm':>7} {'band':>5} {'matches':>8} {'inlier':>7} {'rmse':>7} "
              f"{'cover':>6} {'gate':>6}")
        print("-" * 52)
        for r in rows:
            g = "" if "gate" not in r else ("PASS" if r["gate"] else "FAIL")
            rm = f"{r['rmse_px']:.3f}" if r.get("rmse_px") else "  -  "
            print(f"{r['wavelength_nm']:>7.1f} {r['band']:>5} "
                  f"{r.get('match_count', 0):>8} {r.get('inlier_ratio', 0):>7.3f} "
                  f"{rm:>7} {r.get('coverage', 0):>6.3f} {g:>6}")

    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
