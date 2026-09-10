"""Register Chandrayaan-2 TMC-2 ortho against Kaguya TC. Control-gated.

WHY THIS CASE EXISTS
--------------------
The problem statement names OHRC, TMC **and** IIRS. OHRC was demonstrated first
because it is the headline instrument, but TMC-2 is the one that carries the
mission's actual coverage: 8436 catalogued products against OHRC's 624, and the
single strip used here is 3.55 Gpx against OHRC's 0.94.

It is also the harder and more representative task. OHRC and its Kaguya
reference differ in scale by 28x but the OHRC strip had to be projected onto the
Kaguya grid first, so much of the geometry was resolved before matching. TMC-2
ships as a map-projected GeoTIFF on a SelenoGraphic sphere, so source and
reference arrive in different frames, at different ground sample distances,
from different spacecraft, years apart, under uncontrolled illumination. That is
the PS's production task stated exactly.

GEOMETRY, AND A UNIT TRAP
-------------------------
TMC-2's transform is in **degrees** (GEOGCS on a sphere of radius 1737400 m).
Kaguya's is in **projected metres** on the same sphere. Mixing them is BUG-006
in reverse -- there, projected northing was fed to cos() as if it were degrees.
Every conversion here is explicit and one-directional: TMC pixel -> degrees ->
Kaguya pixel.

The correspondence is therefore a geometric prior, not a guess: both products
declare a CRS, so the Kaguya window covering a TMC window is computed, never
searched for. Scale likewise comes from declared GSD (TMC-2 5.05 m/px, Kaguya
7.403 m/px, ratio 1.466) rather than being estimated -- blind scale estimation
is confounded (BUG-010).

    python scripts/tmc_vs_kaguya.py --lat 1.5
    python scripts/tmc_vs_kaguya.py --lat 1.5 --size 768 --no-gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sandhi import controls, metrics, outputs, pipeline  # noqa: E402

MOON_RADIUS_M = 1737400.0
TMC_GSD_M = 5.05
KAGUYA_GSD_M = 7.403


def strip_window(src, lat_deg: float, size: int):
    """A TMC-2 window centred on the imaged swath at a given latitude.

    The imaged data is a diagonal band inside a much larger bounding rectangle,
    drifting across roughly 5000 columns down the strip, so the column has to be
    found rather than assumed -- taking the rectangle's centre lands in zero
    padding for most latitudes.
    """
    t = src.transform
    row = int(round((lat_deg - t.f) / t.e))
    row = int(np.clip(row, 0, src.height - size))

    band = src.read(1, window=Window(0, row, src.width, 8))
    valid = np.flatnonzero(band.max(axis=0) > 0)
    if valid.size < size:
        raise SystemExit(f"no imaged swath at lat {lat_deg} (only {valid.size} valid columns)")
    col = int(np.clip((valid[0] + valid[-1]) // 2 - size // 2, 0, src.width - size))

    img = src.read(1, window=Window(col, row, size, size)).astype(np.float32)
    west, north = t * (col, row)
    east, south = t * (col + size, row + size)
    return img, BoundingBox(west, min(south, north), east, max(south, north)), (row, col)


def kaguya_over(bounds: BoundingBox, kind: str = "evening"):
    """Read the Kaguya band covering a lon/lat box, over every local tile.

    Uses `triple_io.open_triple`, which reads the detached `.LBL` through GDAL's
    PDS driver and asserts the three grids agree, rather than globbing for
    `.IMG` directly -- the label is where the projection lives.

    Kaguya's transform is in projected metres while the box is in degrees, so it
    converts by arc length on the sphere, using the radius the CRS declares
    rather than a constant (BUG-006 was exactly this conversion, done wrong).
    `boundless` matters: near a tile edge the box can run past the product, and
    a silent short read would misalign everything downstream.
    """
    from triple_io import find_tiles, open_triple

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
                continue                      # box lies outside this tile
            band = src.read(1, window=win, boundless=True, fill_value=0)
        if (band > 0).mean() > 0.5:
            return band.astype(np.float32), tile
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=1.5,
                    help="latitude on the TMC-2 strip to register")
    ap.add_argument("--size", type=int, default=1024, help="TMC-2 window, px")
    ap.add_argument("--reference", default="evening", choices=["evening", "morning"])
    ap.add_argument("--no-gate", action="store_true", help="skip the control gate")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    from ch2_io import open_tmc_pair
    t = open_tmc_pair()

    with rasterio.open(t.ortho_path) as src:
        img, bounds, (row, col) = strip_window(src, args.lat, args.size)

    print(f"TMC-2 window ({row},{col}) {args.size}x{args.size}")
    print(f"  lon {bounds.left:.4f}..{bounds.right:.4f}  "
          f"lat {bounds.bottom:.4f}..{bounds.top:.4f}")
    print(f"  valid {100*(img > 0).mean():.1f}%  std {img.std():.1f}")

    ref, tile = kaguya_over(bounds, args.reference)
    if ref is None:
        print(f"\nNo Kaguya {args.reference} product covers this box.")
        print("Fetch one:  python scripts/kaguya.py fetch --bbox "
              f"{bounds.left:.1f} {bounds.right:.1f} {bounds.bottom:.1f} {bounds.top:.1f}")
        return 1
    print(f"Kaguya {args.reference} from {tile}: {ref.shape}, "
          f"valid {100*(ref > 0).mean():.1f}%")

    print(f"\nregistering at declared GSD {TMC_GSD_M} -> {KAGUYA_GSD_M} m/px "
          f"(ratio {KAGUYA_GSD_M/TMC_GSD_M:.3f})")
    r = pipeline.register(img, ref, src_gsd=TMC_GSD_M, ref_gsd=KAGUYA_GSD_M,
                          gsd_m=KAGUYA_GSD_M)
    if r is None:
        print("\nNO FIT. Not enough matches survived.")
        return 1

    m = r["metrics"]
    print("")
    print("=" * 60)
    # rmse_source_px is the PS's own unit -- "sub-pixel accuracy OF SOURCE
    # IMAGE" -- so it belongs in the printed block, not only in the JSON.
    for k in ("match_count", "inlier_count", "inlier_ratio", "rmse_px",
              "rmse_m", "rmse_source_px", "sub_pixel_source",
              "coverage", "entropy"):
        if k in m and m[k] is not None:
            print(f"  {k:<16} {m[k]}")
    print(f"  {'model':<14} {r['fit']['kind']}")
    print(f"  {'stages':<14} {r['stages']}")
    print("=" * 60)

    report = {"lat": args.lat, "size": args.size, "reference": args.reference,
              "tmc_window": [row, col], "kaguya_tile": tile,
              "bounds": list(bounds), "metrics": m,
              "model": r["fit"]["kind"], "stages": r["stages"]}

    if not args.no_gate:
        print("\nCONTROL GATE")
        # Gate the images the pipeline ACTUALLY matched -- `r["src"]`/`r["ref"]`
        # are post-resampling, so both are on the common grid. Passing the raw
        # pair here hands the gate a 1024 px source and a 699 px reference and
        # it dies inside LoFTR, because controls does no GSD handling of its own.
        gate = controls.run(controls.default_match_fn(), r["src"], r["ref"])
        for c in gate.checks:
            print(f"  {c['name']:<18} {'PASS' if c['passed'] else 'FAIL'}  {c.get('detail','')}")
        print(f"  -> {'PASS' if gate.passed else 'FAIL'}")
        report["gate"] = {"passed": bool(gate.passed),
                          "checks": {c["name"]: bool(c["passed"]) for c in gate.checks}}
        if not gate.passed:
            print("\nGATE FAILED - this number is not reportable.")

    out = ROOT / args.out_dir
    out.mkdir(exist_ok=True)
    warped = pipeline.warp(r["src"], r)
    paths = outputs.write_all(r, warped, out, "tmc2")
    (out / "tmc_vs_kaguya.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out/'tmc_vs_kaguya.json'} and {len(paths)} deliverables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
