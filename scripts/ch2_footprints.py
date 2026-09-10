"""Where do Chandrayaan-2 OHRC and TMC-2 actually point, and what overlaps LROC NAC?

Product filenames on PRADAN encode a timestamp but no location, so choosing what
to download without footprints means buying 1.2 GB blind. These shapefiles ship
separately from the imagery and answer it for a few MB.

Cross-matches the corner coordinates against ODE's LROC NAC index, so a
candidate is only reported if reference imagery exists over the same ground at
more than one illumination.

    python scripts/ch2_footprints.py --summary
    python scripts/ch2_footprints.py --pick tmc --max-lat 60 --limit 10
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import shapefile

from survey_coverage import INC_MAX, INC_MIN, ode_count

ROOT = Path(__file__).resolve().parent.parent
CH2 = ROOT / "data" / "raw" / "ch2"

CORNERS = [("UL_LAT", "UL_LON"), ("UR_LAT", "UR_LON"),
           ("BL_LAT", "BL_LON"), ("BR_LAT", "BR_LON")]

INCIDENCE_BINS = [(0, 30), (30, 50), (50, 70), (70, 90)]


def load(pattern: str) -> list[dict]:
    """Read every matching shapefile into plain dicts of its attributes."""
    out = []
    for path in sorted(glob.glob(str(CH2 / "*ShapeFiles" / "*" / pattern))):
        reader = shapefile.Reader(path)
        names = [f[0] for f in reader.fields[1:]]
        for rec in reader.records():
            row = dict(zip(names, list(rec)))
            row["_file"] = Path(path).stem
            out.append(row)
    return out


def bbox(rec: dict) -> tuple | None:
    """(west, east, south, north) from the corner fields, or None if unusable."""
    lats, lons = [], []
    for lat_key, lon_key in CORNERS:
        if lat_key in rec and lon_key in rec:
            try:
                lat, lon = float(rec[lat_key]), float(rec[lon_key])
            except (TypeError, ValueError):
                continue
            if -90 <= lat <= 90:
                lats.append(lat)
                lons.append(lon % 360)
    if len(lats) < 2:
        return None
    # Strips crossing the prime meridian would need unwrapping; skip them
    # rather than silently produce a bbox spanning the whole Moon.
    if max(lons) - min(lons) > 180:
        return None
    return (min(lons), max(lons), min(lats), max(lats))


def summarise() -> int:
    for label, pattern in [("OHRC", "ch2_ohr_cal*.shp"), ("TMC-2", "ch2_tmc_cal*.shp"),
                           ("TMC-2 ortho", "ch2_tmc_derived_ortho*.shp"),
                           ("TMC-2 DTM", "ch2_tmc_derived_dtm*.shp"),
                           # IIRS: needs IIRS_ShapeFiles.zip from PRADAN's Other
                           # Downloads. Absent until then, and load() returns [].
                           ("IIRS", "ch2_iir*.shp")]:
        recs = load(pattern)
        if not recs:
            continue
        boxes = [b for b in (bbox(r) for r in recs) if b]
        if not boxes:
            print(f"{label}: {len(recs)} records, no usable corners")
            continue
        centres = np.array([(b[2] + b[3]) / 2 for b in boxes])

        bands = [("polar S (<-60)", centres < -60),
                 ("mid S (-60..-20)", (centres >= -60) & (centres < -20)),
                 ("equatorial (-20..20)", np.abs(centres) <= 20),
                 ("mid N (20..60)", (centres > 20) & (centres <= 60)),
                 ("polar N (>60)", centres > 60)]
        print(f"\n{label}: {len(recs)} records, {len(boxes)} with usable corners")
        for name, mask in bands:
            n = int(mask.sum())
            bar = "#" * int(40 * n / max(len(centres), 1))
            print(f"  {name:<22} {n:>6}  {bar}")
    return 0


# Side of the square probe window, in degrees, used to score a strip. Roughly
# 30 km at the equator -- big enough to hold plenty of NAC frames, small enough
# that the count means "coverage here" rather than "this box is enormous".
PROBE_DEG = 1.0


def probe_box(b: tuple, at_lat: float) -> tuple | None:
    """A small square box where a footprint crosses `at_lat`, or None if it misses.

    TMC-2 derived products are strips up to ~57 degrees long. Scoring the whole
    bounding box measures strip length, not suitability: a longer strip always
    wins on raw NAC count. Probing a fixed-size window makes candidates
    comparable. See BUGS.md BUG-009.
    """
    west, east, south, north = b
    if not (south <= at_lat <= north):
        return None
    half = PROBE_DEG / 2
    lon_c = (west + east) / 2
    # Never widen the box beyond the strip itself, or we score empty space.
    return (max(west, lon_c - half), min(east, lon_c + half),
            at_lat - half, at_lat + half)


def area_deg2(b: tuple) -> float:
    west, east, south, north = b
    return max(east - west, 1e-9) * max(north - south, 1e-9)


def pick(kind: str, max_lat: float, limit: int, at_lat: float = 0.0) -> int:
    pattern = {"ohrc": "ch2_ohr_cal*.shp",
               "tmc": "ch2_tmc_derived_ortho*.shp"}[kind]
    recs = load(pattern)

    cands = []
    for rec in recs:
        b = bbox(rec)
        if not b:
            continue
        if kind == "ohrc":
            # Small footprints (3 km swath), so the whole box IS the probe.
            centre = (b[2] + b[3]) / 2
            if abs(centre) <= max_lat:
                cands.append((b, b, rec))
        else:
            # Long strips: score a fixed window where the strip crosses at_lat.
            probe = probe_box(b, at_lat)
            if probe and abs(at_lat) <= max_lat:
                cands.append((probe, b, rec))

    if kind == "ohrc":
        cands.sort(key=lambda x: abs((x[0][2] + x[0][3]) / 2))
        print(f"{len(cands)} OHRC products within +/-{max_lat:g} deg latitude")
        print(f"scoring the whole footprint (3 km swath, so the box is the target)\n")
    else:
        # Prefer narrower strips: less wasted download per useful square degree.
        cands.sort(key=lambda x: area_deg2(x[1]))
        print(f"{len(cands)} TMC-2 strips crossing latitude {at_lat:g}")
        print(f"scoring a {PROBE_DEG:g}x{PROBE_DEG:g} deg probe window at that latitude,")
        print("NOT the whole strip - see BUGS.md BUG-009\n")

    print(f"{'PRODUCT_ID':<44} {'lat':>7} {'lon':>7} {'strip deg2':>11} "
          f"{'NAC':>5} {'/deg2':>7} {'bins':>5}")
    print("-" * 92)

    shown = 0
    for probe, full, rec in cands:
        if shown >= limit:
            break
        total = ode_count("CDRNAC4", probe, ihid="LRO", iid="LROC")
        if not total:
            continue
        bins = sum(
            1 for lo, hi in INCIDENCE_BINS
            if (ode_count("CDRNAC4", probe, ihid="LRO", iid="LROC",
                          **{INC_MIN: lo, INC_MAX: hi}) or 0) >= 2
        )
        density = total / area_deg2(probe)
        flag = "  <- multi-illumination" if bins >= 2 else ""
        print(f"{str(rec.get('PRODUCT_ID',''))[:44]:<44} "
              f"{(probe[2]+probe[3])/2:>7.2f} {(probe[0]+probe[1])/2:>7.2f} "
              f"{area_deg2(full):>11.1f} {total:>5} {density:>7.0f} {bins:>5}{flag}")
        shown += 1
    return 0



def pick_iirs(limit: int = 10) -> int:
    """Rank IIRS reflectance products by overlap with Kaguya tiles ALREADY on disk.

    Ranking by latitude or by footprint area, as the OHRC and TMC-2 pickers do,
    is the wrong question for IIRS: it is a 16-35 degree long strip, so area
    measures strip length rather than usefulness, and a product is only usable if
    a reference image exists for the same ground. Overlap with the downloaded
    Kaguya triples is the quantity that decides whether it can be registered at
    all, and it costs nothing to compute -- unlike a ~GB download of the wrong
    product.

    Reflectance, not raw or calibrated radiance, is deliberate. IIRS spans
    0.8-5.0 um; beyond ~3 um thermal EMISSION dominates and does not shade like
    visible imagery, so the raw cube is cross-modal in a way that is not merely
    cross-scale. A derived reflectance product is reflected sunlight, the same
    physical quantity Kaguya records.
    """
    from triple_io import find_tiles, open_triple

    tiles = {}
    for name in find_tiles():
        t = open_triple(name)
        lat0, lon0 = t.pixel_latlon(0, 0)
        lat1, lon1 = t.pixel_latlon(t.height - 1, t.width - 1)
        tiles[name] = (min(lon0, lon1), max(lon0, lon1),
                       min(lat0, lat1), max(lat0, lat1))
    if not tiles:
        print("no Kaguya tiles downloaded - nothing to register IIRS against")
        return 1
    print(f"Kaguya tiles on disk: {len(tiles)}")
    for n, b in tiles.items():
        print(f"  {n}  lon {b[0]:.2f}..{b[1]:.2f}  lat {b[2]:.2f}..{b[3]:.2f}")

    cands = []
    for rec in load("ch2_iir_derived_refl*.shp"):
        b = bbox(rec)
        if not b:
            continue
        w, e, s, n = b
        for name, (tw, te, ts, tn) in tiles.items():
            ow, oh = min(e, te) - max(w, tw), min(n, tn) - max(s, ts)
            if ow > 0 and oh > 0:
                cands.append((ow * oh, ow, oh, name, rec, b))
    cands.sort(reverse=True, key=lambda c: c[0])

    print(f"\n{len(cands)} IIRS reflectance products overlap a downloaded tile")
    for area, ow, oh, name, rec, b in cands[:limit]:
        print(f"\n  {rec['PRODUCT_ID']}")
        print(f"    overlaps {name}: {ow:.3f} x {oh:.3f} deg = {area:.4f} deg^2")
        print(f"    footprint lon {b[0]:.3f}..{b[1]:.3f}  lat {b[2]:.3f}..{b[3]:.3f}")
        print(f"    observed  {rec['OBS_ST_TIM']}")
        # PRADAN serves reflectance products with an "_srd" suffix, but the
        # shapefile's DOWNLOAD field records it inconsistently -- only 40 of 729
        # records carry it. Searching the browse page for the bare name returns
        # "File not found", so print the served form as well.
        dl = str(rec["DOWNLOAD"])
        srd = dl if dl.endswith("_srd.zip") else dl[:-4] + "_srd.zip"
        print(f"    download  {srd}")
        if srd != dl:
            print(f"              (shapefile says {dl} -- missing the _srd suffix)")
    if cands:
        print("\n  Angles in these records are 0.0, as for TMC-2: IIRS labels carry no")
        print("  usable illumination geometry, so it must be recovered by correlation.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--pick", choices=["ohrc", "tmc", "iirs"])
    parser.add_argument("--max-lat", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--at-lat", type=float, default=0.0,
                        help="latitude at which to probe TMC-2 strips")
    args = parser.parse_args()

    if args.pick == "iirs":
        return pick_iirs(args.limit)
    if args.pick:
        return pick(args.pick, args.max_lat, args.limit, args.at_lat)
    return summarise()


if __name__ == "__main__":
    raise SystemExit(main())
