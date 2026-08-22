"""Estimate the sun azimuth of a real image by matching it against rendered DEM.

Kaguya TC map-product labels carry no sun geometry (`START_TIME = UNK`, no
incidence or azimuth fields), only `STANDARD_GEOMETRY = (30, 0, 30)` describing
the photometric correction applied. So Stage A cannot read the illumination
direction off the label -- it has to be recovered.

Sweeping azimuth and taking the best correlation against a DEM render does
that, and simultaneously tests the entire Stage-B premise on real data:

    if the renderer is correct, the MORNING image must peak at an EASTERN
    azimuth and the EVENING image at a WESTERN one.

Nothing forces that outcome. It is a falsifiable prediction.

    python scripts/estimate_sun.py --size 384 --step 15
"""

import argparse

import numpy as np

from photometric import render
from triple_io import find_tiles, open_triple

# Both products are corrected to incidence 30 deg, so sun elevation is 90 - 30.
STANDARD_INCIDENCE_DEG = 30.0


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compass(azimuth_deg: float) -> str:
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int(((azimuth_deg % 360) + 22.5) // 45) % 8]


def sweep(tile: str | None, size: int, step: int, shadows: bool) -> int:
    tiles = find_tiles()
    if not tiles:
        print("No downloaded triples. Run: python scripts/kaguya.py fetch")
        return 1

    tile = tile or tiles[0]
    triple = open_triple(tile)
    row = (triple.height - size) // 2
    col = (triple.width - size) // 2
    bands = triple.read_window(row, col, size)

    dem = bands["dem"]
    if not np.isfinite(dem).all():
        dem = np.nan_to_num(dem, nan=float(np.nanmean(dem)))

    # Latitude of the window centre, for the east-west spacing correction.
    # Must go through pixel_latlon - the transform is in projected metres.
    lat, _ = triple.pixel_latlon(row + size / 2, col + size / 2)
    map_scale_m = abs(triple.transform.a)

    elevation = 90.0 - STANDARD_INCIDENCE_DEG
    print(f"tile {tile}  window ({row},{col}) {size}x{size}")
    print(f"latitude {lat:.3f} deg   scale {map_scale_m:.3f} m/px   "
          f"sun elevation {elevation:.0f} deg   shadows={shadows}\n")

    azimuths = list(range(0, 360, step))
    results = {}
    print(f"{'azimuth':>8} {'':4} {'morning':>9} {'evening':>9}")
    for az in azimuths:
        rendered = render(dem, map_scale_m, lat, az, elevation, shadows=shadows)
        cm = correlate(rendered, bands["morning"])
        ce = correlate(rendered, bands["evening"])
        results[az] = (cm, ce)
        print(f"{az:>6}deg {compass(az):>4} {cm:>+9.3f} {ce:>+9.3f}")

    best_m = max(azimuths, key=lambda a: results[a][0])
    best_e = max(azimuths, key=lambda a: results[a][1])

    print(f"\nbest morning azimuth: {best_m:>3} deg ({compass(best_m)})  "
          f"r={results[best_m][0]:+.3f}")
    print(f"best evening azimuth: {best_e:>3} deg ({compass(best_e)})  "
          f"r={results[best_e][1]:+.3f}")

    # Angular separation between the two recovered directions.
    sep = abs(best_m - best_e) % 360
    sep = min(sep, 360 - sep)
    print(f"separation: {sep} deg")

    east = 0 < best_m < 180
    west = 180 < best_e < 360
    print()
    if east and west:
        print("PREDICTION HELD: morning recovered from the east, evening from the west.")
        print("Stage B renders real terrain into the correct illumination direction.")
    else:
        print("PREDICTION FAILED - check the azimuth convention before trusting Stage B.")
        print(f"  morning eastern? {east}   evening western? {west}")
    return 0 if (east and west) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default=None)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--no-shadows", action="store_true")
    args = parser.parse_args()
    return sweep(args.tile, args.size, args.step, shadows=not args.no_shadows)


if __name__ == "__main__":
    raise SystemExit(main())
