"""Project a raw OHRC strip into a map frame so it can be matched.

OHRC ships unprojected: `crs` is None, no geotransform. Its only geolocation is
a sidecar CSV sampling Longitude/Latitude on a 100-pixel lattice (121 x 783 =
94,743 points for this strip). To match against a map-projected reference the
strip has to be resampled into that reference's grid.

Rather than a full sensor model, fit a bivariate polynomial from (lon, lat) to
(pixel, scan) and use it to inverse-map each output pixel back into the strip.
The fit residual is reported and asserted -- a geolocation model nobody measured
is a silent source of registration error, and it would be indistinguishable from
a matcher failure downstream.

    python scripts/ohrc_project.py                 # fit quality only
    python scripts/ohrc_project.py --gsd 7.403     # project to a Kaguya-like grid
"""

import argparse

import numpy as np

from ch2_io import open_ohrc

MOON_RADIUS_M = 1737400.0
DEFAULT_DEGREE = 3

# A geolocation model worse than this makes sub-pixel registration meaningless,
# since the error floor would already exceed the target.
MAX_FIT_RESIDUAL_PX = 5.0


def _design(lon: np.ndarray, lat: np.ndarray, degree: int) -> np.ndarray:
    """Bivariate polynomial design matrix, centred to keep conditioning sane."""
    x = (lon - _design.lon0) / _design.lon_scale
    y = (lat - _design.lat0) / _design.lat_scale
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append((x**i) * (y**j))
    return np.column_stack(cols)


def fit_inverse(lon, lat, pixel, scan, degree: int = DEFAULT_DEGREE):
    """Fit (lon, lat) -> (pixel, scan). Returns (coeffs, residual_px, degree)."""
    # Centre and scale, or the higher-order terms blow up the condition number.
    _design.lon0, _design.lat0 = float(lon.mean()), float(lat.mean())
    _design.lon_scale = max(float(lon.std()), 1e-9)
    _design.lat_scale = max(float(lat.std()), 1e-9)

    A = _design(lon, lat, degree)
    coef_p, *_ = np.linalg.lstsq(A, pixel, rcond=None)
    coef_s, *_ = np.linalg.lstsq(A, scan, rcond=None)

    pred_p, pred_s = A @ coef_p, A @ coef_s
    residual = float(np.sqrt(((pred_p - pixel) ** 2 + (pred_s - scan) ** 2).mean()))
    return (coef_p, coef_s), residual, degree


def apply_inverse(coeffs, lon, lat, degree: int = DEFAULT_DEGREE):
    """Map (lon, lat) to fractional (pixel, scan)."""
    A = _design(np.asarray(lon, float), np.asarray(lat, float), degree)
    coef_p, coef_s = coeffs
    return A @ coef_p, A @ coef_s


def target_grid(ohrc, gsd_m: float, margin: float = 0.0):
    """A north-up lon/lat grid at `gsd_m`, covering the strip."""
    west, east, south, north = ohrc.bounds()
    west, east = west - margin, east + margin
    south, north = south - margin, north + margin

    deg_per_px = np.degrees(gsd_m / MOON_RADIUS_M)
    width = int(np.ceil((east - west) / deg_per_px))
    height = int(np.ceil((north - south) / deg_per_px))
    return dict(west=west, north=north, deg_per_px=deg_per_px,
                width=width, height=height)


def project(ohrc, coeffs, grid, degree: int = DEFAULT_DEGREE,
            chunk: int = 256) -> np.ndarray:
    """Resample the strip onto `grid` by nearest neighbour.

    Nearest neighbour is deliberate: the output GSD is far coarser than the
    0.26 m source, so this is decimation, and interpolating first would only
    blur detail that decimation discards anyway.
    """
    import rasterio

    out = np.zeros((grid["height"], grid["width"]), dtype=np.uint8)
    cols = np.arange(grid["width"])

    with rasterio.open(ohrc.path) as src:
        # Read the whole strip once; 938 MB at 8-bit is fine, and windowed
        # reads inside a zip are far slower than one sequential pass.
        strip = src.read(1)

    for row0 in range(0, grid["height"], chunk):
        rows = np.arange(row0, min(row0 + chunk, grid["height"]))
        lon = grid["west"] + (cols[None, :] + 0.5) * grid["deg_per_px"]
        lat = grid["north"] - (rows[:, None] + 0.5) * grid["deg_per_px"]
        lon_f = np.broadcast_to(lon, (len(rows), grid["width"])).ravel()
        lat_f = np.broadcast_to(lat, (len(rows), grid["width"])).ravel()

        px, sc = apply_inverse(coeffs, lon_f, lat_f, degree)
        px = np.rint(px).astype(np.int64)
        sc = np.rint(sc).astype(np.int64)
        inside = (px >= 0) & (px < ohrc.width) & (sc >= 0) & (sc < ohrc.height)

        block = np.zeros(len(rows) * grid["width"], dtype=np.uint8)
        block[inside] = strip[sc[inside], px[inside]]
        out[rows[0]:rows[-1] + 1] = block.reshape(len(rows), grid["width"])

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsd", type=float, default=None,
                        help="project to this ground sample distance, in metres")
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    parser.add_argument("--out", default="data/interim/ohrc_projected.tif")
    args = parser.parse_args()

    ohrc = open_ohrc()
    print(f"OHRC {ohrc.width} x {ohrc.height}, {len(ohrc.lon)} geolocation samples")

    print("\ngeolocation fit, (lon,lat) -> (pixel,scan):")
    best = None
    for degree in (1, 2, 3, 4):
        _, residual, _ = fit_inverse(ohrc.lon, ohrc.lat, ohrc.pixel, ohrc.scan, degree)
        flag = ""
        if best is None or residual < best[1]:
            best = (degree, residual)
            flag = "  <- best so far"
        print(f"  degree {degree}: RMS residual {residual:8.3f} px "
              f"({residual*0.26:7.3f} m){flag}")

    degree = args.degree
    coeffs, residual, _ = fit_inverse(ohrc.lon, ohrc.lat, ohrc.pixel, ohrc.scan, degree)
    print(f"\nusing degree {degree}: {residual:.3f} px = {residual*0.26:.3f} m on the ground")
    assert residual < MAX_FIT_RESIDUAL_PX, (
        f"geolocation fit residual {residual:.2f} px exceeds {MAX_FIT_RESIDUAL_PX} px; "
        "a polynomial is not modelling this strip well enough to register with"
    )

    if args.gsd is None:
        print("\nOK - geolocation model usable. Pass --gsd to project.")
        return 0

    grid = target_grid(ohrc, args.gsd)
    print(f"\nprojecting to {grid['width']} x {grid['height']} at {args.gsd} m/px")
    img = project(ohrc, coeffs, grid, degree)

    filled = float((img > 0).mean())
    print(f"  filled {100*filled:.1f}% of the grid (the strip is diagonal, so "
          "the rest is outside it)")
    assert filled > 0.10, f"only {100*filled:.1f}% filled - projection likely wrong"

    import rasterio
    from rasterio.transform import from_origin

    out = ROOT_OUT = args.out
    transform = from_origin(grid["west"], grid["north"],
                            grid["deg_per_px"], grid["deg_per_px"])
    crs = ('GEOGCS["SelenoGraphic",DATUM["Moon",SPHEROID["Moon",1737400,0]],'
           'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]]')
    with rasterio.open(out, "w", driver="GTiff", width=grid["width"],
                       height=grid["height"], count=1, dtype="uint8",
                       crs=crs, transform=transform, nodata=0) as dst:
        dst.write(img, 1)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
