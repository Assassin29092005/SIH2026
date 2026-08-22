"""Read Chandrayaan-2 OHRC and TMC-2 products, without unpacking them.

GDAL's /vsizip/ handler reads inside the archives directly, so the ~10 GB of
uncompressed imagery in the three downloaded products never has to hit disk.
Windowed reads work, at a couple of seconds each for OHRC.

Two very different product shapes:

**OHRC** -- raw strip, 12000 x 78175, 8-bit, 0.26 m/px (3.1 x 20.3 km). NOT map
projected: `crs` is None and there is no geotransform. Geolocation comes from
the sidecar CSV, which samples Longitude/Latitude on a Pixel/Scan grid.

**TMC-2 derived** -- map-projected GeoTIFF pairs on a SelenoGraphic sphere.
Ortho is 10380 x 341544 at ~5.05 m/px; the DTM covers identical bounds at
exactly half that, so ortho pixel (i, j) is DTM pixel (i/2, j/2).

Neither carries sun angles, so illumination must be recovered exactly as for
Kaguya -- see scripts/ablation.py.

    python scripts/ch2_io.py
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
CH2 = ROOT / "data" / "raw" / "ch2"

# TMC-2 DTM is exactly half the ortho resolution over identical bounds.
TMC_DTM_RATIO = 2


def vsizip(zip_path: Path, inner: str) -> str:
    """GDAL path addressing a file inside a zip, without extracting it."""
    return "/vsizip/" + str(zip_path).replace("\\", "/") + "/" + inner


def inner_named(zip_path: Path, suffix: str, prefix: str = "") -> str:
    """First entry in the archive matching a suffix (and optional path prefix)."""
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(suffix) and name.startswith(prefix):
                return name
    raise FileNotFoundError(f"no {prefix}*{suffix} in {zip_path.name}")


@dataclass
class Ohrc:
    """A raw OHRC strip plus its geolocation grid."""

    path: str
    width: int
    height: int
    lon: np.ndarray  # sampled longitudes
    lat: np.ndarray  # sampled latitudes
    pixel: np.ndarray  # column index of each sample
    scan: np.ndarray  # row index of each sample

    def read_window(self, row: int, col: int, size: int) -> np.ndarray:
        with rasterio.open(self.path) as src:
            return src.read(1, window=Window(col, row, size, size))

    def latlon_at(self, row: float, col: float) -> tuple[float, float]:
        """Nearest sampled geolocation. The CSV grid is coarse (every 100 px),
        so this locates a strip, it does not georeference a pixel."""
        d = (self.scan - row) ** 2 + (self.pixel - col) ** 2
        i = int(np.argmin(d))
        return float(self.lat[i]), float(self.lon[i])

    def bounds(self) -> tuple[float, float, float, float]:
        """(west, east, south, north) of the whole strip."""
        return (float(self.lon.min()), float(self.lon.max()),
                float(self.lat.min()), float(self.lat.max()))


def open_ohrc(zip_path: Path | None = None) -> Ohrc:
    zip_path = zip_path or next(CH2.glob("ch2_ohr_*.zip"))
    label = inner_named(zip_path, ".xml", prefix="data/")
    path = vsizip(zip_path, label)

    with rasterio.open(path) as src:
        width, height = src.width, src.height

    geom = inner_named(zip_path, ".csv", prefix="geometry/")
    with zipfile.ZipFile(zip_path) as z, z.open(geom) as fh:
        reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
        rows = [
            (float(r["Longitude"]), float(r["Latitude"]), float(r["Pixel"]), float(r["Scan"]))
            for r in reader
        ]
    arr = np.array(rows)
    return Ohrc(path, width, height, arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])


@dataclass
class TmcPair:
    """A TMC-2 ortho image and its co-registered DTM."""

    ortho_path: str
    dtm_path: str
    width: int
    height: int
    transform: object
    crs: object
    dtm_nodata: float

    def read_window(self, row: int, col: int, size: int) -> dict[str, np.ndarray]:
        """Ortho window plus the DTM over the same ground, upsampled to match."""
        with rasterio.open(self.ortho_path) as src:
            ortho = src.read(1, window=Window(col, row, size, size))

        k = TMC_DTM_RATIO
        half = size // k
        with rasterio.open(self.dtm_path) as src:
            dem = src.read(1, window=Window(col // k, row // k, half, half)).astype(np.float32)
        dem[dem == self.dtm_nodata] = np.nan

        # Nearest-neighbour upsample keeps this dependency-free; the DTM is
        # smoother than the ortho anyway, so interpolation buys little.
        dem_up = np.repeat(np.repeat(dem, k, axis=0), k, axis=1)[:size, :size]
        return {"ortho": ortho, "dem": dem_up, "dem_native": dem}

    def pixel_latlon(self, row: float, col: float) -> tuple[float, float]:
        lon, lat = self.transform * (col, row)
        return float(lat), float(lon)


def open_tmc_pair(oth_zip: Path | None = None, dtm_zip: Path | None = None) -> TmcPair:
    oth_zip = oth_zip or next(CH2.glob("ch2_tmc_*_d_oth_*.zip"))
    # Ortho and DTM product ids differ only by this token; see CLAUDE.md.
    dtm_zip = dtm_zip or Path(str(oth_zip).replace("_d_oth_", "_d_dtm_"))
    if not dtm_zip.exists():
        raise FileNotFoundError(f"no DTM beside {oth_zip.name}: expected {dtm_zip.name}")

    ortho_path = vsizip(oth_zip, inner_named(oth_zip, ".tif", prefix="data/"))
    dtm_path = vsizip(dtm_zip, inner_named(dtm_zip, ".tif", prefix="data/"))

    with rasterio.open(ortho_path) as so, rasterio.open(dtm_path) as sd:
        assert abs(so.width / sd.width - TMC_DTM_RATIO) < 0.01, (
            f"ortho/DTM width ratio {so.width/sd.width:.3f}, expected {TMC_DTM_RATIO}"
        )
        for a, b in zip(so.bounds, sd.bounds):
            assert abs(a - b) < 1e-6, f"ortho and DTM bounds differ: {so.bounds} vs {sd.bounds}"
        return TmcPair(ortho_path, dtm_path, so.width, so.height,
                       so.transform, so.crs, sd.nodata)


def demo() -> int:
    print("OHRC")
    ohrc = open_ohrc()
    west, east, south, north = ohrc.bounds()
    print(f"  {ohrc.width} x {ohrc.height}  8-bit raw strip")
    print(f"  geolocation samples: {len(ohrc.lon)}")
    print(f"  extent lon {west:.3f}..{east:.3f}   lat {south:.3f}..{north:.3f}")

    win = ohrc.read_window(40000, 5000, 512)
    lat, lon = ohrc.latlon_at(40000 + 256, 5000 + 256)
    print(f"  window (40000,5000) 512: min={win.min()} max={win.max()} "
          f"mean={win.mean():.1f} at {lat:.3f}N {lon:.3f}E")
    assert win.shape == (512, 512), "OHRC window wrong shape"
    assert win.std() > 1.0, "OHRC window is flat - blank region or read failure"

    print("\nTMC-2")
    tmc = open_tmc_pair()
    print(f"  ortho {tmc.width} x {tmc.height}")
    # Equator on this strip, which spans 28.5S to 28.4N.
    col, row = ~tmc.transform * (71.0, 0.0)
    bands = tmc.read_window(int(row) - 256, int(col) - 256, 512)
    ortho, dem = bands["ortho"], bands["dem"]
    finite = dem[np.isfinite(dem)]
    print(f"  ortho window: min={ortho.min()} max={ortho.max()} mean={ortho.mean():.0f}")
    print(f"  DEM window:   {finite.min():.0f}..{finite.max():.0f} m "
          f"(relief {finite.max()-finite.min():.0f} m, {100*np.isfinite(dem).mean():.0f}% valid)")

    assert ortho.shape == dem.shape == (512, 512), "ortho/DEM window shapes differ"
    assert np.isfinite(dem).mean() > 0.5, "DEM window mostly nodata"

    mask = np.isfinite(dem) & (ortho > 0)
    corr = float(np.corrcoef(ortho[mask].astype(np.float64), dem[mask])[0, 1])
    print(f"  corr(ortho, DEM) = {corr:+.3f}")
    assert abs(corr) > 0.1, (
        f"ortho and DEM look unrelated (r={corr:+.3f}) - check the 2:1 window mapping"
    )

    print("\nOK - both products readable in place, TMC ortho/DEM aligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(demo())
