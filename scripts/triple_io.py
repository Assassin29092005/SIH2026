"""Read a Kaguya TC triple as aligned (morning, evening, dem) arrays.

Because all three products share one map grid (see CLAUDE.md), a window read at
the same row/col offsets returns exactly co-located pixels. There is no
registration step here by design -- that is the whole point of using these
tiles as the Stage-B training source.

    python scripts/triple_io.py                 # self-check on whatever is downloaded
    python scripts/triple_io.py --tile N18E009N15E012SC --size 512
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "kaguya"

STEMS = {
    "morning": "TCO_MAPM04_",
    "evening": "TCO_MAPE04_",
    "dem": "DTM_MAP_02_",
}

# Kaguya TC DEM stores elevation in metres relative to a 1737.4 km sphere,
# with this value marking "no data". Images use 0 for null.
DEM_NULL = -32768


@dataclass
class Triple:
    """One tile's morning image, evening image and DEM, on a shared grid."""

    tile: str
    paths: dict[str, Path]
    width: int
    height: int
    transform: object
    crs: object

    def read_window(self, row: int, col: int, size: int) -> dict[str, np.ndarray]:
        """Read the same size x size window from all three products."""
        if row < 0 or col < 0 or row + size > self.height or col + size > self.width:
            raise ValueError(
                f"window ({row},{col},{size}) outside {self.height}x{self.width}"
            )

        window = rasterio.windows.Window(col, row, size, size)
        out = {}
        for kind, path in self.paths.items():
            with rasterio.open(path) as src:
                band = src.read(1, window=window)
            # The DEM is signed 16-bit; the images are unsigned. GDAL honours the
            # label's SAMPLE_TYPE, so this asserts the driver got it right rather
            # than silently trusting it.
            if kind == "dem":
                assert np.issubdtype(band.dtype, np.signedinteger) or np.issubdtype(
                    band.dtype, np.floating
                ), f"DEM read as {band.dtype}; expected signed (see CLAUDE.md)"
                band = np.where(band == DEM_NULL, np.nan, band).astype(np.float32)
            out[kind] = band
        return out


def find_tiles() -> list[str]:
    """Tile codes that have all three .IMG files present locally."""
    if not RAW.exists():
        return []
    tiles = []
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        if all((d / f"{stem}{d.name}.IMG").exists() for stem in STEMS.values()):
            tiles.append(d.name)
    return tiles


def open_triple(tile: str) -> Triple:
    """Open a downloaded triple, verifying the three grids actually agree."""
    paths, grids = {}, {}
    for kind, stem in STEMS.items():
        # GDAL's PDS driver reads the detached .LBL and pulls in the .IMG beside it.
        label = RAW / tile / f"{stem}{tile}.LBL"
        if not label.exists():
            raise FileNotFoundError(f"missing {label}")
        paths[kind] = label
        with rasterio.open(label) as src:
            grids[kind] = (src.width, src.height, src.transform, src.crs)

    shapes = {(g[0], g[1]) for g in grids.values()}
    assert len(shapes) == 1, f"grid shapes disagree across products: {grids}"

    width, height, transform, crs = grids["morning"]
    return Triple(tile, paths, width, height, transform, crs)


def demo(tile: str | None = None, size: int = 512) -> int:
    """Self-check: open a triple, read an aligned window, sanity-check the data."""
    tiles = find_tiles()
    if not tiles:
        print(f"No complete triples under {RAW}")
        print("Run: python scripts/kaguya.py fetch --site equatorial")
        return 1

    tile = tile or tiles[0]
    print(f"tile: {tile}   (available: {len(tiles)})")

    triple = open_triple(tile)
    print(f"grid: {triple.width} x {triple.height}   crs={triple.crs}")

    # Centre window - avoids tile edges, which can be null-filled.
    row = (triple.height - size) // 2
    col = (triple.width - size) // 2
    bands = triple.read_window(row, col, size)

    print(f"\nwindow ({row},{col}) {size}x{size}:")
    for kind, arr in bands.items():
        finite = arr[np.isfinite(arr)]
        print(
            f"  {kind:<8} {str(arr.dtype):<8} shape={arr.shape} "
            f"min={finite.min():.1f} max={finite.max():.1f} mean={finite.mean():.1f}"
        )

    morning, evening = bands["morning"], bands["evening"]
    dem = bands["dem"]

    assert morning.shape == evening.shape == dem.shape, "window shapes differ"

    # The pair must be genuinely different images. Identical arrays would mean we
    # fetched the same product twice and the whole illumination premise is void.
    assert not np.array_equal(morning, evening), (
        "morning and evening windows are identical - check the product stems"
    )

    # ...but of the same terrain, so raw intensities must be strongly RELATED.
    #
    # The sign is expected to be NEGATIVE. Morning light comes from the east and
    # evening light from the west, so slopes lit at dawn are shadowed at dusk.
    # Anti-correlation is the crater/dome inversion measured directly, and it is
    # the reason raw-intensity matching fails across illumination. Asserting a
    # positive correlation here would assume away the entire problem.
    a = morning.astype(np.float64).ravel()
    b = evening.astype(np.float64).ravel()
    if a.std() > 0 and b.std() > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
        sense = "anti-correlated (expected)" if corr < 0 else "positively correlated"
        print(f"\n  morning/evening correlation: {corr:+.3f}  <- {sense}")
        assert abs(corr) > 0.2, (
            f"|correlation| {abs(corr):.3f} too low - the two windows look unrelated, "
            "so they are probably not the same tile"
        )

    relief = np.nanmax(dem) - np.nanmin(dem)
    print(f"  DEM relief across window: {relief:.1f} m")
    assert relief > 1.0, "DEM window is flat - suspicious, check the signed read"

    print("\nOK - triple opens, windows align, pair differs but correlates.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default=None)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    return demo(args.tile, args.size)


if __name__ == "__main__":
    raise SystemExit(main())
