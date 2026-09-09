"""Training-pair generation for fine-tuning the matcher on lunar imagery.

WHAT THIS TARGETS, AND WHY IT IS NOT ILLUMINATION
-------------------------------------------------
The obvious fine-tuning target was cross-illumination: train on Kaguya
morning/evening pairs, which is the project's headline difficulty. That is not
supportable, and the reason is measurable.

Morning and evening share a map grid but are independently orthorectified, so
their true correspondence is identity + a translation D that has to be
estimated. Estimating D with phase correlation -- the only method here
independent of the matcher being trained -- gives a median half-vs-half
disagreement of **5.16 px** on 1024 windows (mean 12.33, correlation response
0.06-0.31). The pipeline being trained already achieves **0.513-0.796 px**.
Labels an order of magnitude coarser than the model would teach it our noise.
Using the pipeline's own output as labels instead is circular.

So illumination is left alone: it already works and is control-gated.

Viewpoint is the opposite case. It is the weakest measured axis -- matching
fails beyond a ~12.3 deg emission-angle gap on real LROC NAC pairs -- and it is
the one where exact labels are free, because a homography applied to a real
image yields correspondence that is exact by construction rather than estimated.

GROUND TRUTH AND THE PROJECT RULE
---------------------------------
CLAUDE.md bans warping an image to manufacture correspondence, and allows
"rotation/crop/photometric jitter on real images" as augmentation. The
distinction is what the pairs are used for:

* TRAINING on warped real imagery: allowed, and standard practice.
* EVALUATION: never. Every reported number stays on real pairs, behind the
  four-control gate, on tiles held out from training.

Source pixels are always genuine observations. Only the geometry is imposed,
and only for training.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PairSpec:
    """One training pair: a real crop, a warped copy, and the exact homography."""
    tile: str
    row: int
    col: int
    size: int
    homography: list          # 3x3, maps source pixel -> warped pixel
    tilt_deg: float           # the obliquity this warp emulates
    rotation_deg: float
    scale: float


def perspective_homography(size: int, tilt_deg: float, rotation_deg: float = 0.0,
                           scale: float = 1.0, rng=None,
                           focal_ratio: float = 3.0) -> np.ndarray:
    """Homography for viewing a ground plane tilted `tilt_deg` off nadir.

    Derived from a pinhole camera looking at a rotated plane, not from
    hand-tuned matrix entries. That matters because the curriculum sweeps
    `tilt_deg` against a measured obliquity ceiling, so the parameter has to
    mean what it says: mean foreshortening must come out as cos(tilt), which is
    the dominant real effect (at 58.9 deg emission the ground compresses to
    0.517 in the tilt direction).

    An earlier hand-rolled version divided both axes by a perspective factor,
    which partly cancelled the foreshortening: it produced an area ratio of
    0.713 at 58.9 deg where cos gives 0.517, so the tilt label was wrong.

    `focal_ratio` sets camera distance as a multiple of the image size. Larger
    is closer to orthographic (pure foreshortening, little perspective).
    """
    rng = rng or np.random.default_rng()
    t = np.radians(tilt_deg)
    azimuth = rng.uniform(0, 2 * np.pi)

    half = size / 2.0
    src = np.array([[-half, -half], [half, -half], [half, half], [-half, half]],
                   dtype=np.float64)

    # Rotate the ground plane about an in-plane axis at `azimuth` (Rodrigues).
    u = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R3 = np.eye(3) * np.cos(t) + np.sin(t) * K + (1 - np.cos(t)) * np.outer(u, u)

    d = focal_ratio * size          # camera distance == focal length, so t=0 is identity
    pts3 = np.column_stack([src, np.zeros(len(src))]) @ R3.T
    pts3[:, 2] += d
    dst = d * pts3[:, :2] / pts3[:, 2:3]

    r = np.radians(rotation_deg)
    Rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    dst = (dst @ Rot.T) * scale

    src_px = (src + half).astype(np.float32)
    dst_px = (dst + half).astype(np.float32)
    return cv2.getPerspectiveTransform(src_px, dst_px).astype(np.float64)


def warp_pair(img: np.ndarray, H: np.ndarray, size: int | None = None):
    """Apply H to a real crop. Returns (warped, valid_mask)."""
    h, w = img.shape[:2]
    out_size = (size or w, size or h)
    warped = cv2.warpPerspective(img.astype(np.float32), H, out_size,
                                 flags=cv2.INTER_CUBIC, borderValue=0)
    mask = cv2.warpPerspective(np.ones_like(img, np.float32), H, out_size,
                               flags=cv2.INTER_NEAREST, borderValue=0) > 0.5
    return warped, mask


def correspondences(H: np.ndarray, size: int, step: int = 8,
                    mask: np.ndarray | None = None):
    """Exact dense correspondences implied by H, on a `step` lattice.

    This is the training signal. It is exact because H was applied, not
    estimated -- which is the entire reason viewpoint is the tractable
    fine-tuning target and illumination is not.
    """
    ys, xs = np.mgrid[step // 2:size:step, step // 2:size:step]
    pa = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    pb = cv2.perspectiveTransform(pa.reshape(-1, 1, 2), H).reshape(-1, 2)

    inside = ((pb[:, 0] >= 0) & (pb[:, 0] < size) &
              (pb[:, 1] >= 0) & (pb[:, 1] < size))
    if mask is not None:
        ix = np.clip(pb[:, 0].astype(int), 0, size - 1)
        iy = np.clip(pb[:, 1].astype(int), 0, size - 1)
        inside &= mask[iy, ix]
    return pa[inside], pb[inside]


def tilt_curriculum(n: int, max_tilt: float = 45.0, rng=None) -> np.ndarray:
    """Tilts biased toward the range where matching currently fails.

    Measured envelope: a 12.3 deg emission gap matches, 14.9 deg and beyond does
    not. Concentrating samples just past that boundary spends capacity where the
    model is actually weak instead of on cases it already handles.
    """
    rng = rng or np.random.default_rng()
    # Half easy (0-15), half in the failure band (15-max).
    easy = rng.uniform(0.0, 15.0, n // 2)
    hard = rng.uniform(15.0, max_tilt, n - n // 2)
    return np.concatenate([easy, hard])


def build_pairs(windows, out_dir: Path, *, size: int = 512, per_window: int = 4,
                max_tilt: float = 45.0, seed: int = 0, min_matches: int = 64):
    """Write training pairs from real image windows.

    `windows` yields (tile_name, row, col, image). Images must be genuine
    observations; only the geometry is imposed.
    """
    out_dir = Path(out_dir)
    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = []

    for tile, row, col, img in windows:
        if img.shape[0] < size or img.shape[1] < size:
            continue
        crop = img[:size, :size].astype(np.float32)
        tilts = tilt_curriculum(per_window, max_tilt, rng)
        for i, tilt in enumerate(tilts):
            rot = float(rng.uniform(-180, 180))
            scl = float(rng.uniform(0.8, 1.25))
            H = perspective_homography(size, float(tilt), rot, scl, rng)
            warped, mask = warp_pair(crop, H, size)
            pa, pb = correspondences(H, size, step=8, mask=mask)
            if len(pa) < min_matches:
                continue

            name = f"{tile}_{row}_{col}_{i:02d}"
            np.savez_compressed(
                out_dir / "pairs" / f"{name}.npz",
                image0=crop.astype(np.float32),
                image1=warped.astype(np.float32),
                mask1=mask,
                H=H, pa=pa, pb=pb,
            )
            # int() on row/col is load-bearing: generators that derive positions
            # from numpy arrays yield np.int64, which json.dumps refuses. The
            # pairs are written before the manifest, so this failed only after
            # ~10 minutes of work with every .npz already on disk (BUG-018).
            manifest.append(asdict(PairSpec(
                tile=tile, row=int(row), col=int(col), size=size,
                homography=H.tolist(), tilt_deg=float(tilt),
                rotation_deg=rot, scale=scl)) | {"name": name,
                                                 "n_correspondences": int(len(pa))})

    (out_dir / "manifest.json").write_text(
        json.dumps({"count": len(manifest), "size": size,
                    "note": "TRAINING ONLY. Warped geometry on real pixels. "
                            "Never use for evaluation - see module docstring.",
                    "pairs": manifest}, indent=2), encoding="utf-8")
    return manifest


def kaguya_windows(tiles, per_tile: int = 24, size: int = 512, seed: int = 0):
    """Sample real windows from downloaded Kaguya tiles. Generator."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from triple_io import open_triple

    rng = np.random.default_rng(seed)
    for tile in tiles:
        t = open_triple(tile)
        for _ in range(per_tile):
            row = int(rng.integers(0, t.height - size))
            col = int(rng.integers(0, t.width - size))
            band = t.read_window(row, col, size)
            for kind in ("morning", "evening"):
                img = band[kind].astype(np.float32)
                if img.std() < 1.0:        # blank or saturated
                    continue
                yield f"{tile}-{kind}", row, col, img


def nac_windows(per_file: int = 60, size: int = 512, seed: int = 0, skip: int = 2):
    """Real windows from LROC NAC strips. Generator.

    `skip` drops the first N files in sorted order. That is not arbitrary:
    `scripts/viewpoint.py:nac_pair()` takes `sorted(NAC_DIR.glob("*.IMG"))[:2]`
    as the obliquity ladder, which is the acceptance criterion this training is
    trying to move. Training on those two would make the ladder score itself.
    """
    import rasterio

    rng = np.random.default_rng(seed)
    files = sorted((Path(__file__).resolve().parent.parent /
                    "data" / "raw" / "nac").glob("*.IMG"))[skip:]
    for f in files:
        with rasterio.open(f) as s:
            h, w = s.height, s.width
            for _ in range(per_file):
                row = int(rng.integers(0, h - size))
                col = int(rng.integers(0, w - size))
                img = s.read(1, window=rasterio.windows.Window(col, row, size, size))
                img = img.astype(np.float32)
                if img.std() < 1.0 or not np.isfinite(img).all():
                    continue
                yield f"nac-{f.stem}", row, col, img


def _swath_centre(src, probes: int = 7):
    """Fit the imaged swath's column centre as a function of row.

    A TMC-2 ortho is a map-projected orbital strip, so the imaged data is a
    diagonal band inside a much larger bounding rectangle: measured on
    `ch2_tmc_ndn_20201126T1610528086`, the valid columns run 5634..10021 near the
    top and 421..4837 near the bottom, roughly 4000 px wide throughout. Sampling
    (row, col) uniformly therefore lands in zero padding almost every time --
    three random windows in a row came back empty, which is what made the first
    version of this generator yield nothing at all.

    Returns (centre_fn, half_width).
    """
    rows, centres, widths = [], [], []
    for frac in np.linspace(0.05, 0.95, probes):
        r = int(frac * src.height)
        band = src.read(1, window=rasterio_window(0, r, src.width, 8))
        idx = np.flatnonzero(band.max(axis=0) > 0)
        if idx.size < 512:
            continue
        rows.append(r)
        centres.append(0.5 * (idx[0] + idx[-1]))
        widths.append(idx[-1] - idx[0])
    if len(rows) < 2:
        raise RuntimeError("could not locate the TMC-2 swath")
    slope, intercept = np.polyfit(rows, centres, 1)
    return (lambda r: slope * r + intercept), int(min(widths) // 2)


def rasterio_window(col, row, w, h):
    from rasterio.windows import Window
    return Window(col, row, w, h)


def tmc_windows(n: int = 240, size: int = 512, seed: int = 0, block: int = 1536):
    """Real windows from the Chandrayaan-2 TMC-2 ortho strip. Generator.

    3.55 Gpx on one strip, read in place through /vsizip/. The largest genuine
    source on disk, and the only training source from the PS's own spacecraft:
    the task is Chandrayaan-2 imagery, so Chandrayaan-2 texture is closer to the
    deployment domain than Kaguya is.

    Reads a few large blocks and cuts crops out of them rather than reading each
    crop separately. The file is striped one row per block and uncompressed, so a
    512-px-wide window still touches 512 strips; one 1536 block costs about what
    one 512 window costs, and yields nine.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import rasterio
    from ch2_io import open_tmc_pair

    rng = np.random.default_rng(seed)
    t = open_tmc_pair()
    per_block = (block // size) ** 2
    n_blocks = max(1, int(np.ceil(n / per_block)))

    with rasterio.open(t.ortho_path) as src:
        centre_of, half = _swath_centre(src)
        # Spread blocks evenly down the strip: adjacent crops share terrain, so
        # diversity comes from where the blocks sit, not from how many crops
        # each yields.
        for br in np.linspace(0, src.height - block, n_blocks).astype(int):
            c0 = int(centre_of(br + block / 2) - half +
                     rng.integers(0, max(1, 2 * half - block)))
            c0 = int(np.clip(c0, 0, src.width - block))
            data = src.read(1, window=rasterio_window(c0, br, block, block))
            for dy in range(0, block - size + 1, size):
                for dx in range(0, block - size + 1, size):
                    img = data[dy:dy + size, dx:dx + size].astype(np.float32)
                    if img.std() < 1.0 or not np.isfinite(img).all():
                        continue          # clipped corner of the diagonal swath
                    # Reject crops holding any real amount of the swath's black
                    # padding. A straight high-contrast edge is the strongest
                    # feature in an otherwise texture-poor lunar crop, and it
                    # survives the warp -- so the matcher would learn to align
                    # the padding instead of the terrain. That is the failure
                    # mode of BUG-011, reintroduced through the training set.
                    if (img <= 0).mean() > 0.02:
                        continue
                    yield "tmc2", br + dy, c0 + dx, img


def safe_windows(seed: int = 0, size: int = 512):
    """Every training window that is disjoint from all evaluation data.

    The first fine-tune trained on Kaguya tile N03E021N00E024SC, which is also
    where the OHRC sample's *reference* image comes from (~0.6N 23.4E lies
    inside 0-3N, 21-24E). The other tile, N18E009N15E012SC, is the Kaguya sample
    itself. So both downloaded Kaguya tiles are evaluation data and neither may
    be trained on. See BUGS.md BUG-017.

    What is left, and why it is enough:

    | source            | pixels   | status                                  |
    |-------------------|----------|-----------------------------------------|
    | TMC-2 ortho       | 3.55 Gpx | train — Chandrayaan-2, no eval use      |
    | LROC NAC [2:]     | 1.06 Gpx | train — ladder pair excluded            |
    | Kaguya (2 tiles)  | 0.30 Gpx | EVAL — both sample cases                |
    | OHRC              | 0.94 Gpx | EVAL ONLY per the curriculum            |

    4.6 Gpx across two sensors, against 0.026 Gpx of one tile last time.
    """
    yield from tmc_windows(seed=seed, size=size)
    yield from nac_windows(seed=seed + 1, size=size)


def coarse_assignment(H: np.ndarray, size: int, stride: int = 8):
    """Ground-truth coarse-cell assignment implied by H.

    LoFTR supervises a confidence matrix over coarse cells (stride 8), so the
    exact pixel correspondences have to be reduced to "cell i in image0 matches
    cell j in image1". Returns (i_idx, j_idx) into a flattened (S/stride)^2
    grid, keeping only cells whose centre lands inside image1.

    A cell is assigned by where its CENTRE maps. That is deliberate: assigning
    by overlap would spread a single source cell across several targets and make
    the supervision ambiguous exactly where the warp is strongest.
    """
    g = size // stride
    ys, xs = np.mgrid[0:g, 0:g]
    centres = np.column_stack([
        (xs.ravel() + 0.5) * stride,
        (ys.ravel() + 0.5) * stride,
    ]).astype(np.float64)

    mapped = cv2.perspectiveTransform(centres.reshape(-1, 1, 2), H).reshape(-1, 2)
    cj = np.floor(mapped[:, 0] / stride).astype(int)
    ri = np.floor(mapped[:, 1] / stride).astype(int)
    ok = (cj >= 0) & (cj < g) & (ri >= 0) & (ri < g)

    i_idx = np.arange(g * g)[ok]
    j_idx = (ri[ok] * g + cj[ok])
    return i_idx, j_idx, g
