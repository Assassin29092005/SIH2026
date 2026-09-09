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
            manifest.append(asdict(PairSpec(
                tile=tile, row=row, col=col, size=size,
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
