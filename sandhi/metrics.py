"""Evaluation metrics.

The problem statement asks for RMSE, inlier count and inlier ratio, plus match
points "maintaining uniform distribution across the images". Uniformity is a
first-class metric here because it is an explicit requirement, and because a
transform fitted from points clustered in one corner is tightly constrained
there and extrapolates badly everywhere else.

None of these assume a ground-truth correspondence. Kaguya morning and evening
are independently orthorectified and genuinely offset by 8.25 px, so identity
correspondence is false; scoring is against a fitted model instead.
"""

from __future__ import annotations

import numpy as np

from . import config


def uniformity(pts: np.ndarray, shape: tuple[int, int],
               grid: int | None = None) -> tuple[float, float]:
    """Grid coverage fraction and normalised entropy of the match distribution.

    Coverage is count-limited: with n matches on a grid x grid lattice the
    ceiling is min(1, n / grid**2). Report the match count alongside it.
    """
    g = grid or config.get().uniformity_grid
    if len(pts) == 0:
        return 0.0, 0.0
    h, w = shape
    gy = np.clip((pts[:, 1] / h * g).astype(int), 0, g - 1)
    gx = np.clip((pts[:, 0] / w * g).astype(int), 0, g - 1)
    counts = np.bincount(gy * g + gx, minlength=g * g).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float((counts > 0).mean()), float(-(nz * np.log(nz)).sum() / np.log(g * g))


def coverage_grid(pts: np.ndarray, shape: tuple[int, int],
                  grid: int | None = None) -> np.ndarray:
    """Per-cell match counts, for plotting."""
    g = grid or config.get().uniformity_grid
    h, w = shape
    if len(pts) == 0:
        return np.zeros((g, g), int)
    gy = np.clip((pts[:, 1] / h * g).astype(int), 0, g - 1)
    gx = np.clip((pts[:, 0] / w * g).astype(int), 0, g - 1)
    return np.bincount(gy * g + gx, minlength=g * g).reshape(g, g)


def offset_of(pa: np.ndarray, pb: np.ndarray):
    """Robust translation between two match sets, with its dispersion.

    Median rather than mean: a handful of bad matches should not move the
    estimate. Returns None when there are too few matches to be meaningful.
    """
    if len(pa) < config.get().min_matches:
        return None
    dx = pb[:, 0] - pa[:, 0]
    dy = pb[:, 1] - pa[:, 1]
    return (float(np.median(dx)), float(np.median(dy)),
            float(np.median(np.abs(dx - np.median(dx)))),
            float(np.median(np.abs(dy - np.median(dy)))))


def summarise(pa, pb, inliers, resid, shape, gsd_m: float | None = None,
              source_gsd_m: float | None = None) -> dict:
    """The metric block reported everywhere and written to metrics.json.

    `gsd_m` is the COMMON grid both images were resampled onto; `source_gsd_m`
    is the source product's native resolution. They are usually different and
    the distinction is the deliverable — see the source-pixel block below.
    """
    n = len(pa)
    n_in = int(inliers.sum()) if n else 0
    rmse = float(np.sqrt((resid[inliers] ** 2).mean())) if n_in else float("nan")
    cov, ent = uniformity(pa[inliers], shape) if n_in else (0.0, 0.0)
    out = {
        "match_count": n,
        "inlier_count": n_in,
        "inlier_ratio": float(inliers.mean()) if n else 0.0,
        "rmse_px": rmse,
        "sub_pixel": bool(rmse == rmse and rmse < 1.0),  # NaN-safe
        "coverage": cov,
        "entropy": ent,
    }
    if gsd_m:
        out["rmse_m"] = rmse * gsd_m
        out["gsd_m"] = gsd_m
    if gsd_m and source_gsd_m:
        # The problem statement asks for "sub-pixel accuracy OF SOURCE IMAGE",
        # so the metric has to be stated in the source product's own pixels, not
        # in the common grid both images were resampled onto. The two differ by
        # the scale ratio and the gap can be large: 0.513 common-grid px at
        # 7.403 m is 3.80 m, which is 0.75 TMC-2 pixels but 14.6 OHRC pixels.
        #
        # Reporting only the common-grid figure leaves a reader unable to check
        # the PS clause at all, and reporting it AS the source-pixel figure
        # would overclaim by the scale ratio. Both are written. See BUG-025.
        src_px = rmse * gsd_m / source_gsd_m
        out["source_gsd_m"] = source_gsd_m
        out["rmse_source_px"] = src_px
        out["sub_pixel_source"] = bool(src_px == src_px and src_px < 1.0)
    return out


def cross_window_spread(offsets) -> float:
    """Scatter of the recovered offset across independent windows.

    This is the column an artifact cannot fake. A mask artifact reports zero
    displacement everywhere; noise reports scatter. Agreement across windows on
    a non-zero offset is the evidence that terrain is being measured.
    """
    a = np.asarray(offsets, float)
    if len(a) < 2:
        return float("nan")
    return float(np.hypot(a[:, 0].std(), a[:, 1].std()))
