"""Illumination normalisation.

The Moon has no atmosphere, so no fill light. Surface brightness is dominated by
shading rather than albedo, and shading inverts when the sun moves: a crater lit
from the east is close to pixel-identical to a dome lit from the west. Two
Kaguya images of identical ground, morning versus evening, correlate at -0.560.

Local contrast normalisation is what survived controlled measurement. Subtract a
local mean, divide by a local standard deviation, and match the residual
structure. No DEM is involved, which is the point: anything derived from a
shared DEM injects identical structure into both images, and the matcher locks
onto that instead of the terrain (BUGS.md BUG-011).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def stretch8(img: np.ndarray) -> np.ndarray:
    """Percentile stretch to 8-bit. LoFTR and SIFT both want uint8."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def local_contrast(img: np.ndarray, kernel: int | None = None) -> np.ndarray:
    """Subtract a local mean, divide by a local standard deviation.

    Halves cross-window offset scatter versus raw (3.02 -> 0.85 px), raises the
    inlier ratio from 36% to 54%, and doubles coverage.
    """
    k = kernel or config.get().contrast_kernel
    x = img.astype(np.float32)
    mu = cv2.blur(x, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(x * x, (k, k)) - mu * mu, 1e-6))
    return stretch8((x - mu) / sd)


def prepare(img: np.ndarray, kernel: int | None = None) -> np.ndarray:
    """The normalisation the pipeline actually uses."""
    return local_contrast(img, kernel)
