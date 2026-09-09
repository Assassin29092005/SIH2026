"""The registration pipeline: one function, six stages.

    A  coarse localisation      remove the gross offset
    B  normalisation            local contrast, no DEM
    C  common-GSD resampling    ratio read from metadata, never guessed
    D  dense matching           LoFTR, tiled for coverage
    E  sub-pixel refinement     phase correlation per match
    F  robust fit               RANSAC, model selected on held-out residual

Stage C only engages when a scale ratio is supplied; for a same-GSD pair it is a
no-op. The ratio is read from each product's declared resolution because blind
estimation is confounded — both sides normalised against a shared reference
inherit identical structure, so match count and matcher confidence cannot
discriminate it (BUGS.md BUG-010).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config, matching, metrics, models, normalize, refine


def decimate(img: np.ndarray, k: int) -> np.ndarray:
    """Downsample by an integer factor with area averaging.

    Area averaging, not point sampling: decimating ~28x with nearest-neighbour
    aliases the signal into noise, which is what made the first OHRC run fail
    its control gate.
    """
    if k == 1:
        return img
    h, w = img.shape[0] // k, img.shape[1] // k
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def to_common_gsd(src: np.ndarray, ref: np.ndarray, src_gsd: float, ref_gsd: float):
    """Bring both images to the coarser ground sample distance.

    Both sides must be resampled BEFORE normalisation, not after: normalising at
    different resolutions produces incomparable products (correlation 0.56 at
    k=2, 0.27 at k=4, -0.09 at k=8 between the two orderings).
    """
    if not src_gsd or not ref_gsd or abs(src_gsd - ref_gsd) < 1e-9:
        return src, ref, 1.0
    if src_gsd < ref_gsd:
        k = int(round(ref_gsd / src_gsd))
        return decimate(src, max(k, 1)), ref, float(k)
    k = int(round(src_gsd / ref_gsd))
    return src, decimate(ref, max(k, 1)), 1.0 / max(k, 1)


def register(src: np.ndarray, ref: np.ndarray, *,
             src_gsd: float | None = None, ref_gsd: float | None = None,
             refine_subpixel: bool = True, tiled: bool = True,
             coarse: bool = True, select_model: bool = True,
             gsd_m: float | None = None):
    """Register `src` onto `ref`. Returns a result dict, or None if no fit.

    The returned dict carries the match points, the fitted model, per-match
    residuals, the metric block, and a `stages` record of how many matches
    survived each step — which is where coverage losses show up.
    """
    cfg = config.get()
    stages: dict = {}

    # C -- common GSD, before any normalisation
    scale_k = 1.0
    if src_gsd and ref_gsd:
        src, ref, scale_k = to_common_gsd(src, ref, src_gsd, ref_gsd)
        stages["scale_ratio"] = scale_k

    # B -- normalisation
    a8 = normalize.prepare(src)
    b8 = normalize.prepare(ref)

    # A + D -- coarse align, then tiled match
    if tiled and coarse:
        pa, pb, conf, offset, n_coarse = matching.match_aligned(a8, b8)
        stages["coarse_offset"] = offset
        stages["coarse_matches"] = n_coarse
    elif tiled:
        pa, pb, conf = matching.match_tiled(a8, b8)
    else:
        pa, pb, conf = matching.match(a8, b8)
    stages["matched"] = len(pa)
    if len(pa) == 0:
        return None

    # E -- sub-pixel
    moved = 0.0
    if refine_subpixel:
        pa, pb, kept, moved = refine.refine(a8, b8, pa, pb)
        conf = conf[kept]          # select by index; never slice by length
        stages["refined"] = len(pa)
    if len(pa) < 4:
        return None

    # F -- robust fit
    chosen = None
    if select_model:
        sel = models.select(pa, pb)
        if sel:
            chosen = sel["best"]
            stages["model_selection"] = {
                k: {kk: vv for kk, vv in v.items()
                    if kk in ("dof", "held_out_median", "gain_over_simpler")}
                for k, v in sel["models"].items()}
    f = models.fit(pa, pb, kind=chosen or "similarity")
    if f is None:
        return None

    m = metrics.summarise(pa, pb, f["inliers"], f["resid"], src.shape, gsd_m)
    m["mean_refinement_shift_px"] = moved
    return {"pa": pa, "pb": pb, "conf": conf, "fit": f, "stages": stages,
            "metrics": m, "src": src, "ref": ref, "scale_ratio": scale_k}


def warp(src: np.ndarray, result: dict, shape: tuple[int, int] | None = None):
    """Apply the fitted model to produce the registered image."""
    f = result["fit"]
    h, w = shape or result["ref"].shape[:2]
    img = src.astype(np.float32)
    if f["kind"] == "homography":
        return cv2.warpPerspective(img, f["model"], (w, h), flags=cv2.INTER_CUBIC)
    return cv2.warpAffine(img, f["model"], (w, h), flags=cv2.INTER_CUBIC)
