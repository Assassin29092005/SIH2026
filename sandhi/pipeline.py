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


def resample(img: np.ndarray, factor: float) -> np.ndarray:
    """Resample by an arbitrary factor. >1 shrinks, area-averaged.

    `decimate` handles integer factors; real cross-sensor ratios are rarely
    integers (Kaguya/TMC-2 = 1.466, NAC/TMC-2 = 0.099) and rounding one to 1
    silently skips the resampling entirely. See BUGS.md BUG-020.
    """
    if abs(factor - 1.0) < 1e-6:
        return img
    h = max(1, int(round(img.shape[0] / factor)))
    w = max(1, int(round(img.shape[1] / factor)))
    interp = cv2.INTER_AREA if factor > 1 else cv2.INTER_CUBIC
    return cv2.resize(img.astype(np.float32), (w, h), interpolation=interp)


def to_common_gsd(src: np.ndarray, ref: np.ndarray, src_gsd: float, ref_gsd: float):
    """Bring both images to the coarser ground sample distance.

    Both sides must be resampled BEFORE normalisation, not after: normalising at
    different resolutions produces incomparable products (correlation 0.56 at
    k=2, 0.27 at k=4, -0.09 at k=8 between the two orderings).

    The ratio is used as measured, not rounded to an integer. Rounding was
    invisible for a long time because every ratio exercised was integral by
    construction: the scale ladder steps 1/2/4/8/16x, and OHRC/Kaguya is 28.5x
    which rounds to 28 harmlessly. TMC-2/Kaguya is 1.466x, which rounded to 1
    and skipped the resampling altogether (BUGS.md BUG-020).
    """
    if not src_gsd or not ref_gsd or abs(src_gsd - ref_gsd) < 1e-9:
        return src, ref, 1.0
    if src_gsd < ref_gsd:
        k = ref_gsd / src_gsd
        return resample(src, k), ref, float(k)
    k = src_gsd / ref_gsd
    return src, resample(ref, k), 1.0 / k


def register(src: np.ndarray, ref: np.ndarray, *,
             src_gsd: float | None = None, ref_gsd: float | None = None,
             refine_subpixel: bool = True, tiled: bool = True,
             coarse: bool = True, select_model: bool = True,
             gsd_m: float | None = None,
             source_native_gsd: float | None = None):
    """Register `src` onto `ref`. Returns a result dict, or None if no fit.

    The returned dict carries the match points, the fitted model, per-match
    residuals, the metric block, and a `stages` record of how many matches
    survived each step — which is where coverage losses show up.

    `src_gsd` is the resolution of the array PASSED IN; `source_native_gsd` is
    the resolution of the product it came from. They differ whenever the source
    was projected before being handed over — `ohrc_project.py` turns a 0.26 m
    OHRC strip into a 7.403 m GeoTIFF, so `src_gsd` is 7.403 and the native
    figure is 0.26. The PS's "sub-pixel accuracy of source image" is stated
    against the native one, so conflating them overclaims by 28.5x. BUG-025.
    """
    cfg = config.get()
    stages: dict = {}

    # C -- common GSD, before any normalisation
    scale_k = 1.0
    common_gsd = gsd_m
    if src_gsd and ref_gsd:
        src, ref, scale_k = to_common_gsd(src, ref, src_gsd, ref_gsd)
        stages["scale_ratio"] = scale_k
        # `to_common_gsd` resamples to the COARSER of the two, so the grid the
        # residuals live on is max(src, ref), not the reference's GSD. A caller
        # passing `gsd_m=ref_gsd` is right only while the source is the finer
        # image; it is wrong for IIRS at 85.08 m against Kaguya at 7.403 m,
        # which would report every distance 11.5x too small. Derive it here so
        # no caller can get it wrong. See BUG-025.
        common_gsd = max(src_gsd, ref_gsd)
        stages["common_gsd_m"] = common_gsd

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

    m = metrics.summarise(pa, pb, f["inliers"], f["resid"], src.shape,
                          common_gsd, source_native_gsd or src_gsd)
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
