"""Transform fitting and model selection.

Three model classes, increasing in freedom:

    similarity  4 DOF   shift, rotate, uniform scale
    affine      6 DOF   + shear, anisotropic scale
    homography  8 DOF   + perspective

Selection is on **held-out** residual, because a higher-DOF model always fits
the training points at least as well — picking on in-sample error would select
homography every time and prove nothing about the geometry.

Held-out alone was still not enough. On orthorectified nadir pairs, where no
perspective exists between the images, homography still won 2 of 3 windows by
5-9% — noise rewarding degrees of freedom. Hence `complexity_margin`: parsimony
is the default and perspective has to be earned by a decisive margin.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config

MODELS = {
    "similarity": 4,
    "affine": 6,
    "homography": 8,
}


def fit_model(kind: str, pa: np.ndarray, pb: np.ndarray, ransac_px: float | None = None):
    """Fit one model class. Returns (matrix, inlier_mask) or None."""
    thr = config.get().ransac_px if ransac_px is None else ransac_px
    a, b = pa.astype(np.float32), pb.astype(np.float32)
    if kind == "similarity":
        m, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                             ransacReprojThreshold=thr)
    elif kind == "affine":
        m, inl = cv2.estimateAffine2D(a, b, method=cv2.RANSAC,
                                      ransacReprojThreshold=thr)
    elif kind == "homography":
        m, inl = cv2.findHomography(a, b, method=cv2.RANSAC,
                                    ransacReprojThreshold=thr)
    else:
        raise ValueError(f"unknown model {kind!r}")
    if m is None or inl is None:
        return None
    return m, inl.ravel().astype(bool)


def apply(kind: str, m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if kind == "homography":
        p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), m)
        return p.reshape(-1, 2)
    return (m @ np.c_[pts, np.ones(len(pts))].T).T


def residual(kind: str, m: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    pred = apply(kind, m, pa)
    return np.hypot(pred[:, 0] - pb[:, 0], pred[:, 1] - pb[:, 1])


def fit(pa: np.ndarray, pb: np.ndarray, kind: str = "similarity",
        ransac_px: float | None = None):
    """Fit a single named model. Returns a dict, or None if it cannot be fitted."""
    if len(pa) < 4:
        return None
    got = fit_model(kind, pa, pb, ransac_px)
    if got is None:
        return None
    m, inl = got
    resid = residual(kind, m, pa, pb)

    # RMSE is only meaningful once the inliers over-determine the model. Each
    # point contributes two equations, so `2 * n_inliers > dof` is the threshold.
    # At or below it the model passes exactly through the points and RMSE is 0
    # by construction, not by accuracy -- a fine-tune that collapsed OHRC to 6
    # matches (2 inliers, 4-DOF similarity) reported 0.0000 px, the best number
    # in the project, from the worst fit in it. See BUGS.md BUG-016.
    over_determined = 2 * int(inl.sum()) > MODELS[kind]
    rmse = (float(np.sqrt((resid[inl] ** 2).mean()))
            if inl.any() and over_determined else float("nan"))
    return {"kind": kind, "model": m, "inliers": inl, "resid": resid, "rmse": rmse}


def select(pa: np.ndarray, pb: np.ndarray, seed: int = 0):
    """Choose a model class on held-out residual, with a parsimony margin.

    Returns {"best": kind, "models": {kind: stats}} or None when there are too
    few matches to split.
    """
    cfg = config.get()
    n = len(pa)
    if n < cfg.min_matches_for_split:
        return None

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train, test = idx[: n // 2], idx[n // 2:]

    out = {}
    for kind in MODELS:
        got = fit_model(kind, pa[train], pb[train])
        if got is None:
            continue
        m, _ = got
        r_test = residual(kind, m, pa[test], pb[test])
        # Median, not mean: a few bad matches should not decide which geometry
        # the scene has.
        out[kind] = {"dof": MODELS[kind],
                     "held_out_median": float(np.median(r_test)),
                     "held_out_rmse": float(np.sqrt((r_test ** 2).mean()))}
        full = fit(pa, pb, kind)
        if full:
            out[kind].update(model=full["model"], inliers=full["inliers"],
                             in_sample_rmse=full["rmse"],
                             inlier_ratio=float(full["inliers"].mean()))
    if not out:
        return None

    # Walk simplest to most complex, upgrading only on a decisive improvement.
    order = [k for k in MODELS if k in out]
    best = order[0]
    for kind in order[1:]:
        gain = (out[best]["held_out_median"] - out[kind]["held_out_median"])
        rel = gain / max(out[best]["held_out_median"], 1e-9)
        out[kind]["gain_over_simpler"] = float(rel)
        if rel >= cfg.complexity_margin:
            best = kind
    return {"best": best, "models": out}
