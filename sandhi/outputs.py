"""Write the deliverable: registered product, match points, metrics.

The problem statement asks for "Software and registered product with
corresponding match points" plus evaluation metrics. That is three files.

The registered image is written as **GeoTIFF carrying the reference CRS and
transform** when the reference is georeferenced, so the product drops straight
into a GIS. A PNG is written instead only when there is no georeferencing to
inherit — a PNG is viewable but not usable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _stretch8(img):
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def write_registered(warped: np.ndarray, out_path: Path, *,
                     crs=None, transform=None, nodata=0) -> Path:
    """GeoTIFF when georeferencing is available, PNG otherwise."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if crs is not None and transform is not None:
        import rasterio
        out_path = out_path.with_suffix(".tif")
        data = warped.astype(np.float32)
        with rasterio.open(out_path, "w", driver="GTiff",
                           height=data.shape[0], width=data.shape[1],
                           count=1, dtype="float32", crs=crs,
                           transform=transform, nodata=nodata,
                           compress="deflate") as dst:
            dst.write(data, 1)
        return out_path

    import cv2
    out_path = out_path.with_suffix(".png")
    cv2.imwrite(str(out_path), _stretch8(warped))
    return out_path


def write_match_points(result: dict, out_path: Path) -> Path:
    """One row per correspondence, with its residual and inlier flag."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pa, pb = result["pa"], result["pb"]
    f = result["fit"]
    conf = result.get("conf")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_x", "source_y", "reference_x", "reference_y",
                    "confidence", "residual_px", "inlier"])
        for i in range(len(pa)):
            c = float(conf[i]) if conf is not None and i < len(conf) else ""
            w.writerow([f"{pa[i,0]:.4f}", f"{pa[i,1]:.4f}",
                        f"{pb[i,0]:.4f}", f"{pb[i,1]:.4f}",
                        f"{c:.4f}" if c != "" else "",
                        f"{f['resid'][i]:.4f}", int(f["inliers"][i])])
    return out_path


def write_metrics(result: dict, out_path: Path, extra: dict | None = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = result["fit"]
    payload = dict(result["metrics"])
    payload["model_kind"] = f["kind"]
    payload["model"] = np.asarray(f["model"]).tolist()
    payload["stages"] = result.get("stages", {})
    if extra:
        payload.update(extra)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_all(result: dict, warped: np.ndarray, out_dir: Path, name: str, *,
              crs=None, transform=None, extra: dict | None = None) -> dict:
    """All three deliverables. Returns the paths written."""
    out_dir = Path(out_dir)
    return {
        "registered": write_registered(warped, out_dir / f"{name}_registered",
                                       crs=crs, transform=transform),
        "match_points": write_match_points(result, out_dir / f"{name}_matchpoints.csv"),
        "metrics": write_metrics(result, out_dir / f"{name}_metrics.json", extra),
    }
