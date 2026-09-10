"""One command, one figure: the whole pipeline as something a person can look at.

Everything else in this repository prints numbers. This renders what those
numbers mean -- the two images, the correspondences drawn between them, the
registered overlay, where the match points landed, and the metrics.

    python scripts/demo.py                      # Kaguya cross-illumination
    python scripts/demo.py --case ohrc          # real Chandrayaan-2 OHRC
    python scripts/demo.py --list               # what can be run right now
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from register import (  # noqa: E402
    BUCKET_GRID,
    fit,
    register_pair,
    uniformity,
)
from remeasure import build_raw_hp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SAMPLES = ROOT / "samples"

# Native product resolutions, from the PDS4 labels -- read, never estimated
# (BUG-010). Needed for the "sub-pixel accuracy of source image" metric, which
# is stated in the SOURCE product's pixels, not the common grid. OHRC's label
# declares 0.26 m; the measured footprint is 0.3060 x 0.3225 m, so 0.26 is the
# conservative choice -- a smaller pixel makes the reported error larger.
OHRC_GSD_M = 0.26


def load_sample(case):
    """Committed sample crops, so a fresh clone can run without any download.

    These are genuine observations cropped to a single window -- not synthetic,
    not rendered. Full-resolution products are ~7 GB across three archives, two
    of which need accounts, so the repository carries a 3 MB slice instead.
    """
    meta_path = SAMPLES / "samples.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8")).get(case)
    if not meta:
        return None
    src_p, ref_p = SAMPLES / meta["source"], SAMPLES / meta["reference"]
    if not (src_p.exists() and ref_p.exists()):
        return None
    src = cv2.imread(str(src_p), cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(str(ref_p), cv2.IMREAD_UNCHANGED)
    if src is None or ref is None:
        return None
    return dict(src=src.astype(np.float32), ref=ref.astype(np.float32),
                gsd=meta["gsd_m"], source_gsd=meta["source_gsd_m"],
                title=meta["title"], sub=meta["sub"] + "  [sample]",
                src_label=meta["src_label"], ref_label=meta["ref_label"])


def stretch8(img):
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def checkerboard(a, b, squares=8):
    """Interleave two registered images so misalignment shows as a broken edge.

    Where the warp leaves no data -- a large registration shift moves part of
    the source outside the frame -- fall back to the reference, so empty regions
    read as reference rather than as black holes.
    """
    h, w = a.shape
    a = np.where(a > 0, a, b)
    out = a.copy()
    sy, sx = h // squares, w // squares
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2:
                out[i * sy:(i + 1) * sy, j * sx:(j + 1) * sx] = \
                    b[i * sy:(i + 1) * sy, j * sx:(j + 1) * sx]
    return out


def draw_matches(ax, a8, b8, pa, pb, inl, max_lines=120):
    """Side-by-side with correspondence lines. Green inlier, red rejected."""
    h, w = a8.shape
    canvas = np.zeros((h, w * 2 + 12), np.uint8)
    canvas[:, :w] = a8
    canvas[:, w + 12:] = b8
    ax.imshow(canvas, cmap="gray", interpolation="nearest")

    idx = np.arange(len(pa))
    if len(idx) > max_lines:
        idx = np.linspace(0, len(pa) - 1, max_lines).astype(int)
    for i in idx:
        ax.plot([pa[i, 0], pb[i, 0] + w + 12], [pa[i, 1], pb[i, 1]],
                lw=0.4, alpha=0.55,
                color=("#2ecc71" if inl[i] else "#e74c3c"))
    ax.scatter(pa[inl, 0], pa[inl, 1], s=2, c="#2ecc71")
    ax.scatter(pb[inl, 0] + w + 12, pb[inl, 1], s=2, c="#2ecc71")
    ax.set_axis_off()


def coverage_grid(pa, shape, grid=BUCKET_GRID):
    h, w = shape
    gy = np.clip((pa[:, 1] / h * grid).astype(int), 0, grid - 1)
    gx = np.clip((pa[:, 0] / w * grid).astype(int), 0, grid - 1)
    return np.bincount(gy * grid + gx, minlength=grid * grid).reshape(grid, grid)


def load_kaguya(row=2000, col=2000, size=1024):
    from triple_io import find_tiles, open_triple

    tiles = find_tiles()
    if not tiles:
        return None
    t = open_triple("N18E009N15E012SC" if "N18E009N15E012SC" in tiles else tiles[0])
    b = t.read_window(row, col, size)
    return dict(src=b["morning"].astype(np.float32),
                ref=b["evening"].astype(np.float32),
                gsd=abs(t.transform.a),
                # Source and reference are the same instrument on one grid, so
                # the source's native GSD IS the common grid here.
                source_gsd=abs(t.transform.a),
                title="Kaguya TC morning vs evening",
                sub=f"tile {t.tile}, window ({row},{col}) {size}x{size}, 7.40 m/px",
                src_label="SOURCE  morning", ref_label="REFERENCE  evening")


def load_ohrc(rows=1024):
    import rasterio
    from ohrc_vs_kaguya import kaguya_window, load_projected
    from triple_io import find_tiles, open_triple

    path = ROOT / "data" / "interim" / "ohrc_7m.tif"
    if not path.exists() or "N03E021N00E024SC" not in find_tiles():
        return None
    ohrc, _, bounds = load_projected(path)
    t = open_triple("N03E021N00E024SC")
    bands = kaguya_window(t, bounds)
    w = min(ohrc.shape[1], bands["evening"].shape[1])
    sl = slice(1024, 1024 + rows)
    return dict(src=ohrc[sl, :w].astype(np.float32),
                ref=bands["evening"][sl, :w].astype(np.float32),
                gsd=7.403,
                # The OHRC product is 0.26 m/px; what enters the matcher is that
                # product polynomial-projected to 7.403 m. The PS clause asks
                # about the PRODUCT, so report the native figure -- 28.5x coarser
                # is the honest answer, not the projected grid's 0.5 px.
                source_gsd=OHRC_GSD_M,
                title="Chandrayaan-2 OHRC vs Kaguya TC reference",
                sub=f"OHRC 0.26 m/px projected to 7.40 m/px, ~0.6N 23.4E",
                src_label="SOURCE  Chandrayaan-2 OHRC",
                ref_label="REFERENCE  Kaguya evening")


def load_tmc(lat=0.5, size=1536):
    """Chandrayaan-2 TMC-2 ortho against Kaguya TC, on a common grid.

    TMC-2 is the mission's coverage instrument -- 8436 catalogued products
    against OHRC's 624 -- so this is the case that shows the pipeline on
    Chandrayaan-2 data at survey scale rather than on a single targeted strip.

    Like OHRC, the source arrives resampled: TMC-2 is 5.05 m/px and Kaguya
    7.403 m/px, so the source is brought to the reference GSD here rather than
    inside the demo, because `register_pair` takes no GSD arguments. The ratio
    is read from the products (1.466), never estimated -- see BUG-010, and
    BUG-020 for why the non-integer ratio needed a real resampler.
    """
    import rasterio
    from sandhi.pipeline import resample
    from tmc_vs_kaguya import KAGUYA_GSD_M, TMC_GSD_M, kaguya_over, strip_window

    if not (ROOT / "data" / "raw" / "ch2").exists():
        return None
    from ch2_io import open_tmc_pair
    t = open_tmc_pair()
    with rasterio.open(t.ortho_path) as src:
        img, bounds, (row, col) = strip_window(src, lat, size)
    ref, tile = kaguya_over(bounds, "evening")
    if ref is None:
        return None

    src_common = resample(img, KAGUYA_GSD_M / TMC_GSD_M)
    h = min(src_common.shape[0], ref.shape[0])
    w = min(src_common.shape[1], ref.shape[1])
    return dict(src=src_common[:h, :w].astype(np.float32),
                ref=ref[:h, :w].astype(np.float32),
                gsd=KAGUYA_GSD_M, source_gsd=TMC_GSD_M,
                title="Chandrayaan-2 TMC-2 vs Kaguya TC reference",
                sub=f"TMC-2 5.05 m/px resampled to 7.40 m/px (1.466x), "
                    f"{bounds.bottom:.2f}N {bounds.left:.2f}E, tile {tile}",
                src_label="SOURCE  Chandrayaan-2 TMC-2",
                ref_label="REFERENCE  Kaguya evening")


_FULL = {"kaguya": load_kaguya, "ohrc": load_ohrc, "tmc": load_tmc}


def load_case(case):
    """Prefer full downloaded data; fall back to the committed sample."""
    try:
        got = _FULL[case]()
    except Exception:
        got = None
    return got if got is not None else load_sample(case)


CASES = {name: (lambda n=name: load_case(n)) for name in _FULL}


def run(case: str, out_path: Path):
    data = CASES[case]()
    if data is None:
        print(f"case '{case}' needs data that is not downloaded yet")
        return 1

    src, ref, gsd = data["src"], data["ref"], data["gsd"]
    print(f"{data['title']}\n  {data['sub']}\n  matching...")

    result = register_pair(src, ref)
    if result is None:
        print("  no model could be fitted")
        return 1

    pa, pb = result["pa"], result["pb"]
    f = result["fit"]
    inl = f["inliers"]
    cov, ent = uniformity(pa[inl], src.shape)
    rmse = f["rmse"]

    a8, b8 = build_raw_hp(src, None, None), build_raw_hp(ref, None, None)
    warped = cv2.warpAffine(stretch8(src).astype(np.float32), f["model"],
                            (ref.shape[1], ref.shape[0]), flags=cv2.INTER_CUBIC)

    fig = plt.figure(figsize=(16, 9), facecolor="#12151a")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.16, wspace=0.12)
    for ax_pos, img, label in [((0, 0), stretch8(src), data["src_label"]),
                               ((0, 1), stretch8(ref), data["ref_label"])]:
        ax = fig.add_subplot(gs[ax_pos])
        ax.imshow(img, cmap="gray")
        ax.set_title(label, color="#8fa6c0", fontsize=10, pad=6)
        ax.set_axis_off()

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(checkerboard(stretch8(warped), stretch8(ref)), cmap="gray")
    ax.set_title("REGISTERED  checkerboard against reference",
                 color="#8fa6c0", fontsize=10, pad=6)
    ax.set_axis_off()

    ax = fig.add_subplot(gs[1, :2])
    draw_matches(ax, a8, b8, pa, pb, inl)
    ax.set_title(f"CORRESPONDENCES   green = inlier ({int(inl.sum())}), "
                 f"red = rejected ({int((~inl).sum())})",
                 color="#8fa6c0", fontsize=10, pad=6)

    ax = fig.add_subplot(gs[1, 2])
    grid = coverage_grid(pa[inl], src.shape)
    ax.imshow(grid, cmap="viridis")
    ax.set_title(f"MATCH DISTRIBUTION  {BUCKET_GRID}x{BUCKET_GRID} cells",
                 color="#8fa6c0", fontsize=10, pad=6)
    ax.set_axis_off()

    # RMSE lives on the COMMON grid; the PS asks for accuracy in the source
    # product's own pixels. Both are reported -- see BUG-025.
    src_gsd = data["source_gsd"]
    rmse_src = rmse * gsd / src_gsd
    metrics = {
        "match_count": int(len(pa)), "inlier_count": int(inl.sum()),
        "inlier_ratio": float(inl.mean()), "rmse_px": float(rmse),
        "rmse_m": float(rmse * gsd), "sub_pixel": bool(rmse < 1.0),
        "gsd_m": float(gsd), "source_gsd_m": float(src_gsd),
        "rmse_source_px": float(rmse_src),
        "sub_pixel_source": bool(rmse_src < 1.0),
        "coverage": float(cov), "entropy": float(ent),
    }
    verdict = "SUB-PIXEL" if metrics["sub_pixel"] else f"{rmse:.2f} px"
    src_verdict = ("SUB-PIXEL in source px" if metrics["sub_pixel_source"]
                   else f"{rmse_src:.1f} source px - limited by the "
                        f"{gsd:g} m reference")
    fig.suptitle(data["title"], color="#e8eef6", fontsize=15, y=0.975)
    fig.text(0.5, 0.935, data["sub"], color="#6b7c93", fontsize=9, ha="center")
    fig.text(0.5, 0.040,
             f"matches {metrics['match_count']}   "
             f"inliers {metrics['inlier_count']} ({metrics['inlier_ratio']*100:.0f}%)   "
             f"RMSE {rmse:.3f} px = {rmse*gsd:.2f} m  [{verdict}]   "
             f"coverage {cov:.2f}   entropy {ent:.2f}",
             color="#e8eef6", fontsize=11, ha="center", family="monospace")
    fig.text(0.5, 0.012,
             f"source {src_gsd:g} m/px  ->  {rmse_src:.3f} px of source  "
             f"[{src_verdict}]",
             color=("#2ecc71" if metrics["sub_pixel_source"] else "#c9a227"),
             fontsize=10, ha="center", family="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)

    (out_path.with_suffix(".json")).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  matches {metrics['match_count']}, inliers {metrics['inlier_count']} "
          f"({metrics['inlier_ratio']*100:.0f}%)")
    print(f"  RMSE {rmse:.3f} px = {rmse*gsd:.2f} m  [{verdict}]")
    print(f"  source {src_gsd:g} m/px -> {rmse_src:.3f} source px  [{src_verdict}]")
    print(f"  coverage {cov:.2f}, entropy {ent:.2f}")
    print(f"  wrote {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=sorted(CASES) + ["all"], default="kaguya")
    p.add_argument("--list", action="store_true")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    if args.list:
        print("cases runnable with the data currently present:")
        for name in CASES:
            try:
                full = _FULL[name]() is not None
            except Exception:
                full = False
            sample = load_sample(name) is not None
            src = "full data" if full else ("committed sample" if sample else "MISSING")
            print(f"  {name:<8} {src}")
        return 0

    cases = sorted(CASES) if args.case == "all" else [args.case]
    rc = 0
    for name in cases:
        rc |= run(name, out_dir / f"demo_{name}.png")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
