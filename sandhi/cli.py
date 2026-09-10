"""Command line interface.

    sandhi register  --source A --reference B --out DIR
    sandhi demo      --case all
    sandhi controls  --source A --reference B
    sandhi survey    --site equatorial

Replaces running loose scripts from the repo root. The scripts in `scripts/`
remain as the record of how each result was measured; this is the entry point
for using the pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import config, controls, outputs, pipeline

ROOT = Path(__file__).resolve().parent.parent


def _read(path: Path):
    """Read an image, preferring rasterio so georeferencing survives."""
    p = Path(path)
    try:
        import rasterio
        with rasterio.open(p) as src:
            return (src.read(1).astype(np.float32), src.crs, src.transform,
                    abs(src.transform.a) if src.transform else None)
    except Exception:
        import cv2
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"cannot read {p}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32), None, None, None


def _sample(case: str):
    meta_path = ROOT / "samples" / "samples.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8")).get(case)
    if not meta:
        return None
    return (ROOT / "samples" / meta["source"],
            ROOT / "samples" / meta["reference"], meta)


def cmd_register(args) -> int:
    src, _, _, src_gsd = _read(args.source)
    ref, crs, transform, ref_gsd = _read(args.reference)
    gsd = args.gsd or ref_gsd

    result = pipeline.register(
        src, ref,
        src_gsd=args.source_gsd or src_gsd,
        ref_gsd=args.reference_gsd or ref_gsd,
        refine_subpixel=not args.no_refine,
        select_model=not args.no_model_selection,
        gsd_m=gsd,
        source_native_gsd=args.source_native_gsd,
    )
    if result is None:
        print("no model could be fitted")
        return 1

    m = result["metrics"]
    print(f"model            {result['fit']['kind']}")
    print(f"matches          {m['match_count']}")
    print(f"inliers          {m['inlier_count']} ({m['inlier_ratio']*100:.0f}%)")
    print(f"RMSE             {m['rmse_px']:.3f} px"
          + (f" = {m['rmse_m']:.2f} m" if "rmse_m" in m else "")
          + (f"  (common grid {m['gsd_m']:g} m/px)" if "gsd_m" in m else "")
          + ("   [SUB-PIXEL]" if m["sub_pixel"] else ""))
    # The PS asks for sub-pixel accuracy "of source image", so this is the line
    # that answers it. It is absent when --source-gsd is unknown, rather than
    # silently falling back to the common-grid figure.
    if "rmse_source_px" in m:
        print(f"  in SOURCE px   {m['rmse_source_px']:.3f} px"
              f"  (source {m['source_gsd_m']:g} m/px)"
              + ("   [SUB-PIXEL]" if m["sub_pixel_source"] else ""))
    else:
        print("  in SOURCE px   unknown - pass --source-gsd and --reference-gsd")
    print(f"coverage         {m['coverage']:.2f}   entropy {m['entropy']:.2f}")

    if args.out:
        warped = pipeline.warp(result["src"], result)
        paths = outputs.write_all(result, warped, Path(args.out),
                                  args.name or Path(args.source).stem,
                                  crs=crs, transform=transform)
        for k, v in paths.items():
            print(f"wrote {k:<13} {v}")
    return 0


def cmd_controls(args) -> int:
    if args.case:
        got = _sample(args.case)
        if not got:
            print(f"no bundled sample for {args.case!r}")
            return 1
        src_p, ref_p, _ = got
    else:
        src_p, ref_p = Path(args.source), Path(args.reference)
    src, *_ = _read(src_p)
    ref, *_ = _read(ref_p)
    report = controls.run(controls.default_match_fn(), src, ref)
    print(report)
    return 0 if report.passed else 1


def cmd_demo(args) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import demo as demo_mod
    return demo_mod.main_args(args.case, args.out_dir) if hasattr(demo_mod, "main_args") \
        else _demo_fallback(args)


def _demo_fallback(args) -> int:
    sys.argv = ["demo", "--case", args.case, "--out-dir", args.out_dir]
    sys.path.insert(0, str(ROOT / "scripts"))
    import demo as demo_mod
    return demo_mod.main()


def cmd_survey(args) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.argv = ["survey_coverage"] + (["--site", args.site] if args.site else [])
    if args.bbox:
        sys.argv = ["survey_coverage", "--bbox"] + [str(v) for v in args.bbox]
    import survey_coverage
    return survey_coverage.main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sandhi", description=__doc__.splitlines()[0])
    p.add_argument("--contrast-kernel", type=int)
    p.add_argument("--ransac-px", type=float)
    p.add_argument("--tiles", type=int)
    p.add_argument("--weights", help="fine-tuned LoFTR checkpoint; "
                                     "omit for the pretrained outdoor weights")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("register", help="register a source image onto a reference")
    r.add_argument("--source", required=True)
    r.add_argument("--reference", required=True)
    r.add_argument("--out", help="directory for the registered product")
    r.add_argument("--name", help="basename for the outputs")
    r.add_argument("--gsd", type=float,
                   help="metres per pixel of the common grid, for RMSE in "
                        "metres. Only a fallback: when both --source-gsd and "
                        "--reference-gsd are known it is derived instead")
    r.add_argument("--source-gsd", type=float,
                   help="m/px of the source FILE as supplied; sets the "
                        "resampling ratio")
    r.add_argument("--source-native-gsd", type=float,
                   help="m/px of the source PRODUCT, if it was projected before "
                        "being passed here (OHRC: 0.26, supplied at 7.403). "
                        "Required for the 'sub-pixel accuracy of source image' "
                        "metric; defaults to --source-gsd")
    r.add_argument("--reference-gsd", type=float,
                   help="reference product's native m/px")
    r.add_argument("--no-refine", action="store_true")
    r.add_argument("--no-model-selection", action="store_true")
    r.set_defaults(func=cmd_register)

    c = sub.add_parser("controls", help="run the four-control gate")
    c.add_argument("--case", choices=["kaguya", "ohrc"], help="use a bundled sample")
    c.add_argument("--source")
    c.add_argument("--reference")
    c.set_defaults(func=cmd_controls)

    d = sub.add_parser("demo", help="end-to-end run with figures")
    d.add_argument("--case", default="all")
    d.add_argument("--out-dir", default="outputs")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("survey", help="what imagery exists over a region")
    s.add_argument("--site")
    s.add_argument("--bbox", nargs=4, type=float, metavar=("W", "E", "S", "N"))
    s.set_defaults(func=cmd_survey)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {k: v for k, v in (
        ("contrast_kernel", args.contrast_kernel),
        ("ransac_px", args.ransac_px),
        ("tiles", args.tiles),
        ("weights", args.weights)) if v is not None}
    if overrides:
        config.settings(**overrides)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
