"""Decide whether a fine-tuned LoFTR checkpoint may replace the pretrained one.

The fine-tune reported a large gain on its own validation split: coarse-cell
precision 0.087 -> 0.391 on a held-out Kaguya tile. That number is measured on
warped geometry, which is the training domain, so on its own it cannot justify
adopting the weights. A model can improve on warped pairs by learning the warp
distribution rather than by becoming viewpoint-invariant, and it can do so while
losing cross-illumination performance, which is the axis the project already
works on.

So the bar is set on REAL pairs, and it was set before these results were seen:

1. the four-control gate still passes on both bundled cases,
2. cross-illumination does not regress -- RMSE stays sub-pixel on kaguya and
   ohrc, and match count does not collapse,
3. the obliquity ladder improves: matching must succeed past the measured
   12.3 deg emission-angle ceiling.

Criterion 3 is the one the fine-tune was actually for. 1 and 2 are the guards
against buying it at the cost of everything else.

    python scripts/adopt_check.py --weights data/interim/loftr_lunar_best.pt

Writes outputs/adopt_check.json. Prints a verdict per criterion, and adopts
nothing on its own -- changing the default is a separate, deliberate edit to
sandhi/config.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sandhi import config, controls, pipeline  # noqa: E402

SAMPLES = ROOT / "samples"


def load_case(case: str):
    meta = json.loads((SAMPLES / "samples.json").read_text(encoding="utf-8"))[case]
    src = cv2.imread(str(SAMPLES / meta["source"]), cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(str(SAMPLES / meta["reference"]), cv2.IMREAD_UNCHANGED)
    return src.astype(np.float32), ref.astype(np.float32), meta


def measure(case: str, weights: str | None) -> dict:
    """Register one bundled case under the given weights."""
    config.reset()
    if weights:
        config.settings(weights=weights)
    src, ref, meta = load_case(case)
    r = pipeline.register(src, ref, gsd_m=meta.get("gsd_m"))
    if r is None:
        return {"registered": False}
    m = r["metrics"]
    return {"registered": True,
            "match_count": m["match_count"],
            "inlier_ratio": round(m["inlier_ratio"], 4),
            "rmse_px": round(m["rmse_px"], 4),
            "coverage": round(m["coverage"], 4),
            "entropy": round(m["entropy"], 4),
            "model_kind": r["fit"]["kind"]}


def gate(case: str, weights: str | None) -> dict:
    config.reset()
    if weights:
        config.settings(weights=weights)
    src, ref, _ = load_case(case)
    report = controls.run(controls.default_match_fn(), src, ref)
    return {"passed": bool(report.passed),
            "checks": {c["name"]: bool(c["passed"]) for c in report.checks}}


def check_provenance(weights: str) -> list[str]:
    """Warn if the checkpoint's report is not from the current notebook.

    `kaggle kernels output` downloads the last SAVED version, not the draft
    session. Re-running a notebook without Save Version therefore re-downloads
    the previous run's artefacts, with fresh timestamps and identical content —
    which looks exactly like a successful new download. That happened on
    2026-09-10 and would have spent ten minutes re-evaluating weights this
    script had already rejected.

    The current notebook records a real-pair match count; the run that produced
    the rejected weights did not. That difference is the fingerprint.
    """
    report = Path(weights).with_name("finetune_report.json")
    if not report.exists():
        return [f"no finetune_report.json beside {Path(weights).name} -- "
                "provenance unverifiable"]
    try:
        r = json.loads(report.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"{report.name} is not valid JSON: {e}"]

    problems = []
    if r.get("checkpoint_saved") is False:
        problems.append(
            "the run saved NO checkpoint -- every epoch either failed to improve "
            f"the held-out metric or fell below the real-pair floor "
            f"({r.get('real_match_floor')} of {r.get('baseline_real_matches')}). "
            "Any .pt beside this report is from an earlier run")
    if "baseline_real_matches" not in r:
        problems.append(
            "report has no 'baseline_real_matches' -- it predates the real-pair "
            "guard, so this is an OLD run's checkpoint")
    hist = r.get("history") or [{}]
    if "real_matches" not in hist[0]:
        problems.append("history entries carry no 'real_matches' -- same conclusion")
    if problems:
        problems.append(
            f"report: epochs={r.get('epochs')} lr={r.get('lr')} "
            f"held_out={r.get('held_out_tile')}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="fine-tuned checkpoint")
    ap.add_argument("--skip-provenance", action="store_true",
                    help="evaluate anyway despite a stale or missing report")
    ap.add_argument("--cases", nargs="*", default=["kaguya", "ohrc"])
    ap.add_argument("--skip-gate", action="store_true",
                    help="skip the control gate (slow); NOT valid for adoption")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"no such checkpoint: {args.weights}")
        return 1

    stale = check_provenance(args.weights)
    if stale and not args.skip_provenance:
        print("=" * 72)
        print("STALE CHECKPOINT - refusing to evaluate")
        print("=" * 72)
        for s in stale:
            print(f"  - {s}")
        print("")
        print("  On Kaggle: Save Version AFTER the run, then re-download; or take")
        print("  the files straight from the Output panel of the draft session.")
        print("  Override with --skip-provenance if you know better.")
        return 1

    out = {"weights": args.weights, "cases": {}, "gate": {}}

    print("=" * 72)
    print("CRITERION 2 - cross-illumination must not regress")
    print("=" * 72)
    for case in args.cases:
        base = measure(case, None)
        tuned = measure(case, args.weights)
        out["cases"][case] = {"pretrained": base, "finetuned": tuned}
        print(f"\n{case}")
        for k in ("match_count", "inlier_ratio", "rmse_px", "coverage", "entropy"):
            b, t = base.get(k), tuned.get(k)
            if b is None or t is None:
                print(f"  {k:<14} pretrained {b}  finetuned {t}")
                continue
            arrow = "worse" if (k == "rmse_px") == (t > b) else "better"
            print(f"  {k:<14} {b:>10.4f} -> {t:>10.4f}   {arrow}")
        print(f"  {'model':<14} {base.get('model_kind')} -> {tuned.get('model_kind')}")

    if not args.skip_gate:
        print("")
        print("=" * 72)
        print("CRITERION 1 - the four-control gate must still pass")
        print("=" * 72)
        for case in args.cases:
            g = gate(case, args.weights)
            out["gate"][case] = g
            status = "PASS" if g["passed"] else "FAIL"
            print(f"  {case}: {status}  {g['checks']}")

    verdict = []
    for case, r in out["cases"].items():
        b, t = r["pretrained"], r["finetuned"]
        if not t.get("registered"):
            verdict.append(f"{case}: finetuned FAILED to register")
        elif t["rmse_px"] > 1.0:
            verdict.append(f"{case}: finetuned RMSE {t['rmse_px']:.3f} px is not sub-pixel")
        elif t["match_count"] < 0.5 * b["match_count"]:
            verdict.append(f"{case}: matches collapsed "
                           f"{b['match_count']} -> {t['match_count']}")
    for case, g in out["gate"].items():
        if not g["passed"]:
            verdict.append(f"{case}: control gate FAILED")

    out["blocking"] = verdict
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "adopt_check.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print("")
    print("=" * 72)
    if verdict:
        print("DO NOT ADOPT. Blocking:")
        for v in verdict:
            print(f"  - {v}")
    else:
        print("Criteria 1 and 2 clear. Criterion 3 (obliquity ladder) is still")
        print("required before adoption:")
        print("  python scripts/viewpoint.py --ladder")
    print("=" * 72)
    print("wrote outputs/adopt_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
