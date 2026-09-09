"""End-to-end tests on the bundled samples.

Marked `slow` because they run LoFTR on CPU: the pipeline tests take tens of
seconds and the control gate a couple of minutes per case. Skip with
`-m "not slow"`; run only these with `-m slow`.

`samples/` holds 3 MB of genuine cropped observations, so these run from a fresh
clone with no download and no account.

The control-gate test is the one that matters. An earlier version of this
project reported 100% inlier rates that were pure artifact — the matcher was
aligning a mask rather than the Moon (BUGS.md BUG-011). This encodes the gate
that caught it, so a regression of that kind fails the build rather than being
written into a report.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import sandhi
from sandhi import controls, outputs, pipeline

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

pytestmark = pytest.mark.slow


def _load(case):
    meta_path = SAMPLES / "samples.json"
    if not meta_path.exists():
        pytest.skip("samples/ not present")
    meta = json.loads(meta_path.read_text(encoding="utf-8")).get(case)
    if not meta:
        pytest.skip(f"no sample for {case}")
    src = cv2.imread(str(SAMPLES / meta["source"]), cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(str(SAMPLES / meta["reference"]), cv2.IMREAD_UNCHANGED)
    if src is None or ref is None:
        pytest.skip(f"sample images for {case} unreadable")
    return src.astype(np.float32), ref.astype(np.float32), meta


@pytest.fixture(scope="module")
def kaguya():
    return _load("kaguya")


@pytest.fixture(scope="module")
def ohrc():
    return _load("ohrc")


@pytest.fixture(scope="module")
def tmc():
    return _load("tmc")


# --------------------------------------------------------------- pipeline
def test_kaguya_registers_sub_pixel(kaguya):
    src, ref, meta = kaguya
    r = pipeline.register(src, ref, gsd_m=meta["gsd_m"])
    assert r is not None
    m = r["metrics"]
    assert m["match_count"] > 50
    assert m["sub_pixel"], f"RMSE {m['rmse_px']:.3f} px is not sub-pixel"
    assert m["inlier_ratio"] > 0.3


def test_ohrc_registers_sub_pixel(ohrc):
    """Real Chandrayaan-2 against an independent lunar reference."""
    src, ref, meta = ohrc
    r = pipeline.register(src, ref, gsd_m=meta["gsd_m"])
    assert r is not None
    m = r["metrics"]
    assert m["match_count"] > 500
    assert m["sub_pixel"], f"RMSE {m['rmse_px']:.3f} px is not sub-pixel"


def test_tmc_registers_with_uniform_coverage(tmc):
    """Chandrayaan-2 TMC-2, the mission's survey instrument, against Kaguya.

    This is the project's strongest case on the PS's uniformity requirement --
    coverage ~0.89 against ~0.39 for OHRC -- because TMC-2 and Kaguya are close
    in scale (1.466x) and both map-projected, so the whole frame carries usable
    texture rather than a narrow strip.
    """
    src, ref, meta = tmc
    r = pipeline.register(src, ref, gsd_m=meta["gsd_m"])
    assert r is not None
    m = r["metrics"]
    assert m["match_count"] > 1000
    assert m["sub_pixel"], f"RMSE {m['rmse_px']:.3f} px is not sub-pixel"
    assert m["inlier_ratio"] > 0.8
    assert m["coverage"] > 0.7, f"coverage {m['coverage']:.3f} below the TMC-2 baseline"


def test_coarse_alignment_is_what_makes_ohrc_work(ohrc):
    """OHRC sits ~437 px from the reference, far outside the tile search pad.

    Without coarse alignment the case collapses from >1000 matches to a handful.
    """
    src, ref, _ = ohrc
    with_coarse = pipeline.register(src, ref, coarse=True)
    without = pipeline.register(src, ref, coarse=False)
    assert with_coarse is not None
    n_with = with_coarse["metrics"]["match_count"]
    n_without = without["metrics"]["match_count"] if without else 0
    assert n_with > 10 * max(n_without, 1)


def test_model_selection_matches_the_geometry(kaguya, ohrc):
    """Kaguya is nadir-vs-nadir (no perspective); OHRC is a projected strip."""
    ks, kr, _ = kaguya
    os_, orf, _ = ohrc
    assert pipeline.register(ks, kr)["fit"]["kind"] == "similarity"
    # A polynomial-projected pushbroom strip leaves shear / anisotropic scale,
    # which similarity cannot represent.
    assert pipeline.register(os_, orf)["fit"]["kind"] in ("affine", "homography")


def test_stages_record_where_matches_are_lost(kaguya):
    src, ref, _ = kaguya
    r = pipeline.register(src, ref)
    st = r["stages"]
    assert "matched" in st and "refined" in st
    assert st["refined"] <= st["matched"]


# --------------------------------------------------------------- outputs
def test_writes_the_three_deliverables(kaguya, tmp_path):
    src, ref, meta = kaguya
    r = pipeline.register(src, ref, gsd_m=meta["gsd_m"])
    warped = pipeline.warp(r["src"], r)
    paths = outputs.write_all(r, warped, tmp_path, "test")

    assert paths["registered"].exists()
    assert paths["match_points"].exists()
    assert paths["metrics"].exists()

    rows = paths["match_points"].read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].startswith("source_x,source_y,reference_x,reference_y")
    assert len(rows) - 1 == r["metrics"]["match_count"]

    m = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    for key in ("match_count", "inlier_count", "inlier_ratio", "rmse_px",
                "coverage", "entropy", "model_kind"):
        assert key in m


# ----------------------------------------------------------- CONTROL GATE
@pytest.mark.parametrize("case", ["kaguya", "ohrc", "tmc"])
def test_control_gate_passes(case):
    """The gate that caught the fabricated 100% result. Must not regress."""
    src, ref, _ = _load(case)
    report = controls.run(controls.default_match_fn(), src, ref)
    assert report.passed, f"\n{report}"


def test_gate_rejects_a_matcher_that_ignores_its_input():
    """The gate must fail a method that does not use the imagery.

    A matcher returning a fixed correspondence looks perfect by every accuracy
    metric and is entirely fake. If this ever passes, the gate is broken.
    """
    rng = np.random.default_rng(0)
    fixed_a = rng.uniform(0, 500, (200, 2))
    fixed_b = fixed_a + np.array([3.0, -1.0])

    def blind_match(src, ref):
        return fixed_a, fixed_b            # ignores both images entirely

    src = rng.normal(128, 20, (512, 512)).astype(np.float32)
    report = controls.run(blind_match, src, src.copy())
    assert not report.passed
    names = {c["name"]: c["passed"] for c in report.checks}
    # It must fail because the roll and the degenerate inputs are unaffected.
    assert not all(names.values())
