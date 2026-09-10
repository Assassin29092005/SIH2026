"""Fast unit tests. No imagery, no network, no model download.

These test the maths and the bookkeeping — the places where a wrong sign or a
mis-indexed array produces plausible output rather than a crash. Several of them
encode bugs that actually happened; see BUGS.md.
"""

import numpy as np
import pytest

from sandhi import config, metrics, models, normalize
from sandhi.pipeline import decimate, to_common_gsd
from sandhi.pipeline import resample as pipeline_resample


# ---------------------------------------------------------------- config
def test_config_override_and_reset():
    config.reset()
    assert config.get().ransac_px == 1.5
    config.settings(ransac_px=3.0)
    assert config.get().ransac_px == 3.0
    config.reset()
    assert config.get().ransac_px == 1.5


# ------------------------------------------------------------ normalize
def test_local_contrast_removes_a_smooth_gradient():
    """A smooth ramp added to texture must not survive local normalisation.

    This is the whole premise: illumination is low-frequency, terrain detail is
    not.
    """
    rng = np.random.default_rng(0)
    texture = rng.normal(128, 20, (256, 256)).astype(np.float32)
    ramp = np.linspace(0, 200, 256, dtype=np.float32)[None, :]

    a = normalize.local_contrast(texture)
    b = normalize.local_contrast(texture + ramp)
    # Identical texture under very different illumination should normalise to
    # nearly the same image.
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9


def test_stretch8_handles_all_nan():
    out = normalize.stretch8(np.full((8, 8), np.nan))
    assert out.dtype == np.uint8 and out.max() == 0


# -------------------------------------------------------------- metrics
def test_uniformity_clustered_vs_spread():
    shape = (256, 256)
    clustered = np.column_stack([np.full(64, 10.0), np.full(64, 10.0)])
    g = config.get().uniformity_grid
    ys, xs = np.meshgrid(np.linspace(5, 250, g), np.linspace(5, 250, g))
    spread = np.column_stack([xs.ravel(), ys.ravel()])

    c_cov, c_ent = metrics.uniformity(clustered, shape)
    s_cov, s_ent = metrics.uniformity(spread, shape)
    assert s_cov > c_cov and s_ent > c_ent
    assert c_cov == pytest.approx(1 / (g * g))
    assert s_cov == pytest.approx(1.0)


def test_uniformity_empty_is_zero():
    assert metrics.uniformity(np.zeros((0, 2)), (64, 64)) == (0.0, 0.0)


def test_offset_of_is_robust_to_outliers():
    pa = np.random.default_rng(1).uniform(0, 100, (40, 2))
    pb = pa + np.array([3.0, -2.0])
    pb[:5] += 500          # wild outliers
    got = metrics.offset_of(pa, pb)
    assert got[0] == pytest.approx(3.0, abs=0.5)
    assert got[1] == pytest.approx(-2.0, abs=0.5)


def test_offset_of_returns_none_below_threshold():
    n = config.get().min_matches - 1
    assert metrics.offset_of(np.zeros((n, 2)), np.zeros((n, 2))) is None


def test_summarise_sub_pixel_flag_is_nan_safe():
    pa = np.zeros((4, 2))
    inl = np.zeros(4, bool)
    out = metrics.summarise(pa, pa, inl, np.zeros(4), (64, 64))
    assert out["sub_pixel"] is False        # NaN RMSE must not read as sub-pixel


def test_rmse_is_reported_in_source_pixels_not_common_grid():
    """The PS asks for sub-pixel accuracy OF SOURCE IMAGE. BUG-025.

    A residual that is sub-pixel on the common grid is NOT sub-pixel in the
    source's own pixels whenever the source is finer than the reference, and
    reporting only the common-grid figure hides that by the scale ratio.
    """
    pa = np.zeros((4, 2))
    inl = np.ones(4, bool)
    resid = np.full(4, 0.5)                     # 0.5 px on the common grid

    # OHRC: 0.26 m product matched on a 7.403 m grid. 0.5 px = 3.70 m = 14.2
    # OHRC pixels, so it is sub-pixel on the grid and emphatically not in source.
    out = metrics.summarise(pa, pa, inl, resid, (64, 64),
                            gsd_m=7.403, source_gsd_m=0.26)
    assert out["sub_pixel"] is True
    assert out["sub_pixel_source"] is False
    assert out["rmse_source_px"] == pytest.approx(0.5 * 7.403 / 0.26)

    # TMC-2: 5.05 m product on the same grid clears the bar in its own pixels.
    out = metrics.summarise(pa, pa, inl, resid, (64, 64),
                            gsd_m=7.403, source_gsd_m=5.05)
    assert out["sub_pixel_source"] is True

    # Absent rather than silently equal to the common-grid figure.
    out = metrics.summarise(pa, pa, inl, resid, (64, 64), gsd_m=7.403)
    assert "rmse_source_px" not in out


def test_common_gsd_is_the_coarser_of_the_pair():
    """`to_common_gsd` resamples to the coarser side, so metres follow max().

    A caller passing the REFERENCE gsd is right only while the source is finer.
    IIRS at 85.08 m against Kaguya at 7.403 m would report every distance 11.5x
    too small. BUG-025.
    """
    fine, coarse = np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32)
    _, _, k = to_common_gsd(fine, coarse, 5.05, 7.403)
    assert k == pytest.approx(7.403 / 5.05)     # source decimated to the reference
    _, _, k = to_common_gsd(fine, coarse, 85.08, 7.403)
    assert k == pytest.approx(7.403 / 85.08)    # reference decimated to the source


def test_cross_window_spread():
    tight = [(1.0, 2.0), (1.1, 2.1), (0.9, 1.9)]
    loose = [(1.0, 2.0), (40.0, -30.0), (-20.0, 60.0)]
    assert metrics.cross_window_spread(tight) < metrics.cross_window_spread(loose)


# --------------------------------------------------------------- models
def _apply(kind, m, pts):
    return models.apply(kind, m, pts)


@pytest.mark.parametrize("kind", ["similarity", "affine", "homography"])
def test_fit_recovers_a_known_transform(kind):
    """Geometry check on constructed point sets - maths, not imagery."""
    rng = np.random.default_rng(2)
    pa = rng.uniform(0, 500, (80, 2))
    if kind == "similarity":
        th, s, t = np.radians(7.0), 1.05, np.array([12.0, -5.0])
        R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pb = pa @ R.T + t
    elif kind == "affine":
        A = np.array([[1.03, 0.08], [-0.02, 0.97]])
        pb = pa @ A.T + np.array([4.0, 9.0])
    else:
        H = np.array([[1.02, 0.03, 5.0], [0.01, 0.99, -4.0], [1e-5, 2e-5, 1.0]])
        pb = _apply("homography", H, pa)

    f = models.fit(pa, pb, kind=kind)
    assert f is not None
    assert f["rmse"] < 0.05
    assert f["inliers"].mean() > 0.95


def test_selection_prefers_the_simple_model_when_it_suffices():
    """Parsimony: a pure similarity must NOT be fitted with homography.

    Held-out scoring alone was not enough - homography still won nadir pairs by
    5-9% noise. This guards the complexity margin that fixed it.
    """
    rng = np.random.default_rng(3)
    pa = rng.uniform(0, 500, (120, 2))
    th, s, t = np.radians(4.0), 1.02, np.array([6.0, -3.0])
    R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pb = pa @ R.T + t + rng.normal(0, 0.3, pa.shape)

    sel = models.select(pa, pb)
    assert sel is not None
    assert sel["best"] == "similarity"


def test_selection_upgrades_when_the_geometry_demands_it():
    """A strong shear cannot be represented by similarity, so affine must win."""
    rng = np.random.default_rng(4)
    pa = rng.uniform(0, 500, (120, 2))
    A = np.array([[1.0, 0.25], [0.0, 1.0]])       # pure shear
    pb = pa @ A.T + rng.normal(0, 0.3, pa.shape)

    sel = models.select(pa, pb)
    assert sel is not None
    assert sel["best"] in ("affine", "homography")


def test_selection_returns_none_below_split_threshold():
    n = config.get().min_matches_for_split - 1
    pa = np.random.default_rng(5).uniform(0, 100, (n, 2))
    assert models.select(pa, pa.copy()) is None


# ------------------------------------------------------------- pipeline
def test_decimate_is_area_averaged_not_point_sampled():
    """Point sampling aliases; area averaging preserves the mean.

    Decimating ~28x with nearest-neighbour is what made the first OHRC run fail
    its controls.
    """
    rng = np.random.default_rng(6)
    img = rng.normal(100, 30, (256, 256)).astype(np.float32)
    small = decimate(img, 4)
    assert small.shape == (64, 64)
    assert small.mean() == pytest.approx(img.mean(), abs=1.0)
    # Averaging must reduce variance; point sampling would not.
    assert small.std() < img.std() * 0.7


def test_decimate_identity_at_k1():
    img = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert np.array_equal(decimate(img, 1), img)


def test_to_common_gsd_decimates_the_finer_side():
    fine = np.zeros((512, 512), np.float32)
    coarse = np.zeros((128, 128), np.float32)
    s, r, k = to_common_gsd(fine, coarse, 0.25, 1.0)
    assert s.shape == (128, 128) and r.shape == (128, 128)
    assert k == pytest.approx(4.0)


def test_to_common_gsd_noop_when_equal():
    a = np.zeros((64, 64), np.float32)
    s, r, k = to_common_gsd(a, a, 7.403, 7.403)
    assert k == 1.0 and s.shape == r.shape == (64, 64)


def test_rmse_is_nan_when_the_fit_is_not_over_determined():
    """A model passing exactly through its inliers has RMSE 0 by construction.

    A fine-tune that collapsed OHRC to 6 matches (2 inliers on a 4-DOF
    similarity) reported 0.0000 px — the best RMSE in the project, from its
    worst fit. Degenerate fits must report NaN, not a flattering number.
    See BUGS.md BUG-016.
    """
    pa = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    pb = pa + np.array([3.0, -2.0])

    f2 = models.fit(pa[:2], pb[:2], kind="similarity")   # 2 pts, 4 DOF: exactly determined
    assert f2 is None or np.isnan(f2["rmse"])

    f4 = models.fit(pa, pb, kind="similarity")           # 4 pts: over-determined
    assert f4 is not None and not np.isnan(f4["rmse"])
    assert f4["rmse"] < 0.01


@pytest.mark.data
def test_training_windows_never_touch_evaluation_data():
    """Training sources must be disjoint from every acceptance criterion.

    The 2026-09-09 fine-tune trained on Kaguya tile N03E021N00E024SC, which
    supplies the OHRC sample's reference image, while the other downloaded tile
    is the Kaguya sample itself (BUGS.md BUG-017). Both are evaluation data.
    LROC NAC files [0] and [1] are the obliquity ladder, which is the criterion
    the fine-tune exists to move.
    """
    import itertools
    from pathlib import Path
    from sandhi import training

    root = Path(__file__).resolve().parent.parent
    if not (root / "data" / "raw" / "nac").exists():
        pytest.skip("raw imagery not present")

    names = {n for n, _, _, _ in itertools.islice(training.safe_windows(), 12)}
    assert names, "safe_windows yielded nothing"
    for n in names:
        assert n.startswith(("tmc2", "nac-")), f"unexpected training source {n}"
        assert "N18E009" not in n and "N03E021" not in n, f"Kaguya eval tile in {n}"
        assert "ohrc" not in n.lower(), f"OHRC is eval-only, got {n}"

    ladder = sorted((root / "data" / "raw" / "nac").glob("*.IMG"))[:2]
    for f in ladder:
        assert not any(f.stem in n for n in names), f"ladder image {f.stem} used for training"


def test_to_common_gsd_handles_a_non_integer_ratio():
    """Real cross-sensor ratios are rarely integers.

    TMC-2 (5.05 m/px) against Kaguya (7.403 m/px) is 1.466x. Rounding that to
    an integer gives 1, which skipped the resampling entirely and left the two
    images at different scales AND different sizes — surfacing 60 lines later as
    a LoFTR convolution error on a 150x6 tile. See BUGS.md BUG-020.
    """
    fine = np.zeros((1024, 1024), np.float32)     # TMC-2 window
    coarse = np.zeros((699, 699), np.float32)     # same ground, Kaguya grid
    s, r, k = to_common_gsd(fine, coarse, 5.05, 7.403)
    assert k == pytest.approx(7.403 / 5.05, rel=1e-6)
    assert s.shape == r.shape, f"sizes must match after resampling, got {s.shape} vs {r.shape}"


def test_resample_preserves_the_mean():
    rng = np.random.default_rng(7)
    img = rng.normal(100, 30, (512, 512)).astype(np.float32)
    out = pipeline_resample(img, 1.466)
    assert out.shape == (349, 349)
    assert out.mean() == pytest.approx(img.mean(), abs=1.0)
