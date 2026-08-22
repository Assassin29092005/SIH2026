# Sun-Angle Invariant Lunar Image Correspondence

**Smart India Hackathon 2026 — Problem Statement SIH26166 (ISRO / Department of Space)**

> Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)

Finding correspondences between Chandrayaan-2 imagery and lunar reference imagery at sub-pixel accuracy, with match points spread uniformly across the frame — across differences in illumination, viewpoint and scale.

---

## The problem, in one number

Two Kaguya Terrain Camera images of **identical terrain**, one shot in lunar morning and one in evening, correlate at **−0.560**.

Not weakly. *Negatively.* Morning light arrives from the east and evening light from the west, so slopes lit at dawn sit in shadow at dusk. The Moon has no atmosphere and therefore no diffuse fill light, which means a crater lit from the east is close to pixel-identical to a **dome** lit from the west.

This breaks classical feature matching in a way that is worse than mere degradation. On a 512×512 window of real lunar terrain:

| | Value |
|---|---|
| SIFT keypoints detected, morning | 1795 |
| SIFT keypoints detected, evening | 2935 |
| Matches surviving Lowe ratio test | 6 |
| **Matches actually correct** | **0** |

Detection is healthy. The *descriptors* are illumination-incompatible. SIFT does not fail loudly — it returns a handful of confident, wrong correspondences.

Worth noting: both products already carry `STANDARD_GEOMETRY = (30, 0, 30)`, meaning USGS photometric correction to identical geometry has **already been applied**. They still anti-correlate, because that correction normalises the reflectance function without reference to a DEM, so topographic shading and cast shadows survive it untouched.

## The approach

Rather than hunting for an illumination-invariant descriptor, **render the reference terrain into the source image's lighting** and match like-for-like.

| Stage | What it does |
|---|---|
| **A. Coarse localization** | Use image metadata and geometry as a positional prior, so the search is never global. |
| **B. Photometric normalization** | Render the DEM under the source image's recovered sun geometry (Lommel–Seeliger reflectance plus ray-marched cast shadows), converting cross-illumination matching into same-illumination matching. **This is the differentiator.** |
| **C. Dense matching + sub-pixel** | Semi-dense matcher (LoFTR / RoMa class), fine-tuned. Sub-pixel from soft-argmax over the correlation volume plus a local quadratic peak fit. |
| **D. Robust fit** | MAGSAC++ per tile, then a global TPS or polynomial warp to produce the registered product. |

## Results so far

Measured on Kaguya TC tile `N18E009N15E012SC` at 16.5°N, 7.40 m/px.

**A DEM alone reproduces real lunar imagery, and recovers its illumination direction.** Sweeping sun azimuth and correlating each render against the real image:

| Image | Recovered azimuth | Correlation |
|---|---|---|
| Morning | **90° (due east)** | +0.533 |
| Evening | **225° (south-west)** | +0.534 |

Nothing forced that outcome — morning could have come back western. It was a falsifiable prediction, and it held.

**Sun elevation had to be recovered too, not assumed.** Taking `STANDARD_GEOMETRY`'s 30° incidence as the acquisition geometry produced a nearly flat render (contrast cv 0.045, zero shadow pixels). The true geometry is near-terminator:

| Sun elevation | Best azimuth | r (morning) | Render contrast | Shadowed |
|---|---|---|---|---|
| 5° | 90° | **+0.581** | 1.019 | 44.5% |
| 10° | 75° | +0.573 | 0.723 | 25.7% |
| 15° | 75° | +0.546 | 0.493 | 10.8% |
| 30° | 90° | +0.539 | 0.165 | 0.0% |
| 60° | 90° | +0.533 | 0.045 | 0.0% |

### Stage B ablation — the headline

SIFT matching morning against evening, scored against the identity ground truth, across five independent 512×512 windows of the same tile:

| Window (row, col) | Raw correct / matches | Stage B correct / matches | Inlier rate | RMSE | Uniformity |
|---|---|---|---|---|---|
| (2000, 2000) | 0 / 8 | 28 / 37 | 76% | 0.602 px | 0.23 |
| (4000, 7000) | 0 / 8 | 76 / 84 | 90% | 0.743 px | 0.45 |
| (5888, 5888) | 0 / 6 | 21 / 26 | 81% | 0.504 px | 0.20 |
| (8000, 3000) | 0 / 12 | 40 / 50 | 80% | 0.768 px | 0.31 |
| (9500, 9500) | 0 / 10 | 6 / 12 | 50% | 0.627 px | 0.06 |
| **Total** | **0 / 44** | **171 / 209** | **81.8%** | **sub-pixel throughout** | |

Raw SIFT does not merely score badly across illumination — it produces **zero correct matches in 44 attempts**. With Stage B the same detector reaches 81.8% inlier rate at sub-pixel RMSE.

### Stage C — adding a dense matcher

Four combinations on identical windows, scored against the identity ground truth. LoFTR uses kornia's public `outdoor` weights, trained on MegaDepth and never on a lunar image.

| Method | Correct | Matches | Inlier rate | RMSE | Uniformity |
|---|---|---|---|---|---|
| SIFT raw | 0 | 26 | 0% | — | 0.00 |
| SIFT + Stage B | 187 | 244 | 77% | 0.640 px | 0.44 |
| **LoFTR raw** | **0** | **186** | **0%** | — | 0.00 |
| **LoFTR + Stage B** | **10504** | **10510** | **100%** | **0.679 px** | **1.00** |

Two conclusions, and the second is the important one:

1. The dense matcher is transformative — 56× more correct matches than SIFT, at complete grid coverage. Sub-pixel RMSE and uniformity 1.00 satisfy both explicit requirements of the problem statement.
2. **Stage B is load-bearing, not a nicety.** A modern learned dense matcher scores **0 out of 186** on raw cross-illumination lunar imagery. It fails just as completely as SIFT. Stage B is what takes it from 0% to 100%.

Verified against controls, because a 100% rate deserves suspicion:

| Control | Expected | Observed |
|---|---|---|
| Normalised pair still distinct? | not identical | corr −0.316, mean abs diff 76/255, 0.4% identical pixels |
| Image matched against itself | ~0 displacement | 2775/2775, RMSE 0.205 |
| Known 7 px shift | recover ±7 | +7.29 |
| Unrelated terrain | near zero matches | 0 / 5 |

### Scale invariance — the open problem

Matching a full-resolution source against a decimated reference, three 1024×1024 windows:

| Ratio | Effective m/px | Raw rate | Stage B rate | RMSE (coarse px) |
|---|---|---|---|---|
| 1× | 7.40 | 0% | **91%** | 0.63 |
| 2× | 14.81 | 0% | 31% | 1.62 |
| 4× | 29.61 | 1% | 14% | 3.61 |
| 8× | 59.23 | 1% | 6% | 9.31 |
| 16× | 118.45 | 0% | 2% | 23.00 |

Stage B beats raw at every ratio but degrades sharply. **Illumination is solved; scale is not.** Closing this is the remaining work, and it is where OHRC-to-basemap matching actually lives.

**A methodological caution worth recording three times over:** global correlation between the normalised pair is *negative* (−0.316) in the very window where LoFTR matches at 100%. Earlier, correlation got worse (−0.318 → −0.476) exactly where matching went from 0 to 21 correct. Global linear correlation measures whole-image brightness agreement; matching depends on local structure. Optimising Stage B against correlation would have tuned it precisely backwards.

## Data

Everything here is real, public, and current. No synthetic imagery, and no ground truth manufactured by warping an image.

| Role | Source | Access |
|---|---|---|
| Cross-illumination pairs + DEM | Kaguya/SELENE TC morning, evening and DTM map products | JAXA DARTS via PDS ODE — **no login** |
| Reference imagery | LROC NAC / WAC | PDS ODE — no login |
| Source imagery | Chandrayaan-2 OHRC, TMC-2, IIRS | [PRADAN](https://chmapbrowse.issdc.gov.in/) — **login required** |

**Ground truth is exact and needs no estimation.** The three Kaguya products share one map grid — verified from their labels: identical projection, `MAP_RESOLUTION`, projection offsets, bounds and 12288×12288 dimensions. So pixel *(i,j)* in morning **is** pixel *(i,j)* in evening **is** pixel *(i,j)* in the DEM. True correspondence is the identity, and a match's error is simply `hypot(x₂−x₁, y₂−y₁)`.

## Setup

Python 3.11. No conda, no WSL, no ISIS3 — `rasterio`'s wheels bundle GDAL with the PDS4/ISIS3/PDS drivers needed to read planetary products directly.

```bash
pip install -r requirements.txt
python scripts/check_env.py
```

## Scripts

Each leaves a runnable self-check behind; none silently assumes what it can verify.

| Script | Purpose |
|---|---|
| `check_env.py` | Dependencies and GDAL planetary drivers |
| `ode_catalog.py` | Enumerate ODE-indexed lunar product types |
| `survey_coverage.py` | Per-region imagery survey; `--self-check` guards against silent filter failure |
| `kaguya.py` | Resolve, verify and download morning/evening/DEM triples |
| `triple_io.py` | Open a triple, read aligned windows |
| `photometric.py` | Stage B renderer; 7 analytic self-checks |
| `estimate_sun.py` | Recover sun azimuth by render-vs-image correlation |
| `ablation.py` | Headline experiment: matching with and without Stage B |

```bash
python scripts/survey_coverage.py --site equatorial   # what exists where
python scripts/kaguya.py labels --site equatorial     # ~12 KB, validates a triple
python scripts/kaguya.py fetch  --site equatorial     # ~906 MB per triple
python scripts/photometric.py                         # analytic self-checks
python scripts/estimate_sun.py                        # recover illumination direction
```

## Evaluation

Metrics are those the problem statement names — RMSE, inlier count, inlier ratio — plus **match uniformity**, which it also requires ("maintaining uniform distribution across the images") and which most implementations skip. Uniformity is scored as grid coverage fraction and normalised entropy of per-cell match counts.

Results are reported stratified by **|Δsun azimuth|** and **scale ratio**, since a single aggregate number hides exactly where classical methods collapse.

Baselines: SIFT, ASIFT, phase correlation, off-the-shelf SuperPoint+LightGlue.

## Project conventions

See [CLAUDE.md](CLAUDE.md) for the full working agreement. The load-bearing ones:

- **Data must be genuine, public and current.** No synthetic imagery as a training or evaluation source.
- **Ground truth comes from geometry, never from a warp we invented.** Augmenting real images is fine and gets documented; manufacturing correspondence is not.
- **Zero extra hardware.** Laptops plus free Kaggle GPU hours.
- **Fine-tune, never train from scratch.**

Every bug found is logged in [BUGS.md](BUGS.md) with its root cause and fix. Several entries record cases where a *test* was wrong and the code was right — including two where the experiment passed while an intermediate quantity was badly incorrect. That log is part of the deliverable, not housekeeping.
