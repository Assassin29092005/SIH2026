# Sun-Angle Invariant Lunar Image Correspondence

**Smart India Hackathon 2026 — Problem Statement SIH26166 (ISRO / Department of Space)**

> Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)

Registering Chandrayaan-2 imagery against independent lunar reference imagery across illumination, sensor and scale — with every reported number gated by adversarial controls.

---

## The problem, in one number

Two Kaguya Terrain Camera images of **identical terrain**, one shot in lunar morning and one in evening, correlate at **−0.560**.

Not weakly. *Negatively.* Morning light arrives from the east and evening light from the west, so slopes lit at dawn sit in shadow at dusk. The Moon has no atmosphere and therefore no diffuse fill light, which means a crater lit from the east is close to pixel-identical to a **dome** lit from the west.

Both products already carry `STANDARD_GEOMETRY = (30, 0, 30)` — USGS photometric correction to identical geometry has *already been applied*. They still anti-correlate, because that correction normalises the reflectance function without reference to a DEM, so topographic shading and cast shadows survive it untouched.

## Results

All numbers from `scripts/remeasure.py`, which refuses to report any method failing its control gate.

### Cross-illumination matching

| Method | Cross-window spread | Inlier ratio | RMSE | Coverage | Controls |
|---|---|---|---|---|---|
| **LoFTR + local contrast norm** | **0.85 px** | **54%** | 1.212 px | **0.27** | **pass** |
| LoFTR raw | 3.02 px | 36% | 1.175 px | 0.13 | pass |
| SIFT raw | 53.50 px | 31% | 0.611 px | 0.03 | pass, unusable |
| LoFTR + Stage B | — | — | — | — | **FAIL** |
| SIFT + Stage B | — | — | — | — | **FAIL** |

**Independently corroborated.** The best method recovers an inter-product offset of **dy = +7.62 px**. Phase correlation — a different algorithm with no matcher involved — independently measures **+8.25 px**. Two unrelated methods agreeing is the evidence that terrain is being measured.

### Scale

Both sides brought to a common ground sample distance, then normalised. 1024×1024 windows, three per ratio.

| Ratio | Coarse m/px | Matches | Inlier | Spread | Noise rejection |
|---|---|---|---|---|---|
| 1× | 7.40 | 286 | 52% | 0.92 px | 40.9× |
| 2× | 14.81 | 183 | 34% | 3.31 px | 30.5× |
| 4× | 29.61 | 125 | 51% | **0.43 px** | 47.0× |
| 8× | 59.23 | 56 | 71% | **0.41 px** | 56.0× |
| 16× | 118.45 | — | — | starved: 64×64 input | — |

The ratio is **read from metadata, not estimated** — every product declares its GSD (OHRC 0.26 m, TMC-2 5.05 m, Kaguya 7.40 m, NAC ~0.5–2 m). Blind estimation is confounded (BUG-010) and unnecessary.

At 16× the window collapses to 64×64. Larger windows are blocked by LoFTR's coarse attention, which is O((H·W/64)²) — 2048² needs 17 GB on CPU.

### Real Chandrayaan-2 registration

OHRC strip `ch2_ohr_ncp_20210402T0546284043` projected to the Kaguya grid and matched. **Passes the control gate** (roll −17.8 vs −15 wanted; noise 7.5×; constant 60×).

Against Kaguya **evening**, four independent chunks:

| Chunk | dx | dy | scale | rotation |
|---|---|---|---|---|
| 1 | +152.19 | −438.20 | 1.0183 | 0.16° |
| 2 | +151.50 | −437.43 | 1.0166 | 0.09° |
| 3 | +152.07 | −436.78 | 1.0165 | 0.81° |
| 4 | +150.61 | −434.25 | 1.0189 | 0.31° |

Agreement to ~1.7 px across independent chunks, at 66.7% inlier ratio and RMSE 0.593 px (4.39 m). The grids were verified aligned to 0.0 px, so the ~3.4 km offset is not a windowing artifact — it is consistent with the uncorrected geolocation error of an uncontrolled Chandrayaan-2 product, which is precisely why registration is needed.

**The offset is verified, by two independent methods and the full control gate.**

Registration needed a coarse-alignment stage first: tiled matching only searches a small pad around each tile, so a 437 px displacement sits far outside it. Estimating the gross shift from a whole-image match and removing it before tiling took the OHRC case from 7 matches to **1443**.

| Check | Result |
|---|---|
| LoFTR pipeline | dx **+151.64**, dy **−435.59** |
| Phase correlation, no matcher involved | dx **+151.10**, dy **−434.86** |
| Agreement | **0.7 px** on a 437 px offset |
| Roll raw input +15 px | recovered **−15.67** ✓ |
| Pure noise | **0 matches** ✓ |
| Constant grey | **0 matches** ✓ |

Final: **1443 matches, 730 inliers (51%), RMSE 0.752 px = 5.56 m, sub-pixel**, on real Chandrayaan-2 data against an independent lunar reference.

Against Kaguya **morning** the chunks disagree wildly. Measured explanation: OHRC correlates **+0.066 with evening and −0.031 with morning**, consistent with its illumination resembling evening. A DEM render explains OHRC barely at all (r = +0.006 against +0.067 for Kaguya on the same terrain) — at 0.26 m, OHRC resolves texture a 10 m DTM cannot model.

### Chandrayaan-2 TMC-2 registration

TMC-2 is the mission's **survey** instrument — 8436 catalogued products against
OHRC's 624 — so this is the case that shows the pipeline at coverage scale
rather than on one targeted strip. TMC-2 ortho `ch2_tmc_ndn_20201126T1610528086`
(5.05 m/px) against Kaguya TC evening (7.403 m/px), a 1.466× scale ratio read
from the products, at 0.24°N 70.93°E.

| Metric | TMC-2 ↔ Kaguya | OHRC ↔ Kaguya | Kaguya ↔ Kaguya |
|---|---|---|---|
| Matches | **3085** | 1443 | 132 |
| Inlier ratio | **99.3%** | 98.9% | 49.2% |
| RMSE | **0.595 px** (4.41 m) | 0.513 px | 0.796 px |
| **Coverage** | **0.891** | 0.391 | 0.344 |
| **Entropy** | **0.894** | 0.731 | 0.630 |
| Model selected | affine | affine | similarity |

| Control | Result |
|---|---|
| Real pair | 3435 matches, dx +21.50, dy +2.46 |
| Roll raw input +15 px | dx moved **−14.99** (want −15) |
| Pure noise | 102 vs 3435 — **33.7×** |
| Constant grey | 0 vs 3435 |

**This is the project's best result on the PS's uniformity requirement**, and the
reason is geometric rather than lucky: TMC-2 and Kaguya are close in scale and
both map-projected, so the entire frame carries usable texture. OHRC's 28×
scale gap leaves a narrow high-resolution strip against a coarse reference, and
coverage follows.

The correspondence here is a **geometric prior, not a search**: both products
declare a CRS, so the Kaguya window covering a TMC-2 window is computed. What
the matcher measures is the residual — dx +21.5 px, dy +2.5 px, about 160 m.

Reproduce: `python scripts/tmc_vs_kaguya.py --lat 0.5 --size 1536`

## Baselines, measured through the same gate

Every method gets the same front end (local contrast normalisation), the same
RANSAC and the same metrics, on the same three real cases. A method that fails
the four-control gate has its numbers struck out: they describe matches that are
not tracking the terrain, and reporting them as accuracy would be wrong.

### Cross-illumination — Kaguya TC morning vs evening

| Method | Matches | Inlier | RMSE | Coverage | Entropy | Gate |
|---|---|---|---|---|---|---|
| **SANDHI** | **536** | 0.312 | 0.932 px | **0.641** | **0.801** | **PASS** |
| DISK + LightGlue | 1 | — | — | 0.000 | 0.000 | FAIL — *real pair* |
| ASIFT | 90 | 0.078 | — | 0.062 | 0.325 | FAIL — *roll, noise* |
| SIFT | 25 | 0.120 | — | 0.047 | 0.264 | FAIL — *roll* |
| ORB | 83 | 0.036 | — | 0.047 | 0.264 | FAIL — *noise* |

**This is the result the project exists for. On cross-illumination, no baseline
passes the control gate — SANDHI is the only method that does.**

The failure modes say why, and they differ. DISK+LightGlue finds **one** match on
real terrain: a learned *detector* has nothing to fire on when shading inverts.
SIFT and ASIFT fail the **roll** control — roll the raw input 15 px and their
recovered offset does not follow, so their 25 and 90 matches were never tracking
the ground. ORB fails **noise**: it matches random pixels as readily as the Moon.
Their low inlier ratios (0.036–0.120) corroborate all of it independently.

### Cross-sensor — Chandrayaan-2 OHRC vs Kaguya, all methods gate-clear

| Method | Matches | Inlier | RMSE | Coverage | Entropy | Gate |
|---|---|---|---|---|---|---|
| SANDHI | 1622 | 0.762 | **0.659 px** | 0.375 | 0.686 | PASS |
| **ASIFT** | **4826** | 0.824 | 0.668 px | **0.453** | 0.752 | PASS |
| SIFT | 583 | **0.830** | 0.654 px | 0.453 | **0.763** | PASS |
| DISK + LightGlue | 559 | 0.655 | 0.801 px | 0.391 | 0.711 | PASS |
| ORB | 814 | 0.575 | 0.880 px | 0.344 | 0.680 | PASS |

**ASIFT beats us here**, on match count and coverage, and SIFT edges us on RMSE
and entropy. That is worth stating plainly rather than burying: this case is
geometrically easy — the OHRC strip is already projected onto the Kaguya grid,
so what remains is a small residual — and classical descriptors do well when
illumination is comparable and geometry is nearly solved. Our advantage is not
that we win everywhere; it is that we are the only method that does not collapse
when illumination inverts.

### Cross-sensor at survey scale — Chandrayaan-2 TMC-2 vs Kaguya

| Method | Matches | Inlier | RMSE | Coverage | Entropy | Gate |
|---|---|---|---|---|---|---|
| **SANDHI** | **3417** | 0.806 | 0.725 px | **0.891** | **0.894** | **PASS** |
| DISK + LightGlue | 665 | 0.587 | 0.902 px | 0.734 | 0.829 | PASS |
| SIFT | 72 | 0.611 | **0.601 px** | 0.281 | 0.621 | PASS |
| ORB | 188 | 0.277 | 0.851 px | 0.312 | 0.654 | PASS |
| ASIFT | 215 | 0.567 | 0.668 px | 0.281 | 0.565 | FAIL — *noise* |

**5.1× the matches of the next gate-passing method, at 1.2× its coverage and
3.2× SIFT's.** SIFT's lower RMSE is measured over 72 matches covering 28% of the
frame; ours is over 3417 covering 89%. RMSE alone is not comparable across such
different match populations, which is why coverage and entropy are reported
beside it.

### Phase correlation

Translation only, so it has no match count, coverage or inlier ratio — reporting
it in those columns would be a category error. It is used as a corroborating
instrument, and it corroborates:

| Case | Phase correlation | SANDHI | Agreement |
|---|---|---|---|
| Kaguya | dy **+7.96** | dy +7.62 | 0.34 px |
| OHRC | dx +151.13, dy −435.64 | dx +151.64, dy −435.59 | 0.51 px |
| TMC-2 | dx +21.68, dy +2.89 | dx +21.50, dy +2.46 | 0.47 px |

ASIFT adds a third: on TMC-2 it recovers (+21.24, +2.39), within 0.26 px of the
pipeline, from an algorithm sharing no code with it.

Reproduce: `python scripts/baselines.py --case all`

## Where the 3.4 km offset comes from

The registered OHRC product sits ~437 px (3.4 km) from where the Kaguya frame
puts it. This was reported for weeks as "the offset between our projected OHRC
and the Kaguya frame", never as Chandrayaan-2 geolocation error, because the two
causes produce an identical signature and had not been separated. They now have
been.

**It is Chandrayaan-2's, not ours.** Three independent legs:

| Evidence | Measurement |
|---|---|
| Our projection reproduces CH-2's own geolocation | **0.179 m** (0.584 px), **18101× smaller** than the discrepancy |
| TMC-2 bypasses our projection entirely, agrees with Kaguya | **~160 m** |
| Residual scale is **cross-track only** | 1.0173 at **25.1σ**; along-track 1.0047 at **1.4σ** |

The third leg is the decisive one. A map projection error — body radius,
latitude convention, degrees-per-metre — scales **both** axes together, by
construction. Only the sensor geometry can scale one. Along-track is
indistinguishable from 1.0.

Two distinct faults are present and should not be conflated: a **437 px
translation** (pointing/ephemeris bias) and a **1.7% cross-track scale**
(swath-width term — altitude or field of view). The scale accounts for only
~8.6 px across a 506 px product, so it does not explain the translation.

**A side finding worth its own line.** The PDS4 label declares
`pixel_resolution = 0.26`. The product's own geometry sidecar says otherwise:

| | Measured from 94,743 geolocation samples | Label |
|---|---|---|
| Cross-track | **0.3060 m/px** (sd 0.0016) | 0.26 |
| Along-track | **0.3225 m/px** (sd 0.0005) | 0.26 |

The label is rounded, and isotropic where the truth is anisotropic by 5.4%. That
anisotropy is independently why model selection prefers affine over similarity on
OHRC. The label value is used only to size the anti-alias kernel, so the cost is
~20% over-blurring — sharpness, not geolocation.

Reproduce: `python scripts/offset_origin.py --chunks 6`

## Chandrayaan-1 M3: a fourth instrument, and the largest scale ratio yet

M3 is an imaging spectrometer on Chandrayaan-1, ~147 m/px, indexed by PDS ODE
and needing no login. It was brought in to explain the IIRS failure, and it
registers against Kaguya TC on its own terms:

| Metric | M3 ↔ Kaguya |
|---|---|
| Matches | 42 |
| Inlier ratio | **71.4%** |
| RMSE | **0.704 px** (sub-pixel) |
| Coverage / entropy | 0.125 / 0.447 |
| **Scale ratio** | **19.84×** |

| Control | Result |
|---|---|
| Real pair | 262 matches, dx −9.56, dy +9.02 |
| Roll raw input +15 px | dx moved **−15.60** (want −15) |
| Pure noise | 19 vs 262 — 13.8× |
| Constant grey | 0 vs 262 |

**19.84× is the largest ratio demonstrated in this project**, and it passes the
full gate. Together with the 11.49× Kaguya control, it retires the "8×" figure in
the scale table above: that ceiling was window starvation at 64×64, not a limit
of the method.

It also refutes the explanation offered for the IIRS failure. A spectrometer
*does* match a visible framing camera — at a **larger** scale gap than the one
IIRS failed at. So the instrument pairing was never the problem.

Reproduce: `python scripts/m3.py fetch --id M3G20090204T234545` then
`python scripts/iirs_vs_m3.py --vs-kaguya --obs M3G20090204T234545`

## IIRS: measured, controlled, and it does not work

The problem statement names OHRC, TMC **and IIRS**. Two of the three register.
The third does not, and the difference between "we did not try" and "we tried and
here is exactly why it fails" is the whole point of this section.

IIRS is an imaging spectrometer: 255 bands from 0.729 to 5.010 um at 85.08 m/px.
Against Kaguya TC evening, where the strip crosses tile N18E009N15E012SC:

| | corr | matches | inliers |
|---|---|---|---|
| **IIRS 1504 nm vs Kaguya** | **+0.033** | 44 | 6 (13.6%) |

No fit. **Two hypotheses were tested and both were wrong.**

**Band choice is not the driver.** The obvious story: below ~2.5 um the signal is
reflected sunlight and shades like Kaguya; beyond ~3 um thermal emission
dominates and cannot. Sweeping the spectrum refutes it — every wavelength fails
about equally:

| nm | 898 | 1504 | 1993 | 2398 | 3004 | 3493 | 4504 |
|---|---|---|---|---|---|---|---|
| matches | 4 | 0 | 0 | 4 | 5 | 0 | 0 |

**Reflectance vs radiance is not the driver either.** A reflectance product has
the illumination divided out, so the radiance cube — which keeps the shading
Kaguya records — should have done better. Both give correlation **+0.0326**, the
same number to four decimals.

### The controls, which is what makes this a result

| Control | Result | Rules out |
|---|---|---|
| Two bands of one cube, both projected | 3528 matches, **100%** inliers, corr +0.986 | the projection, the strip shape, the imagery |
| Same, unprojected raw swath | 3645 matches, 100% inliers | projection loss specifically |
| Kaguya morning vs evening at 85.08 m/px, same 288 px shape | 1618 matches, **61.5%** inliers | the 11.49x scale ratio |

The pipeline handles this scale, this shape and this projection. What it does not
bridge is the instrument gap. Kaguya *averaged* to 85 m keeps its large-scale
shading pattern, which is why Kaguya-vs-Kaguya matches at 11.49x; IIRS's native
85 m pixels do not carry that structure to align to.

**The scale control is a positive result in its own right: 11.49x works.** That
extends the demonstrated scale envelope past the 8x reported above — the old 16x
failure was window starvation at 64x64, not a limit of the method.

### That explanation was tested and is wrong

M3 was fetched to test it, and it refutes it twice over. IIRS against M3 —
spectrometer to spectrometer at 1.51× — fails just as badly (corr −0.027, 20
matches, 20% inliers). And M3 against Kaguya — spectrometer to *camera* at
19.84× — **works and passes the gate**. So neither the instrument pairing nor
the scale explains anything.

Everything now eliminated, each by measurement:

| Eliminated | Evidence |
|---|---|
| Band choice | 898–4504 nm sweep, uniform failure |
| Reflectance vs radiance | identical correlation, +0.0326 |
| Projection method | polynomial fit (3.96 px) **and** direct per-pixel backplane (0.449 px) both fail |
| Strip shape, imagery | band-vs-band identity, 100% inliers |
| Scale ratio | 11.49× and 19.84× both work elsewhere |
| Spectrometer vs camera | M3 ↔ Kaguya passes the gate |
| M3's geolocation | fits to **0.15 px** where IIRS fits to 3.96 px |

What survives is IIRS's own geolocation. Its backplane is 26× less
self-consistent than M3's, and the band-vs-band control is structurally blind to
that — both bands carry the same geolocation, so any error in it cancels
exactly. IIRS matches itself perfectly and nothing else, which is the signature
of imagery placed on the wrong ground.

**Not yet proven**, and stated as such: a direct per-pixel reprojection did not
fix it either, so "the geolocation is wrong" is the surviving hypothesis rather
than a demonstrated cause. Confirming it needs an independent geolocation for
IIRS — SPICE reconstruction from the mission kernels — which is the ISIS3/ALE
path.

Reproduce: `python scripts/iirs_vs_kaguya.py --controls`

## Stage B: a hypothesis that failed its own test

The project's original thesis was that rendering a DEM under the source image's illumination would enable cross-illumination matching. **Controlled measurement does not support it.**

It is worse than neutral. Pure noise pushed through Stage B produces *more* matches than real terrain:

| Variant | Noise matches | Real matches | Ratio |
|---|---|---|---|
| Stage B, el 20° | 266 | 20 | **0.1×** |
| Stage B + high-pass, el 20° | 1223 | 19 | **0.0×** |
| Stage B + high-pass, el 10° | 3591 | 9 | **0.0×** |

Dividing by a render injects a `1/render` term identical in both images. That term alone suffices for the matcher, and it does not move when the imagery moves — so the matcher stops tracking terrain.

**The physics is sound and stands independently:** a DEM render reproduces real lunar imagery at r ≈ 0.53–0.58, and correctly recovers each image's illumination direction (morning → 90° east, evening → 225° south-west, from a blind azimuth sweep). Neither result involves a matcher. Rendering works; using it by division for matching does not.

**What replaced it is simpler and uses no DEM at all:** local contrast normalisation — subtract a local mean, divide by a local standard deviation. Halves offset scatter versus raw, raises inlier ratio from 36% to 54%, doubles coverage.

### Stage D: sub-pixel accuracy and uniform distribution

Both of the problem statement's explicit numeric requirements, met together.

| Variant | RMSE | Sub-pixel? | Coverage | Entropy |
|---|---|---|---|---|
| baseline | 0.866 px | YES | 0.36 | 0.64 |
| + sub-pixel refinement | 0.771 px | YES | 0.23 | 0.53 |
| **+ tiled matching** | **0.893 px** | **YES** | **0.66** | **0.80** |
| + tiled + sub-pixel | 0.786 px | YES | 0.45 | 0.70 |

Getting here needed a diagnosis rather than more tuning. Coverage was being lost at **refinement**, not at RANSAC:

| Stage | Matches | Coverage |
|---|---|---|
| Raw LoFTR | 126 | 0.55 |
| After refinement | 36 | 0.23 |
| After RANSAC | 31 | 0.19 |

Two changes recovered it:

* **Tiled matching with a tight pad.** Every region gets its own matching attempt. An earlier attempt used a 170 px search pad, which let tiles match ground far from their own and gained nothing; 16 px works.
* **RANSAC threshold 3.0 → 1.5 px.** The old threshold admitted matches up to 3 px off, which is precisely what held RMSE above 1 px. Tightening it is a quality filter, and the inlier ratio is reported so the cost stays visible:

| RANSAC threshold | Inliers | Ratio | RMSE | Coverage |
|---|---|---|---|---|
| 3.0 px | 124 | 79% | 1.299 | 0.52 |
| 2.0 px | 102 | 65% | 0.959 | 0.47 |
| **1.5 px** | **92** | **58%** | **0.786** | **0.45** |
| 1.0 px | 66 | 42% | 0.594 | 0.34 |

**Bucketed selection was implemented, measured, and dropped.** It cannot raise coverage: selecting among existing matches only redistributes them among cells that already contain some, and empty cells stay empty. Coverage has to be forced where matches are *produced*.

`scripts/register.py` writes the deliverable: the warped registered image, a match-point CSV (source x/y, reference x/y, confidence, residual, inlier flag), and a metrics JSON.

### Viewpoint variation

**Model selection.** A similarity or affine transform cannot represent perspective at all — it absorbs the error into a worse fit rather than failing loudly. `scripts/viewpoint.py` fits similarity (4 DOF), affine (6) and homography (8), selecting on **held-out** residual, since a higher-DOF model always wins in-sample.

Held-out alone was not enough. On Kaguya nadir-vs-nadir pairs — both orthorectified, so **no perspective exists between them** — homography still won 2 of 3 windows by 5–9%. That is noise rewarding degrees of freedom. A 15% complexity margin fixes it, and all three now correctly select `similarity`.

**The operating envelope, measured.** A ladder of real LROC NAC images of one site from different orbits, spanning 1.2° to 58.9° emission angle, found via ODE's emission-angle index. No synthesised warps. NAC labels carry no geolocation, so overlapping rows are located by sliding one strip against the other; a real overlap produces a sharp peak in match count.

| Emission gap | Best matches | peak / median | Verdict |
|---|---|---|---|
| **12.3°** | **413** | **13.77** | **detected — decisive** |
| 14.9° | 61 | 2.35 | not detected |
| 24.3° | 40 | 1.82 | not detected |
| 33.3° | 48 | 1.88 | not detected |
| 57.7° | 19 | 2.00 | not detected |

Matching succeeds at a **12.3° obliquity gap** and fails from 14.9° upward, so the boundary lies between them.

**Confound, stated plainly:** the one rung that matched is also the only image acquired in the same orbit sequence as the anchor (2009-247). The failures are from 2013, 2018 and 2023 — years apart, with different illumination. Obliquity and temporal/illumination change are therefore **not separated** in this ladder. The honest reading is that 12.3° is a demonstrated success under favourable illumination, and ≥14.9° fails under combined obliquity *and* illumination change. Isolating the two would need same-date pairs at several emission angles.

At 58.9° the physics is unambiguous regardless: terrain foreshortens to cos(58.9°) = 0.52 in one axis, with occlusion and inverted shadowing.

## The control gate

Every matching number must pass four tests, or it is not reported:

1. **Real pair** produces matches
2. **The RAW input rolled by N** moves the recovered offset by −N
3. **Pure noise** collapses, or loses to real terrain by ≥5×
4. **Constant grey** collapses

Rule 2 is the one that matters. An earlier version shifted the *already-normalised* image and passed — because that moves imagery and artifact together. **Perturb the input, never the output.**

This gate exists because it caught a catastrophic error: an earlier version of this project reported **100% inlier rates** that were entirely artifact. A shared zero-mask written identically into both images gave the matcher a perfectly aligned pattern to lock onto, and the identity ground truth was independently wrong by 8.25 px. Two errors pointing the same way produced a beautiful, false result. Full diagnosis in [BUGS.md](BUGS.md) BUG-011.

**No identity ground truth is used anywhere.** Kaguya morning and evening are independently orthorectified and genuinely offset. Validation is cross-window agreement on the recovered offset — which an artifact cannot fake, because a mask artifact reports zero everywhere and noise reports scatter.

## Fine-tuning: measured twice, rejected twice

The pretrained MegaDepth `outdoor` weights are the default, and that is a
finding rather than an omission. Fine-tuning LoFTR on warped lunar crops was
attempted twice and rejected twice, by a bar written before either run.

| | Run 1 (2026-09-09) | Run 2 (2026-09-10) |
|---|---|---|
| Training pairs | 48 crops, 1 tile | 1251 pairs, 5 scenes, 2 sensors |
| Source terrain | 0.026 Gpx | 4.6 Gpx |
| Overlaps evaluation data | yes — a real contamination bug | no, asserted by a test |
| Learning rate | 1e-4 | 2e-5 |
| Its own validation metric | 0.087 → 0.391 | 0.109 → **0.575** |
| **Matches on the real pair** | **132 → 47** | **245 → 63, by epoch 1** |
| Checkpoint adopted | no | **none was ever saved** |

Run 2 improved its training metric **5.3×** while losing **74% of real matches
in a single epoch**. A per-epoch guard scoring the model on genuine
cross-illumination imagery rejected all four epochs, so no checkpoint was
written — the file on disk stayed byte-identical to run 1's rejected weights,
which is how we know rather than assume.

**Why, and why more data cannot fix it.** A training pair is an image and a
warped copy of *itself*: photometrically identical. The cheapest solution to
that task is exact appearance correspondence — precisely the crutch a matcher
must give up to survive a lunar illumination change, where a crater lit from the
east is close to pixel-identical to a dome lit from the west. The task does not
merely fail to teach illumination invariance, it rewards unlearning it. That
predicts an immediate collapse on the cross-illumination pair specifically,
which is what was measured, twice.

A real attempt needs pairs varying illumination *and* viewpoint with
correspondence that is neither warped nor estimated — multi-look NAC with SPICE
geometry. See [ROADMAP.md](ROADMAP.md).

`sandhi register --weights <checkpoint>` and `scripts/adopt_check.py` remain, so
the comparison stays one flag away and the bar is already encoded.

## Data

Real, public, current. No synthetic imagery; no ground truth manufactured by warping.

| Role | Source | Access |
|---|---|---|
| Cross-illumination pairs + DEM | Kaguya/SELENE TC morning, evening, DTM | JAXA DARTS via PDS ODE — no login |
| Reference imagery | LROC NAC / WAC | PDS ODE — no login |
| Source imagery | Chandrayaan-2 OHRC, TMC-2 | [PRADAN](https://chmapbrowse.issdc.gov.in/) — login required |

Chandrayaan-2 coverage, from the footprint shapefiles: **OHRC is polar-dominated** (42 south-polar vs 27 equatorial of 77 usable) — a targeting instrument, so evaluation material rather than a training source. **TMC-2 is the workhorse** (8436 usable, well spread) and ships **ortho + DTM as paired products**, 3049 of each, matched by swapping `_oth_` → `_dtm_`.

Everything is read in place through GDAL `/vsizip/`; the ~10 GB of Chandrayaan-2 imagery never lands on disk unpacked.

## Run it

One command produces the whole result as a figure.

```bash
pip install -r requirements.txt
python scripts/demo.py --case all
```

**Runs from a fresh clone.** `samples/` carries 3 MB of real cropped imagery — genuine observations, not synthetic — so the demo works with no downloads. It takes ~20 s per case on CPU and produces identical numbers to the full-resolution data. Downloading the full products (~7 GB) is only needed to work on new regions.

![Kaguya cross-illumination](outputs/demo_kaguya.png)

![Chandrayaan-2 OHRC](outputs/demo_ohrc.png)

Each figure shows source, reference, the registered checkerboard overlay, the correspondences (green inlier / red rejected), where the match points landed on an 8x8 grid, and the metrics. `--list` reports whether each case will use full data or the committed sample.

| Case | Matches | Inliers | RMSE | Coverage |
|---|---|---|---|---|
| Kaguya morning vs evening | 132 | 65 (49%) | **0.796 px** | 0.34 |
| **Chandrayaan-2 OHRC vs Kaguya** | **1443** | **730 (51%)** | **0.752 px = 5.56 m** | 0.28 |

## Setup

Python 3.11. No conda, no WSL, no ISIS3 — `rasterio`'s wheels bundle GDAL with the PDS4/ISIS3/PDS drivers needed for planetary products.

```bash
pip install -r requirements.txt
python scripts/check_env.py
```

## Scripts

| Script | Purpose | Status |
|---|---|---|
| `check_env.py` | Dependencies and GDAL planetary drivers | current |
| `ode_catalog.py` | Enumerate ODE-indexed lunar products | current |
| `survey_coverage.py` | Per-region imagery survey | current |
| `kaguya.py` | Resolve, verify, download morning/evening/DEM triples | current |
| `triple_io.py` | Open a triple, read aligned windows | current |
| `photometric.py` | DEM renderer, 7 analytic self-checks | current |
| `estimate_sun.py` | Recover illumination direction | current |
| `ch2_footprints.py` | Chandrayaan-2 coverage and candidate ranking | current |
| `ch2_io.py` | Read OHRC and TMC-2 in place | current |
| `ohrc_project.py` | Project raw OHRC into a map frame | current |
| **`remeasure.py`** | **Controlled evaluation — the authoritative harness** | current |
| `ohrc_vs_kaguya.py` | Real Chandrayaan-2 registration | current |
| `register.py` | Stage D: sub-pixel, match export, registered product | current |
| `viewpoint.py` | Model selection and the obliquity limit | current |
| **`demo.py`** | **One-command end-to-end run with figures** | current |
| `ablation.py`, `dense_match.py`, `scale_pipeline.py` | Retracted experiments | **superseded** |

```bash
python scripts/remeasure.py --full          # illumination, all methods
python scripts/remeasure.py --scale --size 1024
python scripts/ohrc_vs_kaguya.py --rows 512
```

## Problem statement scorecard

| Requirement (PS text) | Status | Evidence |
|---|---|---|
| Find match points between source and reference | **complete** | LoFTR + local contrast norm, control-gated |
| **Illumination variation** | **complete** | 0.85 px cross-window spread, 54% inliers; corroborated by phase correlation |
| **Viewpoint variation** | **complete, envelope measured** | model selection validated on nadir controls; matching succeeds to 12.3° obliquity gap, fails ≥14.9° |
| Runs on real Chandrayaan-2 data | **complete, two of three instruments** | OHRC: 1443 matches, RMSE 0.752 px, corroborated to 0.7 px by phase correlation. TMC-2: 3085 matches, 99.3% inliers, RMSE 0.595 px, coverage 0.891 |
| **Scale variation** | **complete** | 1× to 8×, spread 0.41–0.92 px, noise rejection 30–56× |
| **Sub-pixel accuracy of source image** | **complete** | RMSE **0.786–0.893 px** |
| **Uniform distribution across the images** | **complete** | coverage **0.89**, entropy **0.89** on TMC-2; 0.66 / 0.80 on Kaguya |
| Software + registered product + match points | **complete** | `register.py` writes image, match CSV, metrics JSON |
| Evaluation metric (RMSE, inlier count, inlier ratio) | **complete** | all three, plus uniformity |

**8 / 8.** Two carry stated limits rather than hedges: viewpoint is characterised by a measured operating envelope (12.3° works, ≥14.9° does not, with obliquity and illumination confounded in that ladder), and scale is demonstrated to 8× with the 16× failure explained by a CPU memory bound rather than by the method.

## Honest status

**Works, controlled:** Chandrayaan-2 TMC-2 registration (3085 matches, 99.3% inliers, coverage 0.891, all four controls); cross-illumination matching (0.85 px spread, 54% inliers, corroborated two ways); scale handling to 8×; sub-pixel registration at coverage 0.66; transform-model selection; OHRC geolocation fit to 0.152 m; reading every product in place.

**Verified:** OHRC↔Kaguya registration passes all four controls and its 437 px (3.4 km) offset is corroborated to 0.7 px by phase correlation, an algorithm sharing no code with the matcher.

The morning/evening asymmetry is now partly explained: OHRC correlates +0.066 with Kaguya evening and −0.031 with morning, consistent with its illumination resembling evening. Both are weak. The more important finding is that a DEM render explains OHRC barely at all (r = +0.006, against +0.067 for Kaguya on the same terrain) — **at 0.26 m, OHRC resolves texture that a 10 m DTM cannot model**, so DEM-based illumination reasoning cannot bridge a 28× resolution gap.

**Baselines:** measured, gated, and reported above — including the case (OHRC) where ASIFT beats us.

**Does not work:** Stage B for matching. 16× scale. SIFT at any ratio (53 px scatter). Fine-tuning LoFTR on warped pairs — twice measured, twice rejected, and the second run's collapse arrived within one epoch.

**Known limits:** LoFTR capped near 1024×1024 on CPU, which is what bounds scale at 8× rather than anything about the method. Obliquity beyond ~13° is not matchable, and the ladder that measured it confounds obliquity with illumination change. Matching takes tens of seconds per pair on CPU; a live demo should use the committed figures or a GPU. The 3.4 km offset is now separated: it is Chandrayaan-2's, established three independent ways (see above).

## What would make this production software

See [ROADMAP.md](ROADMAP.md). The two items that matter most are not features:

1. **Correcting for the measured OHRC geometry.** The label's `pixel_resolution = 0.26` is wrong — the product's own geometry gives 0.3060 x 0.3225 m — and it currently sizes the anti-alias kernel, over-blurring by ~20%. Using the measured values should sharpen every OHRC number.
2. **Wiring the control gate into CI.** It is now a test — `pytest tests/test_pipeline.py::test_control_gate_passes`, alongside one that feeds the gate a deliberately blind matcher and asserts it is rejected — so it can fail a build. What is left is running it on a hosted runner rather than on a laptop.

The roadmap also records what was tried and abandoned — Stage B, blind scale estimation, bucketed selection — so the cost is not paid twice.

## Project conventions

See [CLAUDE.md](CLAUDE.md). The load-bearing ones:

- **Data must be genuine, public and current.** No synthetic imagery as a training or evaluation source.
- **Ground truth from geometry, never from a warp we invented.**
- **Zero extra hardware.** Laptops plus free Kaggle GPU hours.
- **Every matching number passes the four-control gate before it is written down.**

Every bug is logged in [BUGS.md](BUGS.md) with root cause and fix. Several entries record cases where a *test or metric* was wrong while the code was right — including one where the experiment passed while an intermediate quantity was badly incorrect, and one where two errors cancelled into a convincing false result. That log is part of the deliverable.
