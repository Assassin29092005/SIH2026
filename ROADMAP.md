# ROADMAP — from prototype to running software

Where this project actually stands, and what stands between it and software someone other than us could depend on.

Written 2026-08-23. Nothing here is scheduled; it is a list of what is missing and why it matters.

## What exists today

A validated method with a demo. All eight problem-statement requirements are met with control-gated evidence, one command produces a figure, and a fresh clone runs from committed samples.

What that is **not**: a program someone else can point at their own imagery and trust. Everything below is the gap.

---

## Phase 1 — Correctness gaps that could change conclusions

These are open scientific questions, not polish. Each one could move a number that is currently reported as a result.

### 1.1 Separate Chandrayaan-2 geolocation error from our own projection error — **ANSWERED 2026-09-10**

**It is Chandrayaan-2's, not ours.** Three independent legs, none of which
relies on the others:

1. **Our projection reproduces CH-2's own geolocation to 0.179 m.** The degree-3
   inverse fit lands within 0.584 px of the geometry sidecar, which is 18101x
   smaller than the 3.4 km under investigation. Whatever error the CSV carries,
   we inherit it rather than create it.
2. **TMC-2 bypasses our projection entirely and agrees with Kaguya to ~160 m.**
   It ships as a georeferenced GeoTIFF, so it never touches the polynomial. The
   Kaguya frame, the matcher, the normalisation and the RANSAC are therefore not
   the source of a kilometre-scale discrepancy.
3. **The residual scale is cross-track only:** 1.0173 at **25.1 sigma**, with
   along-track at 1.0047, **1.4 sigma** — indistinguishable from 1.0. A map
   projection error (body radius, latitude convention, degrees-per-metre) scales
   both axes together by construction. Only the sensor geometry can scale one.

Two distinct terms are present and should not be conflated: a **~437 px (3.4 km)
translation**, which is a pointing/ephemeris bias, and a **1.7% cross-track
scale**, which is a swath-width term (altitude or field of view). The scale
accounts for only ~8.6 px across the 506 px product, so it does not explain the
translation — they are separate faults that happen to appear together.

Corroborating, from the same investigation: the PDS4 label declares
`pixel_resolution = 0.26`, but the product's own geometry gives **0.3060 m
cross-track and 0.3225 m along-track** (sd 0.0016 and 0.0005 over the full
20 km strip) — the label is rounded and isotropic where the truth is neither.
That anisotropy is also why model selection picks affine over similarity for
OHRC. The label value is used only for the anti-alias kernel, so it costs
sharpness (~20% over-blur) rather than geolocation.

Reproduce: `python scripts/offset_origin.py --chunks 6`

### 1.2 Isolate obliquity from illumination in the viewpoint envelope
The obliquity ladder shows matching succeeds at a 12.3° emission gap and fails from 14.9°. But the one success is also the only image acquired in the same orbit sequence as the anchor; the failures are years apart with different illumination. **Obliquity and illumination are confounded.**

Needs same-date NAC pairs across several emission angles. Until then, 12.3° is a demonstrated success under favourable conditions, not a measured obliquity limit.

### 1.3 Extend scale beyond 8×
16× fails because a 1024 px window decimates to 64×64, and larger windows exceed CPU memory — LoFTR's coarse attention is O((H·W/64)²), so 2048² needs 17 GB. This is a resource bound, not a method bound. A GPU, or a coarse-to-fine cascade that never holds a large attention matrix, should reach the ~19× and ~28× ratios the real OHRC→TMC-2 and OHRC→Kaguya cases need.

### 1.4 Non-rigid deformation
Only similarity, affine and homography are fitted. Real lunar registration over long strips has terrain-induced distortion no global model captures. Thin-plate splines or a local mesh would fit it — and the existing held-out model-selection framework already provides the honest way to decide whether the extra freedom is earned.

### 1.5 IIRS — **MEASURED 2026-09-10, does not register**

Done, and negative. IIRS 1504 nm against Kaguya gives correlation +0.033, 44
matches, 6 inliers. Three controls isolate the cause: two bands of the same cube
match at 100% inliers projected and unprojected (so the projection, strip shape
and imagery are all fine), and Kaguya against itself decimated to 85.08 m/px in
the same 288 px shape matches at 61.5% (so the 11.49x ratio is fine). The gap is
the instrument pairing itself -- an infrared spectrometer against a visible
framing camera at 85 m/px.

Two hypotheses were tested and refuted: band choice (the whole 898-4504 nm sweep
fails equally) and reflectance-vs-radiance (identical correlation, +0.0326).

**Next, if revived:** Chandrayaan-1 M3, an imaging spectrometer at ~140 m/px that
IS indexed by PDS ODE and needs no PRADAN login. IIRS-to-M3 is
spectrometer-to-spectrometer at 1.6x rather than spectrometer-to-camera at 11.5x,
which is the pairing the controls suggest should work.

`python scripts/iirs_vs_kaguya.py --controls`

---

### 1.6 OHRC against LROC NAC — the reference is what blocks the PS sub-pixel clause
The PS asks for "sub-pixel accuracy **of source image**". OHRC's product is 0.26 m/px; the reference it is currently matched against, Kaguya TC, is 7.403 m/px. Meeting the clause in OHRC's own pixels would mean locating a feature to **0.035 reference pixels** — finer than the reference image itself resolves. No matcher achieves that, so this is a data pairing problem, not an algorithm problem. Measured today: 0.513 grid px = 3.80 m = **14.6 OHRC px** (BUGS.md BUG-025).

**LROC NAC at ~0.5 m/px is the reference that makes it reachable.** Sub-0.26 m then means sub-0.52 NAC pixels, inside the 0.5–0.8 reference-pixel range this pipeline already delivers. The scale gap also drops from 28.5x to **1.9x**, well inside the demonstrated envelope, so this is an easier registration than the one already passing.

What it needs, in order:

1. **Footprint intersection.** `ch2_footprints.py --pick ohrc` already ranks OHRC products by NAC coverage across illumination bins, and reports that the six OHRC products nearest the equator all have NAC in all four bins. That picker output is the input to this step; it has not been acted on.
2. **Check the six NAC products on disk first.** `data/raw/nac/` holds `M106719774LC`, `M106726943RC`, `M1114007294RC`, `M1118716779RC`, `M1274103575RC`, `M1443174042RC` — 3.0 GB, downloaded 2026-08-23, and **no script reads them.** Confirm whether any overlaps the OHRC strip at ~0.6°N 23.4°E before downloading more.
3. **NAC ingestion.** NAC EDR/CDR are PDS3 pushbroom, not map-projected: same class of problem as OHRC, so `ohrc_project.py`'s inverse-polynomial approach transfers. Budget for the geometry, not the I/O.
4. **Register at 1.9x**, then report `rmse_source_px` — which the pipeline now emits, so the clause is checkable straight from the metrics JSON rather than by hand.
5. **Gate it.** No number leaves this step without `scripts/remeasure.py`'s four controls, same as every other result here.

The honest framing until this is done: the clause is **met on TMC-2** at 0.872 source px, and OHRC is reference-limited with a quantified reason and a known fix. Do not quote OHRC's 0.513 grid px as if it were the source-pixel figure.

## Phase 2 — Making it software rather than scripts

### 2.1 One package, one entry point
Nineteen scripts in `scripts/`, imported by path manipulation, with cross-imports between experiment files. Should be an installable package with a single CLI:

```
lunareg register  --source X --reference Y --out DIR
lunareg fetch     --site ... --instrument ...
lunareg evaluate  --controls
```

### 2.2 Configuration instead of module constants
`RANSAC_PX`, `TILES`, `TILE_PAD`, `POOL_ELEVATIONS`, `REFINE_MAX_SHIFT` are module-level constants edited in place. They belong in a config file with the measured defaults documented, since several were chosen by experiment and that reasoning currently lives only in comments.

### 2.3 A real test suite
There are self-checks (`--self-check`) that assert real properties, which is better than nothing, but they need downloaded data and take minutes. Needs a `pytest` suite split into fast unit tests on the committed samples and slow integration tests behind a marker, plus CI.

**Critically: the four-part control gate must be a test, not a script.** It is the thing that caught BUG-011, and it should fail a build.

### 2.4 Georeferenced outputs
`register.py` writes a PNG. It should write a GeoTIFF carrying the reference CRS and transform, so the registered product is usable in GIS rather than only viewable.

### 2.5 Whole-product processing
Everything operates on windows. Real use means registering a full OHRC strip (12000 × 78175) against a reference, which needs tiling with overlap, per-tile models, a global consistency check, and a merged output. The chunked loop in `ohrc_vs_kaguya.py` is a sketch of this, not an implementation.

### 2.6 Logging, errors, resume
`print()` throughout; no structured logging. Downloads have resume, but nothing else does — a failed run of a long job starts over. Missing or corrupt inputs mostly surface as exceptions rather than diagnostics.

---

## Phase 3 — Performance

### 3.1 GPU
Everything runs on CPU: 15–27 s per pair. A GPU would cut that to roughly 1–2 s and simultaneously remove the memory bound in 1.3. The code is already `torch`, so this is device placement rather than a rewrite.

### 3.2 Model size
LoFTR's MegaDepth weights are used untouched. Fine-tuning on lunar imagery — the Kaguya morning/evening corpus is the obvious training set — should improve the 51–55% inlier ratio and possibly the obliquity ceiling. **This has never been attempted.** It is also the only item here that would need meaningful GPU hours.

---

## Phase 4 — Usability

### 4.1 Data acquisition
Kaguya and NAC are scripted through ODE. Chandrayaan-2 is manual: log into PRADAN, browse, download. Automating it needs credential handling, and PRADAN offers no API — only a bulk-download script generated from the browser.

### 4.2 Interface
A PNG figure. A web UI where a user uploads two images, watches the match overlay, and downloads the registered product would make it usable by someone who does not read Python.

### 4.3 Documentation split
README serves developers and reviewers simultaneously. Real software separates a user guide from the engineering record.

---

## Deliberately not planned

- **Stage B / DEM-based photometric normalisation.** Tested, failed its controls, documented in BUGS.md BUG-011. Do not revive it without new evidence.
- **Blind scale estimation.** Confounded (BUG-010), and unnecessary since every product declares its GSD.
- **Bucketed match selection.** Implemented, measured, cannot raise coverage. The fix was tiled matching.
- **LoFTR fine-tuned on warped Kaguya crops, as run on 2026-09-09.** Not the idea — *that* run. It improved its own validation metric 4.5x (coarse-cell precision 0.087 -> 0.391 on a held-out tile) and destroyed real-pair matching: Kaguya 132 -> 47 matches, OHRC **1443 -> 6**. The control gate still passed, which is the point — the gate detects fabricated matches, not a model that has simply become worse. Diagnosis: the split holds out one of only two downloaded tiles, so training saw **48 warped crops of a single 3x3 degree patch**, and 8 epochs at lr 1e-4 was enough to overwrite the general MegaDepth features the cross-sensor OHRC case depends on. Do not re-run it in that configuration. See "Fine-tuning, next attempt" below.

Each of these cost real time. They are listed so nobody spends that time again.

---

## Fine-tuning on warped pairs — closed, twice measured

Two runs, seven weeks of ideas apart in content and one day apart in time. Every
fault diagnosed after the first was fixed before the second. The outcome did not
move.

| | Run 1 (2026-09-09) | Run 2 (2026-09-10) |
|---|---|---|
| Training pairs | 48 crops, 1 tile | 1251 pairs, 5 scenes |
| Source terrain | 0.026 Gpx | 4.6 Gpx, two sensors |
| Overlaps evaluation data | yes (BUG-017) | no, asserted by test |
| Learning rate | 1e-4 | 2e-5 |
| CNN backbone | frozen | frozen |
| Warped val precision | 0.087 -> 0.391 | 0.109 -> 0.575 |
| **Real-pair matches** | **132 -> 47** | **245 -> 63 by epoch 1** |
| Checkpoint adopted | no | **none ever saved** |

Run 2 improved its training metric **5.3x** and lost **74% of real matches
within a single epoch**. The per-epoch real-pair guard rejected all four epochs,
so `loftr_lunar_best.pt` was never written — the file on disk stayed
byte-identical to run 1's rejected checkpoint, which is how we know.

**Why it fails, and why more data cannot fix it.** A training pair here is an
image and a warped copy of *itself*. The two views are photometrically
identical, so the cheapest solution to the training task is exact appearance
correspondence — and that is precisely the crutch a matcher must give up to
survive a lunar illumination change, where a crater lit from the east is close
to pixel-identical to a dome lit from the west. The task does not merely fail to
teach illumination invariance; it rewards unlearning it. That predicts the
collapse to appear on the cross-illumination pair specifically and immediately,
which is exactly what was measured.

**What a real attempt would need.** Pairs that vary illumination *and* viewpoint
with correspondence that is neither warped nor estimated. That means real
multi-look NAC imagery with geometry from SPICE, which needs the ISIS3 / ALE
path CLAUDE.md budgets a week for. Until that exists there is no honest label
for the axis that matters, and the pretrained MegaDepth `outdoor` weights remain
the default in `sandhi/config.py`.

`--weights` and `scripts/adopt_check.py` stay: the comparison is one flag away
and the bar is already encoded, so a future attempt is cheap to judge.

---

## Suggested order

1. **1.1** — it decides whether the headline number means what we say it means ✔ answered
2. **1.6** — OHRC ↔ NAC. The only PS clause not fully met, the fix is known, and step 2 of it is free: six NAC products are already on disk and unread
3. **2.3** — lock the control gate into CI before the codebase grows
4. **3.1** — GPU, which unblocks 1.3 and makes everything else faster to iterate on
5. **2.1 / 2.2** — package it, once the science has stopped moving
6. Everything else by need

Items 1.1 and 2.3 are the two that protect against being wrong. 1.6 is the one that closes a stated requirement. The rest make it pleasant.
