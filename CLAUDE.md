# CLAUDE.md

Project instructions. Read fully before touching code.

## What this is

SIH 2026 entry for **PS SIH26166** (ISRO / Department of Space, Software, Space Technology).

**Title:** Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)

**Deliverable per the PS:**
- Generic software finding correspondence between Chandrayaan-2 optical images and lunar reference images, at **sub-pixel accuracy**, with match points **uniformly distributed across the image**.
- Registered product + the corresponding match points.
- Evaluation metrics (RMSE, inlier match count, inlier ratio, etc.).

Idea-submission deadline: **20 September 2026**.

## Hard constraints

These are non-negotiable. They override convenience.

1. **Data must be genuine, public, and current.** Real observations only. No synthetic or simulated imagery as a training or evaluation source. Every dataset needs a live public access path, verified before use.
2. **Ground truth must come from geometry, never from a warp we invented.** Warping one image to manufacture correspondence pairs is synthetic GT and is banned. Use map-projected images that share a controlled lunar frame, so correspondence follows from projection.
   - Rotation/crop/photometric jitter on *real* images is augmentation, not synthesis. Allowed. Document it wherever results are reported.
3. **₹0 extra hardware.** Laptops only. GPU work runs on free Kaggle (~30 GPU-hr/week) or Colab. If a step needs hardware we do not own, redesign the step.
4. **Fine-tune, never train from scratch.** The free GPU budget does not cover training a matcher from zero. Any plan that requires it is wrong.

## Approach

Four stages. The differentiator is Stage B — do not let it get dropped for expedience.

- **A. Coarse localization.** PDS4 labels carry sub-solar lat/lon, incidence/emission/phase angles, and SPICE-derived pointing. Use them as a positional prior so we never search globally.
- **B. Photometric normalization.** Render the LOLA/GLD100 DEM patch under the *source image's* sun geometry (Lommel–Seeliger or Hapke). This converts cross-illumination matching into same-illumination matching.
  - Why it matters: the Moon has no atmosphere, so no diffuse fill light. A crater lit from the east is near pixel-identical to a dome lit from the west. Classical descriptors do not merely degrade on this — they return confident wrong matches. Stage B sidesteps invariance instead of fighting it.
  - The on/off ablation of this stage, plotted against Δsun-azimuth, is our central scientific claim. Keep it runnable at all times.
- **C. Dense matching + sub-pixel refinement.** Lunar terrain is texture-poor, so detector-based methods starve. Use a dense/semi-dense matcher (LoFTR / RoMa class), fine-tuned. Sub-pixel comes from soft-argmax over the correlation volume plus a local quadratic fit on the peak. Without this head we cannot claim sub-pixel, and the PS demands it.
- **D. Robust fit.** MAGSAC++ per tile over affine/homography, then a global TPS or polynomial warp for the registered product.

## Data sources

All verified reachable as of 2026-08-22.

| Role | Source | Access |
|---|---|---|
| Source imagery | Chandrayaan-2 OHRC (~0.25 m/px), TMC-2 (~5 m/px), IIRS (~80 m/px) | [PRADAN](https://pradan.issdc.gov.in/ch2/) / [chmapbrowse](https://chmapbrowse.issdc.gov.in/) — free signup, PDS4, no lock-in |
| Reference imagery | LROC NAC (~0.5 m/px), WAC (~100 m/px) | [LROC downloads](https://lroc.im-ldi.com/images/downloads), [QuickMap](https://quickmap.lroc.im-ldi.com/), PDS ODE |
| Topography | LOLA + GLD100 DEM (~118 m/px) | PDS Geosciences Node / USGS Astrogeology |
| Cross-illumination pairs | Kaguya/SELENE Terrain Camera **morning and evening global mosaics** | JAXA SELENE archive / USGS Astrogeology |

The Kaguya TC morning/evening mosaics are the same terrain, globally, under opposite illumination, already co-registered. Free, real, pre-built cross-illumination training data. This is the highest-value dataset in the list — the PS's own dataset field hints at it (truncated `SE`).

The PS dataset field also says "specific datasets link will be provided - TBD". **Write the data loader against an abstract image-pair interface** so an official ISRO eval set can be swapped in without a rewrite.

### Verified access, 2026-08-22

Use the **PDS ODE REST API** (`https://oderest.rsl.wustl.edu/live2/`) for everything except Chandrayaan-2. Public, no key, no login. Parameter reference cached at `docs/ode_rest_params.txt` — **read it before adding any filter**, and see `BUGS.md` BUG-002 for why.

- **Chandrayaan-2 is NOT indexed by ODE.** Only Chandrayaan-1 M3 is. CH-2 OHRC/TMC-2/IIRS requires a PRADAN account. Someone must register at [chmapbrowse](https://chmapbrowse.issdc.gov.in/) — this is a hard blocker on the source-image side and should happen immediately.
- **LROC NAC:** `ihid=LRO&iid=LROC&pt=CDRNAC4`, 2,887,274 calibrated products, footprints and incidence angles both indexed and queryable.
- **Kaguya/SELENE TC:** `ihid=SLN&iid=TC`. `TCMORM` (morning, 7,200 tiles), `TCEVEM` (evening, 7,200), `TCDTMM` (DEM, 7,200) share one 3°×3° grid and differ only in the product-id stem:
  - `TCO_MAPM04_N21E009N18E012SC` / `TCO_MAPE04_...` / `DTM_MAP_02_...`
  - So a complete (morning, evening, DEM) triple resolves by string substitution. ~288 MB per file, direct HTTP from JAXA DARTS, no login. Note the URL path uses lowercase `m04`/`e04` while the filename is uppercase.
  - This is the Stage-B training set: co-registered same-terrain imagery under opposite illumination, plus the DEM the photometric renderer needs.

**Verified from real labels (tile `N18E009N15E012SC`): correspondence within a triple is the identity.** All three products are SIMPLE CYLINDRICAL, `MAP_RESOLUTION` 4096 px/deg (`MAP_SCALE` 7.403 m/px), identical `LINE_PROJECTION_OFFSET` / `SAMPLE_PROJECTION_OFFSET`, identical bounds, 12288 x 12288, 16-bit. So pixel (i,j) in morning **is** pixel (i,j) in evening **is** pixel (i,j) in the DEM.

Consequences — do not lose these:
- Stage-B training crops need no registration step. Crop the same window index from all three and the pair is exactly aligned. Ground truth is geometric, not estimated.
- **`SAMPLE_TYPE` differs by product: images are `MSB_UNSIGNED_INTEGER`, the DEM is `MSB_INTEGER` (signed).** Reading the DEM as unsigned turns every negative elevation into a huge positive. Both are big-endian.
- Verify with `python scripts/kaguya.py labels --site <name>`, which fails loudly if the grid ever disagrees.

**Site selection matters more than expected.** At the Chandrayaan-3 landing site (69°S) NAC coverage is 390 products but collapses into just 2 illumination bins — 333 of them at 70-90° incidence, because it is polar. A mid-latitude box populates all 4 bins evenly (216/189/145/153). **Train on mid-latitude tiles; reserve polar sites for evaluation**, where they match OHRC's actual targeting.

Run `python scripts/survey_coverage.py --site <name>` before committing to any region.

### Chandrayaan-2 footprints, resolved 2026-08-23

PRADAN needs a login but the **shapefiles are separate small downloads** (`OHRC_ShapeFiles.zip` 127 KB, `TMC2_ShapeFiles.zip` 4.5 MB) under Other Downloads. Product filenames carry a timestamp but no location, so never download imagery before checking footprints — OHRC products are ~1.2 GB each and 624 of them exist.

`python scripts/ch2_footprints.py --summary` and `--pick {ohrc,tmc}` do this.

**OHRC is polar-dominated.** Of 77 products with usable corner coordinates: 42 south-polar, 7 north-polar, 27 equatorial, 1 mid-latitude. It is a targeting instrument, not a survey one, so it cannot be a training source. Use it for evaluation and the demo. 28 products sit within +/-30 deg, and the six nearest the equator all have LROC NAC coverage across all four illumination bins.

**TMC-2 is the working source.** 8436 products with usable corners, well spread: 4036 equatorial, 2725 mid-N, 1450 mid-S.

**TMC-2 ships ortho + DTM pairs — 3049 of each, and every ortho has a matching DTM**, resolvable by replacing `_oth_` with `_dtm_` in the product id (verified 3049/3049). This is the Kaguya triple trick again, on genuine Chandrayaan-2 data.

Note TMC-2 gives one illumination per site, not a morning/evening pair. So the production task is **TMC-2 ortho (or OHRC) matched against LROC NAC** — cross-sensor, cross-illumination and cross-scale at once, which is precisely what the PS asks for. Kaguya remains the controlled training set because it alone provides identity-correspondence pairs.

### Chandrayaan-2 products in hand, 2026-08-23

Three archives under `data/raw/ch2/`, all verified with `testzip()`. **Read them in place via GDAL `/vsizip/` — do not extract**, that would put ~10 GB of imagery on disk for no gain. Windowed reads work; OHRC takes ~2 s per window. `scripts/ch2_io.py` wraps both product types.

PRADAN's `FileSizeInBytes` column is the **uncompressed** product size, not the archive size. The OHRC zip is 555 MB containing a 938 MB `.img` — a partial download would look plausible against the catalogue number, so verify with `zipfile.testzip()`, never by size.

**OHRC** `ch2_ohr_ncp_20210402T0546284043_d_img_d18` — 12000 x 78175, 8-bit, 0.26 m/px, a 3.1 x 20.3 km strip at ~0.6N 23.4E. **Not map projected**: `crs` is None, no geotransform. Geolocation comes from the sidecar `geometry/**.csv`, which samples Longitude/Latitude against Pixel/Scan every 100 px (94,743 samples). That locates the strip; it does not georeference a pixel.

**TMC-2** `ch2_tmc_ndn_20201126T1610528086` ortho + DTM — GeoTIFF on a SelenoGraphic sphere (radius 1737400). Ortho 10380 x 341544 at ~5.05 m/px; DTM 5190 x 170772 at ~10.1 m/px over **identical bounds**, so ortho pixel (i,j) is DTM pixel (i/2, j/2). Verified: corr(ortho, DEM) = +0.364 on a 512 window at the equator, 224 m relief, 100% valid.

**Neither product carries sun angles.** OHRC's PDS4 label has `pixel_resolution` and `start_date_time` but no incidence/azimuth; TMC-2's shapefile records list `INC_ANGLE`, `EMI_ANGLE`, `PHA_ANGLE` as 0.0. So illumination must be recovered by correlation sweep exactly as for Kaguya.

**Caution on TMC-2 footprint selection.** These derived products are long strips — this one spans 28.4N to 28.5S, ~1730 km. Ranking candidates by centre latitude, as `ch2_footprints.py --pick tmc` currently does, measures strip length rather than suitability, and inflates the NAC counts because the bounding box is enormous. OHRC footprints are small (3 km swath) so centre latitude is meaningful there. **Fix the TMC picker to rank on overlap area and per-area NAC density before downloading more.**

## Training curriculum

Easy to hard. Hold out the last tier.

1. NAC ↔ NAC, same illumination — sanity check
2. Kaguya TC morning ↔ evening — pure illumination change, scale fixed
3. NAC ↔ NAC, large Δsun-azimuth
4. TMC-2 ↔ NAC — cross-sensor, ~10× scale
5. **OHRC ↔ NAC — cross-sensor, extreme scale. EVAL ONLY.**

OHRC coverage is sparse and target-of-opportunity, so there is not enough of it to train on regardless. Verify actual overlapping OHRC/NAC footprints on chmapbrowse early — if thinner than hoped, TMC-2 becomes the primary source and OHRC is a demo showcase.

## Evaluation

Report stratified by **|Δsun azimuth|** and **scale ratio**. The stratified table is the deliverable that shows where classical methods collapse and ours does not.

- Reprojection RMSE, in source pixels, against geometric GT
- Inlier ratio at 1px and 3px thresholds
- Inlier match count
- Success rate — fraction of pairs under 2px
- **Match uniformity** — grid coverage fraction + normalized entropy of per-cell match counts

Uniformity is an explicit PS requirement ("maintaining uniform distribution across the images") that most teams will skip. It stays a first-class metric here.

Baselines to beat: SIFT, ASIFT, phase correlation, off-the-shelf SuperPoint+LightGlue.

## Stack

- Python 3.11
- Geospatial/planetary: GDAL, rasterio, ISIS3 or ALE for PDS4 + map projection, SpiceyPy for geometry
- CV/ML: PyTorch, Kornia, OpenCV
- Training: Kaggle notebooks (free GPU)

**Budget a full week for PDS4 ingestion and map projection.** ISIS3 and GDAL will eat it. Discovering this in week four sinks the project.

## Conventions

- Absolute paths in scripts; this repo lives at `D:\SIH`.
- Every non-trivial module leaves one runnable check — an `assert`-based `demo()` / `__main__` self-check, or a small `test_*.py`. No frameworks, no fixtures.
- Mark deliberate simplifications with a `# ponytail:` comment naming the ceiling and the upgrade path.
- Keep the Stage B ablation runnable at every commit.
- Never report a metric without saying which data split produced it.

## Bug log rule

**Every time a bug is found, append an entry to `BUGS.md` — including how it was resolved.**

This is mandatory and applies to all bugs: crashes, silent wrong output, bad data assumptions, environment and dependency breakage, and metric bugs. No entry may be left without a resolution once the bug is fixed; if it is still open, say so explicitly and update it when fixed.

Use the entry format defined at the top of `BUGS.md`. Newest entries go at the top of the log.
