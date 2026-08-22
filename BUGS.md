# BUGS.md

Running bug log for the SIH26166 lunar image correspondence project.

## Rule

**Every bug found gets an entry here, including how it was resolved.**

Applies to all of them — crashes, silent wrong output, bad data assumptions, environment and dependency breakage, metric bugs. A bug that is fixed must have a filled-in **Fix**. A bug still open says `OPEN` in the status and gets updated when it is fixed.

Newest entries at the top of the log.

## Entry format

```markdown
### BUG-NNN — <one-line symptom>

- **Date:** YYYY-MM-DD
- **Status:** OPEN | FIXED
- **Area:** data-ingest | projection | stage-a-localize | stage-b-photometric | stage-c-matching | stage-d-fit | eval | env
- **Symptom:** what was observed, including the exact error line if there was one
- **Root cause:** what was actually wrong, not what it looked like
- **Fix:** what changed, and where (`file.py:line`)
- **Check:** the runnable thing that now fails if this regresses
```

Rules for writing entries:

- **Symptom and root cause are separate fields and usually differ.** If they read the same, the root cause has not been found yet.
- Quote the shortest decisive line of an error, not the whole traceback.
- A fix without a **Check** is incomplete for anything non-trivial. The check is the smallest thing that fails if the logic breaks again.
- Record bugs caused by wrong assumptions about the data (wrong units, wrong projection, flipped axis, mislabelled illumination angle) with the same weight as code bugs. On this project those are the expensive ones.

---

## Log

### BUG-011 — Shared zero-mask faked the headline result; identity ground truth was also wrong

**This invalidates the previously reported 100% inlier rates. Read this before trusting any earlier number.**

- **Date:** 2026-08-23
- **Status:** FIXED (diagnosis); re-measurement outstanding
- **Area:** stage-c-matching, eval
- **Symptom:** Registering real OHRC against Kaguya produced 44,212/44,223 inliers, RMSE 0.454 px, fitted scale 1.0000, rotation 0.002 deg. Too clean. Controls destroyed it:
  - shifting the raw OHRC by 5 px and by 15 px both still recovered dx = -0.14
  - matching OHRC against a **different part of the Moon**: 7399 matches at ~0 displacement
  - replacing OHRC with **pure noise**: 7439 matches at ~0 displacement
  - Re-running the same controls on the earlier Kaguya morning-vs-evening result reproduced the failure: rolling morning by 15 px still gave dx = +0.11, **noise** gave 1721 "correct" matches, and a **constant grey image** gave 1266.
- **Root cause — two independent errors that happened to agree:**
  1. **Shared zero-mask.** `stage_b_pair()` masks *both* images with the *same* validity mask and writes 0 into the masked pixels. At 5 deg sun elevation that blacks out 44-83% of the frame in a byte-identical pattern in both images. LoFTR matches that pattern: it is high-contrast, perfectly aligned, illumination-independent, and does not move when the underlying imagery is shifted. The matcher never needed the imagery.
  2. **Identity ground truth was wrong.** Kaguya morning and evening share a map grid, so I assumed pixel (i,j) corresponds to pixel (i,j). Phase correlation — independent of any matcher — measures a systematic **dy = +8.25 px (~62 m)** offset, consistent across 5 of 6 windows (dy spread 3.42 px). The two mosaics are independently orthorectified and carry a residual co-registration error.
  - The two errors pointed the same way. The mask artifact produced identity-aligned matches, and identity was exactly what the (wrong) ground truth rewarded. A real match at the true +8 px offset was scored *incorrect*; a fake match at 0 px was scored *correct*. That is why the baseline read 0% and Stage B read 100%.
- **Fix:** Never write the same mask into both images. Per-image masks, or no masking with the render clamped at a floor. Ground truth must be identity **plus the measured inter-product offset**, or established per window by phase correlation.
- **Check:** Any matching result must pass all four: real pair recovers ~0, a raw image rolled by N recovers -N, pure noise collapses, constant grey collapses. `RAW (no Stage B)` passes all four (dx -0.33; roll +15 gives -16.48; noise 3 matches; constant 0).
- **Why my earlier control missed it:** in `dense_match.py` I tested a 7 px shift and recovered +7.29, and took that as proof. But I shifted the **already-normalised** image, which moves imagery and mask together. Shifting the **raw** image before normalisation separates them, and only then does the matcher's reliance on the mask show. **A control that perturbs the output cannot detect a confound in how the output was built — perturb the input.**

### BUG-010 — Blind scale estimation is confounded by normalising both sides against the same DEM

- **Date:** 2026-08-23
- **Status:** FIXED (by removing the need for it)
- **Area:** stage-c-matching
- **Symptom:** `scale_pipeline.py --estimate` recovered the correct ratio only for k=2, returning k=3 when the truth was 4 and again k=3 when it was 8. Correct-match counts were flat across every candidate (true k=4 gave 536/763/777/756 for k=1/2/3/4), and LoFTR mean confidence never left 0.90-0.97 regardless of k.
- **Root cause:** Not a tuning problem — the experiment leaks. `match_common_gsd()` normalises **both** sides against the *same* decimated DEM. Both normalised images therefore inherit identical DEM-derived structure, and the matcher locks onto that shared signal whether or not the imagery aligns. Match count and confidence cannot discriminate k because they are partly measuring the DEM against itself. For the real task sharing the reference DEM is correct; it just destroys this metric's validity.
- **Fix:** Stopped estimating. Every product in the project declares its own ground sample distance — OHRC `pixel_resolution = 0.26 m` in the PDS4 label, Kaguya `MAP_SCALE = 7.403`, TMC-2 and NAC in their geotransform and ODE metadata. `gsd_metres()` and `scale_ratio()` in `scripts/scale_pipeline.py` read k instead of guessing, converting degrees to metres for geographic products.
- **Check:** `gsd_metres()` on the TMC-2 ortho and DTM returns 5.052 and 10.104 m, giving k = **2.00** — a ratio independently known to be exactly 2 from the products' identical bounds and 2:1 pixel dimensions.
- **Why this one mattered:** the estimator returned confident, near-miss answers (3 instead of 4) rather than obvious garbage, which reads like a method that needs tuning rather than one that is measuring the wrong thing. The lesson is about experiment design, not scale: **if both sides of a comparison are derived from a shared input, any agreement metric is partly measuring that input.**

### BUG-009 — TMC-2 candidate ranking measured strip length, not suitability

- **Date:** 2026-08-23
- **Status:** FIXED
- **Area:** data-ingest
- **Symptom:** `ch2_footprints.py --pick tmc` reported `ch2_tmc_ndn_20201126T1610528086_d_oth_d32` at latitude -0.05 with 5597 NAC products and all four illumination bins — an apparently ideal equatorial candidate. Reading its shapefile record showed `UL_LAT 28.40`, `BL_LAT -28.50`: a strip spanning 57 degrees of latitude, roughly 1730 km long.
- **Root cause:** Two compounding errors on the same assumption that a footprint is small. Candidates were ranked by centre latitude, `(min+max)/2`, which for a pole-to-pole strip is ~0 regardless of where the strip actually is. And the NAC count was queried over the strip's full bounding box, so a longer strip always scores higher — the number measured box area, not coverage density.
- **Fix:** `probe_box()` in `scripts/ch2_footprints.py` clips each strip to a fixed 1x1 degree window where it crosses a chosen latitude (`--at-lat`), and scoring reports NAC **per square degree** alongside the full strip area. Strips are ranked narrowest-first, since a narrow strip wastes less download per useful square degree. OHRC keeps whole-footprint scoring — its 3 km swath genuinely is the target.
- **Check:** `python scripts/ch2_footprints.py --pick tmc --at-lat 0` — probe densities now land in a comparable 150-200 NAC/deg2 band across candidates instead of scaling with strip length, and reported strip areas (4.7-8.9 deg2) are visibly smaller than the ~98 deg2 product the broken version recommended.
- **Why this one mattered:** it produced a confident, plausible recommendation that would have cost a multi-GB download of a strip whose useful overlap was a tiny fraction of its extent. **A ranking metric has to be invariant to the size of the thing being ranked**; raw counts over variable-sized regions never are.

### BUG-008 — Independent validity masks gave the two images different intensity stretches

- **Date:** 2026-08-23
- **Status:** FIXED
- **Area:** stage-c-matching
- **Symptom:** `scripts/scale_test.py` reported near-zero correct matches at **every** ratio including **1x**, where it should have reproduced the known-good ablation result. Direct comparison on identical input: `ablation.stage_b_matches` 47/67 correct, `scale_test.stage_b_scaled` 0/17.
- **Root cause:** `ablation` intersects the two shadow-validity masks and stretches both images over the *same* pixel population. `scale_test` used each image's own mask, because at k>1 the two arrays have different shapes and could not be intersected directly. `to_uint8` sets its 1st/99th percentile from the valid pixels, so two masks covering different terrain produce two different intensity mappings — after which the descriptors are computed on incomparable scales and cannot match.
- **Fix:** Added `intersect_masks()` in `scripts/scale_test.py`, which brings the masks onto a common footprint across resolutions — replicating the coarse mask up to the fine grid, and requiring a coarse pixel to be majority-valid when going down — before either image is stretched.
- **Check:** `python scripts/scale_test.py` at ratio 1x must roughly match `ablation.py` on the same windows. It now returns 975/1074 correct (91%) at coverage 0.90.
- **Why this one mattered:** the failure looked exactly like a scientific result. "Scale invariance fails" is a plausible, expected-sounding conclusion, and the 1x row was the only thing that exposed it as a harness bug. **Always include a control condition whose answer you already know.**

### BUG-007 — `STANDARD_GEOMETRY` read as acquisition geometry, flattening every render

- **Date:** 2026-08-23
- **Status:** FIXED
- **Area:** stage-b-photometric
- **Symptom:** The first Stage-B ablation did nothing. Correlation moved -0.560 to -0.509 (swing +0.051) and SIFT found 6 raw / 8 normalised matches, none correct. No crash, no warning.
- **Root cause:** `STANDARD_GEOMETRY = (30.0, 0.0, 30.0)` in the Kaguya label describes the photometric **correction that was applied**, not the geometry the image was **acquired** at. I took 30 deg incidence as the sun position and rendered at 60 deg elevation. The morning/evening mosaics are shot near the terminator, and the USGS correction rescales brightness without removing shadows already baked in at low sun. At 60 deg the render was nearly flat — contrast cv **0.045** against the real image's 0.278, and **zero** shadow pixels — so dividing by it barely altered the image.
- **Fix:** `scripts/ablation.py` now treats elevation as a free parameter (`ELEVATION_CANDIDATES`) and recovers it alongside azimuth via `best_geometry()`. Recovered geometry is near-terminator (morning az=90 el=5, evening az=240 el=10), where render contrast is cv 1.019 — a 22x increase.
- **Check:** `python scripts/ablation.py` — raw vs Stage-B SIFT matching scored against the identity ground truth. With the bug present, Stage B gains nothing; with it fixed, 0/44 correct becomes 171/209 across five windows.
- **Related:** the two-stage search in `best_geometry()` exists because ray-marched shadows at every one of 252 candidates did not finish in 120 s. The coarse pass ranks without shadows and only the top 6 pay for them.

### BUG-006 — Projected northing in metres was passed to a cosine as if it were degrees

- **Date:** 2026-08-22
- **Status:** FIXED
- **Area:** projection
- **Symptom:** `scripts/estimate_sun.py` printed `latitude 500338.984 deg` for a window whose true latitude is 16.5 deg. It did not crash, and the experiment still produced a correct-looking result.
- **Root cause:** `Triple.transform` maps pixel to **projected metres**, not degrees — the CRS is Equirectangular on a 1737400 m sphere. The code unpacked `transform * (col, row)` straight into a latitude variable and handed it to `pixel_spacing()`, which takes `cos(latitude)`. `cos(500338.984 deg)` reduces to `cos(298.98 deg) = 0.485` instead of `cos(16.5 deg) = 0.959`, so east-west ground spacing was 3.59 m rather than 7.10 m and every surface normal was computed with east-west slopes roughly twice too steep.
- **Fix:** Added `Triple.pixel_latlon()` in `scripts/triple_io.py`, converting projected metres to degrees by arc length (`deg = m / R`) with the sphere radius read from the CRS rather than hardcoded, plus a range assertion on the result. `estimate_sun.py` now calls it.
- **Check:** `python -c "...pixel_latlon(0,0)"` returns 18.0001N / 8.9999E against label bounds of 15.000244-18.0 N and 9.0-11.999756 E. The in-method assertion fires if the transform units ever change.
- **Why this one mattered:** **the experiment passed with the bug in place.** Azimuth recovery still found east for morning and west for evening, because a wrong east-west scale distorts slope magnitudes without flipping the east/west asymmetry the test keys on. A passing headline result is not evidence that intermediate quantities are right. After the fix both correlations improved (morning +0.518 to +0.533, evening +0.513 to +0.534) and the morning estimate moved to exactly 90 deg — that improvement is the only reason we can tell the fix helped.

### BUG-005 — Loader self-check assumed morning/evening images correlate positively

- **Date:** 2026-08-22
- **Status:** FIXED
- **Area:** data-ingest
- **Symptom:** `python scripts/triple_io.py` failed on the first real tile with `AssertionError: correlation -0.560 too low - are these really the same tile?`
- **Root cause:** The assertion was wrong; the data is fine. Morning light arrives from the east and evening light from the west, so slopes lit at dawn are shadowed at dusk. Raw intensities of identical terrain are therefore **anti**-correlated. Requiring `corr > 0.1` encoded the assumption that same terrain implies similar pixels — which is precisely the assumption this project exists to disprove.
- **Fix:** `scripts/triple_io.py` now asserts `abs(corr) > 0.2` (the windows must be *related*, not *similar*) and prints the sign with its interpretation. Measured: **-0.560** on tile `N18E009N15E012SC`, window (5888,5888) 512x512.
- **Check:** `python scripts/triple_io.py` — opens a triple, reads an aligned window from all three, asserts shapes match, the pair differs, `|corr| > 0.2`, and DEM relief is non-flat.
- **Keep this number.** -0.560 is a direct measurement of the crater/dome inversion on real lunar data, and it is the quantitative justification for Stage B in the report. After photometric normalisation this correlation should swing positive — that swing is the headline ablation result.

### BUG-004 — ODE's reported file size under-reports by ~2%, so resume logic could accept a truncated file

- **Date:** 2026-08-22
- **Status:** FIXED
- **Area:** data-ingest
- **Symptom:** Found by inspection, not by a failure. ODE reports `288001 KB` for the Kaguya `.IMG` products, but the true size is `301,989,888` bytes = `294,912 KB`, confirmed against the label's own `LINES x LINE_SAMPLES x SAMPLE_BITS/8` (12288 x 12288 x 2).
- **Root cause:** `download()` decided a file was complete via `have >= expect_kb * 1024 * 0.999`, using ODE's estimate as the target. Because that estimate is ~2% low, any download interrupted between 294.6 MB and 302.0 MB would be judged complete on the next run and never resumed.
- **Fix:** Added `true_size()` in `scripts/kaguya.py`, which takes the size from a `HEAD` request's `Content-Length` and compares **exactly**. Oversized files are refetched, and `download()` now raises `IOError` if the final byte count disagrees with the server. ODE's number is only used as a floor when the server declines to give a length.
- **Check:** Re-running `python scripts/kaguya.py fetch --site equatorial` prints `verified` for each file and re-downloads nothing.
- **Why this one mattered:** a truncated `.IMG` does not fail to open. GDAL reads the intact leading rows and returns zeros or garbage past the truncation, so training would have silently consumed blank imagery from the southern part of every affected tile.

### BUG-003 — Photometric self-check asserted the wrong direction; the renderer was correct

- **Date:** 2026-08-22
- **Status:** FIXED
- **Area:** stage-b-photometric
- **Symptom:** `python scripts/photometric.py` failed check 2 with `AssertionError: azimuth sign convention is inverted`. A ramp built by `_ramp(dz_dcol=+k)` scored 0.4226 under an eastern sun and 0.9063 under a western one — the opposite of what the test expected.
- **Root cause:** The test was wrong, not the code. The helper built a plane **rising** toward the east and the assertion treated it as **facing** east. Those are opposites: for `z = a·x_east` the outward normal is `(-a, 0, 1)`, which tilts *west*, so a western sun should light it better. Hand-checking the numbers confirmed the renderer: normal `(-0.342, 0, 0.940)` dotted with the western sun vector `(-0.707, 0, 0.707)` gives exactly the 0.9063 observed. Check 3 carried the identical mistake on the north/south axis.
- **Fix:** Replaced `_ramp` with `_slope_facing_east` and `_slope_facing_north` in `scripts/photometric.py`, named for the direction the surface **faces**, with the normal-vector reasoning in the docstring. Expectations corrected accordingly.
- **Check:** `python scripts/photometric.py` — 7 analytic checks covering flat-surface shading, both azimuth axes, shadow side, shadow length vs sun elevation, crater/dome ambiguity, and the latitude spacing correction.
- **Why this one mattered:** the test failure pointed at correct code. Trusting it would have meant inverting a working sign convention to satisfy a broken assertion, and every rendered reference image afterwards would have been lit from the wrong side — with all self-checks passing. **A failing test is a hypothesis, not a verdict. Verify which side is wrong by hand before editing either.**

### BUG-002 — ODE REST silently ignores unknown query parameters, returning unfiltered results

- **Date:** 2026-08-22
- **Status:** FIXED
- **Area:** data-ingest
- **Symptom:** Querying LROC NAC at the Chandrayaan-3 site with `minincidenceangle=30&maxincidenceangle=60` and with `minincidenceangle=70&maxincidenceangle=89` both returned `Count = 390` — identical. So did `iangle1`/`iangle2`. 390 turned out to be the *unfiltered* count for the bounding box.
- **Root cause:** The parameter names were wrong, but ODE does not reject unrecognized query parameters — it drops them and answers the remaining query. A wrong filter name is indistinguishable from no filter. The real names are **`mininangle`** and **`maxinangle`** (likewise `minemangle`/`maxemangle`, `minphangle`/`maxphangle`), documented only in the ODE REST V2.1.6 User's Manual PDF, not on the landing page.
- **Fix:** Use `mininangle`/`maxinangle`. Manual cached at `docs/ode_rest_params.txt` for the full parameter list. Verified working: same bbox returns 0 / 30 / 333 for incidence 30-50 / 50-70 / 70-90, against 390 unfiltered.
- **Check:** `python scripts/survey_coverage.py --self-check` asserts that an incidence-filtered count is strictly less than the unfiltered count for a known bbox. Fails if a parameter name silently stops biting.
- **Why this one mattered:** it would not have crashed. It would have quietly built the entire Stage-B training set from images at whatever illumination happened to be there, while the logs claimed a controlled illumination split — and the Stage-B ablation, our central scientific claim, would have been measuring nothing. **Rule going forward: never trust a filter that was not observed to change the result count.**

### BUG-001 — Non-ASCII characters print as `?` in Windows console output

- **Date:** 2026-08-22
- **Status:** FIXED
- **Area:** env
- **Symptom:** `python scripts/check_env.py` printed `GDAL 3.10.3 ? 204 drivers`. The em-dash in the f-string came out as a replacement character.
- **Root cause:** Windows console defaults to the legacy `cp1252` code page, not UTF-8. Python encodes stdout with that codec, and any character outside cp1252 is mangled. Not a rasterio or GDAL problem at all — the driver probe itself was correct.
- **Fix:** Replaced the em-dash with an ASCII hyphen in `scripts/check_env.py:69`. Keeping console output ASCII-only is cheaper than forcing a UTF-8 code page on every machine that runs this.
- **Check:** `python scripts/check_env.py` output contains no `?` replacement characters.
- **Note:** This will recur with real force during data ingest. **PDS4 labels are UTF-8 XML** and may carry non-ASCII in target names, mission descriptions, and units. Read and write those files with an explicit `encoding="utf-8"` — never rely on the platform default — or the same mangling silently corrupts metadata rather than just cosmetics.
