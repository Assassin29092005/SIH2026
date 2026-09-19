# Next steps for SANDHI

## Assessment

SANDHI has a credible technical core. Its biggest strength is not merely that it produces matches: it tests whether those matches track lunar terrain. The four-control gate, cross-window validation, documented retractions, and explicit negative IIRS result make the evidence much more trustworthy than a matcher evaluated only by RMSE.

The current evidence supports three clear claims:

- SANDHI handles severe illumination reversal better than the tested baselines. On Kaguya morning versus evening imagery, it is the only listed method that passes the control gate.
- TMC-2 to Kaguya is the strongest problem-statement result: 0.872 source pixels, 0.891 coverage, and 0.894 entropy.
- The project is honest about its boundaries. OHRC is reference-limited with Kaguya, IIRS does not currently register, and the demonstrated 12.3 degree viewpoint result is not a universal limit.

The main weakness is delivery readiness. The science currently lives across reproducible scripts and a package, but an ISRO user still cannot give SANDHI an arbitrary product pair and receive a robust, georeferenced result with clear diagnostics.

## Priority 1: Close the OHRC reference gap

The problem statement asks for sub-pixel accuracy in the source image. Kaguya TC cannot support that measurement for 0.26 m/px OHRC because its 7.403 m/px pixels are too coarse. This is a data-pairing limitation, not a claim that needs better wording.

### Work

1. Inspect the six downloaded LROC NAC products for overlap with the existing OHRC strip.
2. Build NAC ingestion and map projection using the tested OHRC geometry approach.
3. Register OHRC to a NAC reference near 0.5 m/px.
4. Report both `rmse_px` and `rmse_source_px`, along with coverage, entropy, model selection and the four controls.

### Acceptance criteria

- The pair passes real-pair, raw-input-roll, pure-noise and flat-grey controls.
- `rmse_source_px < 1.0` for OHRC is reported only if the measurement passes every control.
- The registered image, match CSV and metrics JSON are reproducible from one documented command.

## Priority 2: Make the control gate non-optional

The gate has already prevented false headline results. It should therefore be enforced by automation rather than team convention.

### Work

1. Add a CI workflow that runs the fast unit suite on every pull request.
2. Run the two sample-based control-gate tests in CI or on a scheduled workflow.
3. Fail the workflow when a reported experiment omits a control result or when a gate check fails.
4. Save metrics JSON and control reports as CI artifacts for traceability.

### Acceptance criteria

- A deliberately input-ignoring matcher fails CI.
- A pull request cannot publish a result without a complete gate report.
- The tests run from a fresh clone using committed samples.

### Audited 2026-09-20 - doable, with two corrections

- **Criterion 1 is nearly free.** `test_gate_rejects_a_matcher_that_ignores_its_input` never reaches LoFTR: 0.24 s of test body, ~9 s per pytest run because importing `sandhi` pulls torch. It is buried only by the blanket `pytestmark = pytest.mark.slow` at `tests/test_pipeline.py:30`, so reach it by node id rather than by marker.
- **Item 3's second clause is already satisfiable**; only the first needs a definition. `sandhi/outputs.py` `write_metrics(result, path, extra)` already merges an arbitrary `extra` dict into the deliverable JSON, and `write_all(..., extra=)` threads it through, so attaching a gate block is a call-site change, not a schema invention. What is missing is a definition of "a reported experiment": of 13 JSON files in `outputs/`, **4 carry a gate, in 5 incompatible shapes**, and the headline deliverable `tmc2_metrics.json` is not one of them while `tmc_vs_kaguya.json` from the same run is.
- **Two CI setup facts to budget for:** `opencv-python` is the GUI build (`grep -c opencv-python-headless uv.lock` returns 0) and `cv2` is imported at module level, so a bare Ubuntu runner needs `libgl1`; and the LoFTR checkpoint is fetched over plain HTTP from a single academic host, so the slow job needs an `actions/cache`.
- The gate itself is expensive - three cases measured at ~766 s locally - so it belongs on a schedule, not on every pull request.

## Priority 3: Deliver a usable registration command

The package should become the primary path for a user-facing run, while `scripts/` remains the frozen measurement record.

### Work

1. Stabilise `sandhi register` around explicit source, reference, GSD, output and validation options.
2. Move tuned constants into a documented configuration file with safe defaults.
3. Emit a GeoTIFF whenever the reference has a CRS, plus the match-point CSV, metrics JSON and control report.
4. Add clear messages for unsupported products, missing metadata, insufficient overlap and gate failures.
5. Add resumable output directories and structured logs for longer jobs.

### Acceptance criteria

- A new user can run a documented example without editing Python source files.
- Outputs carry the reference CRS and transform where applicable.
- A failed validation produces an actionable reason rather than an unqualified result.

### Audited 2026-09-20 - item 2 was already done, and the audit found a live bug

- **Item 2 is complete.** `sandhi/config.py` is a frozen dataclass of 14 constants, each carrying the measurement that chose it, overridable at runtime via `config.settings()` and the CLI flags.
- **BUG-030, found and fixed.** `sandhi register` invented a GSD of 1.0 m/px from rasterio's identity geotransform, and that value silently overrode the caller's `--gsd`, reporting the PS's source-pixel clause **7.4x optimistic**. Also fixed: `transform.a` is degrees on this project's SelenoGraphic products, not metres. Both are now guarded by tests.
- **Item 3's control report is still a genuine gap.** `cmd_register` never calls `controls`; the gate is reachable only through the separate `controls` subcommand. Wiring it in costs ~4x a registration, because the gate runs the matcher four times.
- **Item 4 does not exist yet.** The whole error vocabulary is two strings, `cannot read <path>` and `no model could be fitted`. There is no floor on match count, coverage or inlier ratio: a 7-match run prints `[SUB-PIXEL]` and exits 0.
- **Acceptance criterion 2 cannot be shown from a fresh clone.** All six committed samples have `crs=None`, so the GeoTIFF branch in `outputs.py` never executes on sample data.

## Priority 4: Demonstrate whole-product processing

Current results validate windows and chunks. A deployable workflow needs to process a full strip without silently stitching incompatible local models.

### Work

1. Tile full products with overlap and retain per-tile correspondence metadata.
2. Check neighbouring tile models for translation, scale and rotation consistency.
3. Merge accepted tiles into one registered product and flag rejected regions.
4. Report per-tile and whole-product coverage, entropy and control outcomes.

### Acceptance criteria

- A complete TMC-2 strip produces one georeferenced output and a traceable tile manifest.
- The pipeline flags inconsistent neighbouring tile transforms instead of blending them without warning.

### Audited 2026-09-20 - the code is about a day, the criterion is data-blocked

- **A "complete TMC-2 strip" is not reachable today.** The ortho spans 28.40N to 28.50S over lon 70.18-71.91E. Of the three Kaguya tiles on disk, only `N03E069N00E072SC` (0-3N, 69-72E) overlaps it - **3 degrees of 56.9, or 5.27% of the strip**. The rest needs ~19 more tiles, about 16 GB. The honest deliverable is a bounded section inside the one covering tile.
- **Per-tile metadata is destroyed by design.** `sandhi/matching.py` vstacks every tile's points into one array, so which tile a match came from is lost at concatenation, and `pipeline` then fits one global model. There is no per-tile model to compare against a neighbour, and no manifest, mosaic or neighbour-consistency code anywhere.
- **Do not copy the sketch's aggregation.** `scripts/ohrc_vs_kaguya.py` averages per-chunk RMSE across chunks with no consistency test - the exact unflagged blending this priority's second criterion forbids.

## Priority 5: Keep IIRS as a clearly labelled research track

Do not present IIRS support as completed. The controls show that the current IIRS-to-visible-camera pairing does not register, and the remaining cause is not yet settled.

### Work

1. Reconstruct an independent IIRS geolocation using SPICE or an equivalent geometry workflow.
2. ~~Test spectrometer-to-spectrometer registration, beginning with IIRS to Chandrayaan-1 M3~~ - **already spent, 2026-09-10.** `outputs/iirs_vs_m3.json` records both flip orientations as `registered: false`. Do not schedule it again.
3. Keep every failed pairing, control result and hypothesis test in the evidence record - **not currently satisfied**, see below.

### Acceptance criteria

- Any IIRS success includes the full control-gate report and an independent geometry check.
- Until then, the product and presentation state that IIRS remains an open limitation. **Met** - README, the generated report and the deck all state it.

### Audited 2026-09-20 - item 3 is the real work, not item 1

Three IIRS claims in the deliverables have no code path behind them, which is exactly what item 3 forbids:

- The headline number is a **string literal**. `outputs/iirs_vs_kaguya.json` holds only `{wavelength_nm, band, registered: false}`; the "+0.0326, 44 matches, 6/44" that README, the report and the deck all quote is hardcoded in a print statement. Re-running `--controls` prints it whatever the data says.
- The same polynomial fit is reported as **3.96 px** in README and **3.48 px** in `scripts/make_docs.py`.
- Two whole hypothesis tests - a 35 km blind offset search peaking at 6.0 sigma, and a fixed-pattern striping test at 1.27% against 2.50% - exist **only as report prose** in `make_docs.py`. No script, no JSON, nowhere else in the repo.

Also logged as **BUG-028**: `ground_gsd()` takes a median over a backplane in which 16.06% of adjacent sample pairs are exactly 0 m apart, so it reads **+18.65% high**. The label's 85.08 m is correct to 0.71%; the measurement is what is wrong. Do not "fix" the label.

**A proposed independent check that does not work:** correlating the Kaguya DTM against IIRS's own LOC Height band gives r = +0.991, and it is a null test - LOC Height is topography read out *at the coordinates LOC itself declares*, so it agrees wherever those coordinates point. Same structural blindness as the band-vs-band control. An independent geolocation has to come from outside the product.

## Priority 6: Improve the SIH demonstration

For judging, the clearest story is a short, reproducible evidence chain rather than a long algorithm inventory.

1. Start with the illumination-inversion problem: the same terrain has raw correlation -0.560
   (tile N18E009N15E012SC, window (5888,5888), 512x512 — name the window on the slide,
   because `python scripts/triple_io.py` runs a different tile and prints -0.627).
2. Show the four-control gate and the previously rejected mask artifact.
3. Lead the final result with TMC-2 to Kaguya because it meets the source-pixel clause.
4. Show one registered-image overlay, match distribution and metrics JSON.
5. State OHRC and IIRS limits plainly, followed by the NAC plan for OHRC.

### Audited 2026-09-20 - item 4 is already built; item 3 is contradicted by the demo

- **Item 4 is complete.** `scripts/demo.py` already renders the registered checkerboard overlay, an 8x8 match-distribution heatmap and a metrics JSON per case. Coverage and entropy are visualised, not merely printed.
- **Item 3 is contradicted by the repo's own demo.** See **BUG-027**: `outputs/demo_tmc.json` reports `rmse_source_px` 1.031 with `sub_pixel_source: false`, while `outputs/tmc2_metrics.json` reports 0.872 and true - identical 3085 matches, different fit, because `demo.py` runs `scripts/register.py` (no model selection) and the reported result came through the package. Leading a slide with "TMC-2 because it meets the source-pixel clause" while the one-command demo prints the opposite is the gap to close first.
- **Item 2's mask-artifact figure does not exist and cannot be made from a fresh clone.** Nothing in the repo renders it, and regenerating it needs a DEM, which `samples/` does not carry.
- **Item 1's number needs its window on the slide**, as noted above.

## Avoid revisiting without new evidence

- DEM-division photometric normalisation for matching. It failed the noise control.
- Blind scale estimation. Product metadata provides the scale without the confound.
- Bucketed match selection as a coverage solution. It redistributes existing matches but cannot populate empty terrain regions.
- Fine-tuning on warped self-pairs. It improves the wrong training objective and damaged real cross-illumination matching.
