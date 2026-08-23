# ROADMAP — from prototype to running software

Where this project actually stands, and what stands between it and software someone other than us could depend on.

Written 2026-08-23. Nothing here is scheduled; it is a list of what is missing and why it matters.

## What exists today

A validated method with a demo. All eight problem-statement requirements are met with control-gated evidence, one command produces a figure, and a fresh clone runs from committed samples.

What that is **not**: a program someone else can point at their own imagery and trust. Everything below is the gap.

---

## Phase 1 — Correctness gaps that could change conclusions

These are open scientific questions, not polish. Each one could move a number that is currently reported as a result.

### 1.1 Separate Chandrayaan-2 geolocation error from our own projection error
**Status:** unresolved, and it is the most important item here.

The OHRC↔Kaguya registration recovers a 437 px (~3.4 km) offset, corroborated to 0.7 px by phase correlation. But **two different causes produce exactly this signature**: genuine Chandrayaan-2 geolocation error, or residual error in the polynomial projection we built from OHRC's geolocation CSV. We cannot currently tell them apart.

Resolving it needs an independent geometric anchor — registering the same OHRC strip against a second, independently-controlled reference (LROC NAC map-projected products, or LOLA-controlled basemaps). If both references agree on the offset, it is Chandrayaan-2's. If they disagree, it is ours.

Until this is settled, the 3.4 km figure must stay described as "the measured offset between our projected OHRC and the Kaguya frame", not as Chandrayaan-2's geolocation error.

### 1.2 Isolate obliquity from illumination in the viewpoint envelope
The obliquity ladder shows matching succeeds at a 12.3° emission gap and fails from 14.9°. But the one success is also the only image acquired in the same orbit sequence as the anchor; the failures are years apart with different illumination. **Obliquity and illumination are confounded.**

Needs same-date NAC pairs across several emission angles. Until then, 12.3° is a demonstrated success under favourable conditions, not a measured obliquity limit.

### 1.3 Extend scale beyond 8×
16× fails because a 1024 px window decimates to 64×64, and larger windows exceed CPU memory — LoFTR's coarse attention is O((H·W/64)²), so 2048² needs 17 GB. This is a resource bound, not a method bound. A GPU, or a coarse-to-fine cascade that never holds a large attention matrix, should reach the ~19× and ~28× ratios the real OHRC→TMC-2 and OHRC→Kaguya cases need.

### 1.4 Non-rigid deformation
Only similarity, affine and homography are fitted. Real lunar registration over long strips has terrain-induced distortion no global model captures. Thin-plate splines or a local mesh would fit it — and the existing held-out model-selection framework already provides the honest way to decide whether the extra freedom is earned.

### 1.5 IIRS
The problem statement names OHRC, TMC **and IIRS**. IIRS is hyperspectral at ~80 m/px and has never been touched. Whether the pipeline transfers to it is unknown.

---

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

Each of these cost real time. They are listed so nobody spends that time again.

---

## Suggested order

1. **1.1** — it decides whether the headline number means what we say it means
2. **2.3** — lock the control gate into CI before the codebase grows
3. **3.1** — GPU, which unblocks 1.3 and makes everything else faster to iterate on
4. **2.1 / 2.2** — package it, once the science has stopped moving
5. Everything else by need

Items 1.1 and 2.3 are the two that protect against being wrong. The rest make it pleasant.
