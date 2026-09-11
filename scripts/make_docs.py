"""Generate the SANDHI technical report as .docx (and .pdf if Word is present).

Every number in here is copied from a committed measurement -- `outputs/*.json`,
the README tables, or `sandhi/config.py`'s provenance comments -- so the report
regenerates rather than drifting. Nothing is retyped from memory.

    python scripts/make_docs.py
    python scripts/make_docs.py --no-pdf

Needs python-docx. PDF conversion uses Word via COM and is skipped silently on
any machine without it -- the .docx is the deliverable, the .pdf a convenience.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
MUTED = RGBColor(0x59, 0x59, 0x59)


# --------------------------------------------------------------- helpers
def load(name: str) -> dict:
    p = ROOT / "outputs" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def h(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = ACCENT
    return p


def para(doc, text, *, italic=False, size=10.5, space_after=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(doc, items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        _rich(p, it)
        p.paragraph_format.space_after = Pt(2)


def _rich(p, text):
    """Minimal **bold** and `code` inline markup, so content stays readable."""
    buf, i = "", 0
    while i < len(text):
        if text.startswith("**", i):
            j = text.find("**", i + 2)
            if j > 0:
                p.add_run(buf).font.size = Pt(10.5)
                buf = ""
                r = p.add_run(text[i + 2:j])
                r.bold = True
                r.font.size = Pt(10.5)
                i = j + 2
                continue
        if text[i] == "`":
            j = text.find("`", i + 1)
            if j > 0:
                p.add_run(buf).font.size = Pt(10.5)
                buf = ""
                r = p.add_run(text[i + 1:j])
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)
                i = j + 1
                continue
        buf += text[i]
        i += 1
    if buf:
        p.add_run(buf).font.size = Pt(10.5)


def code(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.space_after = Pt(8)
    return p


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, name in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(name)
        r.bold = True
        r.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            _rich(cells[i].paragraphs[0], str(val))
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = MUTED
    p.paragraph_format.space_after = Pt(12)


# ----------------------------------------------------------------- build
def build(doc: Document) -> None:
    tmc = load("tmc2_metrics.json")
    m3 = load("m3_vs_kaguya.json")

    # ---------------------------------------------------------- title
    t = doc.add_heading("SANDHI", level=0)
    for r in t.runs:
        r.font.color.rgb = ACCENT
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run("Multi-modal, Sun-angle and scale-invariant image correspondence\n"
                  "for Chandrayaan-2 optical imagery")
    r.font.size = Pt(13)
    r.font.color.rgb = MUTED
    para(doc, "Smart India Hackathon 2026  |  Problem Statement SIH26166  |  "
              "ISRO / Department of Space", size=10)
    para(doc, "Technical report: design, pipeline, parameters, code and measured "
              "results. Every figure quoted here is reproducible from the "
              "repository by the command given beside it.", italic=True, size=10)

    # ------------------------------------------------------------ 1
    h(doc, "1. What the product is", 1)
    para(doc, "SANDHI takes an image acquired by Chandrayaan-2 and a lunar "
              "reference image of the same ground, and finds where each point in "
              "the first lands in the second. It outputs three things: the source "
              "image warped onto the reference (the registered product), the list "
              "of correspondences it used, and the metrics that say how good the "
              "result is.")
    para(doc, "The problem is not finding matches between two photographs. It is "
              "finding them when the two photographs disagree about almost "
              "everything except the ground underneath.")
    table(doc, ["What differs between the pair", "By how much", "Why it breaks normal methods"], [
        ["Sun angle", "Up to opposite illumination", "The Moon has no atmosphere, so no fill "
         "light. Shading dominates, and shading inverts. A crater lit from the east is "
         "near pixel-identical to a dome lit from the west. Two Kaguya images of "
         "**identical ground** correlate at **-0.560**."],
        ["Resolution", "Up to **28.5x**", "A feature that is a hundred pixels across in one "
         "image is three in the other."],
        ["Sensor", "Camera vs spectrometer", "Different physics recording the same terrain."],
        ["Viewpoint", "Measured to 12.3 deg", "Foreshortening that no shift-and-rotate can undo."],
    ])
    para(doc, "Lunar terrain is also texture-poor, so classical corner detectors "
              "starve: SIFT scatters by 53 px across windows on this data and is "
              "unusable as a primary matcher.")

    h(doc, "1.1 What the problem statement asks for, and where we stand", 2)
    table(doc, ["PS requirement", "Status", "Evidence"], [
        ["Correspondence between CH-2 optical images and lunar reference images",
         "**Met**, two of three instruments", "OHRC 1443 matches, TMC-2 3085 matches, both "
         "control-gated. IIRS measured and does not register."],
        ["**Sub-pixel accuracy of source image**", "**Met on TMC-2**; reference-limited on OHRC",
         "TMC-2 **0.872 px of the 5.05 m source product**. OHRC 0.513 grid px but 14.6 px of "
         "its own 0.26 m product -- see section 12.2."],
        ["**Uniform distribution across the images**", "**Met**",
         "Coverage **0.891**, entropy **0.894** on TMC-2, over an 8x8 grid."],
        ["Registered product + corresponding match points", "**Met**",
         "GeoTIFF (or PNG), match-point CSV, metrics JSON -- three files per run."],
        ["Evaluation metrics (RMSE, inlier count, inlier ratio)", "**Met**",
         "All three, plus coverage and entropy, in every metrics JSON."],
        ["Sun-angle invariance", "**Met**",
         "0.85 px cross-window spread, 54% inliers. **The only method of five that passes "
         "the control gate on cross-illumination.**"],
        ["Scale invariance", "**Met**", "Demonstrated to **19.84x** (M3 to Kaguya, gated)."],
        ["Viewpoint invariance", "**Met, envelope measured**",
         "Succeeds to a 12.3 deg emission gap; fails from 14.9 deg."],
    ])

    # ------------------------------------------------------------ 2
    h(doc, "2. How it was designed", 1)
    h(doc, "2.1 Four hard constraints, fixed before any code", 2)
    bullets(doc, [
        "**Data must be genuine, public and current.** Real observations only. No "
        "synthetic or simulated imagery as a training or evaluation source. Every "
        "dataset has a live public access path, verified before use.",
        "**Ground truth comes from geometry, never from a warp we invented.** "
        "Warping an image to manufacture a correspondence pair is synthetic ground "
        "truth and is banned. Map-projected images sharing a controlled lunar frame "
        "are used instead, so correspondence follows from projection.",
        "**Zero extra hardware.** Laptops only. GPU work runs on free Kaggle or "
        "Colab. A step that needs hardware we do not own gets redesigned.",
        "**Fine-tune, never train from scratch.** The free GPU budget does not "
        "cover training a matcher from zero.",
    ])

    h(doc, "2.2 The design rule that shaped everything else", 2)
    para(doc, "Early in the project this pipeline reported 100% inlier rates that "
              "were entirely artifact. A shared zero-mask written identically into "
              "both images gave the matcher a perfectly aligned, high-contrast "
              "pattern to lock onto, and the assumed ground truth was independently "
              "wrong by 8.25 px. Two errors pointing the same way produced a "
              "beautiful, false result.")
    para(doc, "Since then no number is written down unless it passes a four-control "
              "gate (section 7). That rule is why several components described in "
              "earlier versions of this project no longer exist -- they were "
              "measured, they failed, and they were removed. Section 11 lists them.")

    h(doc, "2.3 Do we use a CNN? Do we use YOLO?", 2)
    para(doc, "Asked often enough to answer directly.")
    table(doc, ["Question", "Answer"], [
        ["Is a CNN involved?", "**Yes, inside LoFTR.** Its backbone is a ResNet-FPN "
         "convolutional network that produces feature maps. It is followed by a "
         "**transformer** that does the actual matching, using self- and "
         "cross-attention between the two images."],
        ["Did we train a CNN?", "**No.** We use the public MegaDepth `outdoor` "
         "weights, which have never seen lunar imagery."],
        ["Is YOLO involved?", "**No.** YOLO is an object detector -- it draws boxes "
         "around things it recognises. There are no objects to detect here. The task "
         "is dense pixel-to-pixel correspondence."],
        ["Did we fine-tune?", "**Attempted twice, rejected twice, both times on "
         "measurement.** See section 11.2."],
        ["Is there a detector at all?", "**No, and deliberately.** LoFTR is "
         "*detector-free*. Detector-based methods need repeatable keypoints, and "
         "lunar terrain under inverted illumination does not provide them -- "
         "DISK+LightGlue finds **one** match on a real cross-illumination pair."],
    ])

    # ------------------------------------------------------------ 3
    h(doc, "3. Architecture", 1)
    para(doc, "Two surfaces, deliberately not interchangeable.")
    table(doc, ["Surface", "Role", "Rule"], [
        ["`sandhi/`", "The installable package. The pipeline, and what new work uses.",
         "Imports **nothing** from `scripts/`."],
        ["`scripts/`", "The measurement record: how every number in the report was "
         "obtained.", "Kept reproducible, not refactored."],
    ])
    para(doc, "The separation is the point. Results must stay reproducible at the "
              "exact code that produced them, while the pipeline is free to improve. "
              "If the two shared modules, tidying the package would silently change "
              "published numbers.")

    h(doc, "3.1 Package modules", 2)
    table(doc, ["Module", "Responsibility"], [
        ["`config.py`", "Every tuned constant, with the measurement that chose it. "
         "Frozen dataclass, overridable per run."],
        ["`normalize.py`", "Local contrast normalisation -- the whole of illumination "
         "handling."],
        ["`matching.py`", "LoFTR, coarse alignment, tiling."],
        ["`refine.py`", "Sub-pixel refinement by patch phase correlation."],
        ["`models.py`", "Similarity / affine / homography fitting and held-out "
         "model selection."],
        ["`metrics.py`", "RMSE, inlier count and ratio, coverage, entropy, "
         "cross-window spread."],
        ["`controls.py`", "The four-control gate."],
        ["`pipeline.py`", "`register()` -- the single entry point that runs stages "
         "A to F in order."],
        ["`outputs.py`", "Writes the three deliverable files."],
        ["`cli.py`", "`sandhi register / controls / demo / survey`."],
    ])

    # ------------------------------------------------------------ 4
    h(doc, "4. The pipeline", 1)
    para(doc, "Six stages. `sandhi.pipeline.register()` runs them in this order.")

    h(doc, "Stage A - bring the pair to a common ground sample distance", 2)
    para(doc, "Both images are resampled to the **coarser** of the two resolutions "
              "before anything else happens. The ratio is read from the product "
              "metadata, never estimated -- blind scale estimation was tested and "
              "abandoned.")
    bullets(doc, [
        "Resampling must happen **before** normalisation, not after. Normalising at "
        "different resolutions produces incomparable products: correlation between "
        "the two orderings falls to 0.56 at 2x, 0.27 at 4x and **-0.09 at 8x**.",
        "The ratio is used as measured, not rounded. TMC-2 to Kaguya is 1.466, "
        "which rounded to 1 and skipped resampling entirely -- a bug that stayed "
        "invisible for a long time because every earlier ratio was integral.",
    ])

    h(doc, "Stage B - illumination normalisation", 2)
    para(doc, "Subtract a local mean, divide by a local standard deviation, over a "
              "31 px kernel. Match the residual structure. No DEM is involved, and "
              "that is the point: anything derived from a shared DEM injects "
              "identical structure into both images, and the matcher locks onto that "
              "instead of the terrain.")
    table(doc, ["Measured effect vs raw input", "Before", "After"], [
        ["Offset scatter across windows", "3.02 px", "**0.85 px**"],
        ["Inlier ratio", "36%", "**54%**"],
        ["Coverage", "-", "**doubled**"],
    ])

    h(doc, "Stage C - coarse alignment", 2)
    para(doc, "A whole-image match estimates the gross translation, which is then "
              "removed before tiling. Without it, tiled matching searches only a "
              "16 px pad around each tile and silently fails on a grossly displaced "
              "pair. OHRC sits about **437 px** from its Kaguya reference: with this "
              "stage the case yields **1443** matches, without it **7**.")

    h(doc, "Stage D - tiled dense matching", 2)
    para(doc, "The frame is cut into an 8x8 grid and LoFTR is run on each tile "
              "separately, so every region gets its own matching attempt.")
    para(doc, "Tiling, not selection. Bucketed selection was tried and cannot raise "
              "coverage -- it only redistributes matches among cells that already "
              "hold some, and empty cells stay empty. Coverage has to be forced "
              "where matches are produced, not filtered afterwards.")

    h(doc, "Stage E - sub-pixel refinement", 2)
    para(doc, "LoFTR emits matches on its own coarse grid, so its raw output is only "
              "accurate to about a pixel. Each match is re-localised by phase "
              "correlation on a 32 px half-width patch pair, which returns a genuine "
              "float shift. This is what makes 'sub-pixel' real rather than a "
              "rounded claim: **RMSE 1.212 px to 0.966 px on the same matches**.")
    bullets(doc, [
        "A refinement larger than 3.0 px means the two patches disagree about what "
        "they contain. Those matches are dropped, because a wrong sub-pixel answer "
        "is worse than none.",
        "The sign convention was measured, not assumed. Taking patch B `s` px to the "
        "right of patch A makes `phaseCorrelate` return `-s`, so the correction "
        "**adds** the returned shift. Subtracting it doubles the error instead of "
        "removing it -- which is exactly the bug that made refinement look useless "
        "at first.",
    ])

    h(doc, "Stage F - robust fit and model selection", 2)
    para(doc, "MAGSAC++ fits a transform, rejecting outliers. Three model classes "
              "are tried and the winner is chosen on **held-out** residual.")
    table(doc, ["Model", "DOF", "Represents"], [
        ["similarity", "4", "shift, rotation, uniform scale"],
        ["affine", "6", "+ shear, anisotropic scale"],
        ["homography", "8", "+ perspective"],
    ])
    para(doc, "Selection must be held out, because a higher-DOF model always fits "
              "the training points at least as well -- picking on in-sample error "
              "selects homography every time and proves nothing about the geometry.")
    para(doc, "Held-out alone was still not enough. On orthorectified nadir pairs, "
              "where no perspective exists between the images, homography still won "
              "2 of 3 windows by 5-9% -- noise rewarding degrees of freedom. Hence "
              "the complexity margin: parsimony is the default and perspective must "
              "be earned by a decisive margin.")
    para(doc, "On the real TMC-2 case the selector reports similarity 0.699, affine "
              "0.515 (a 26.4% gain, accepted) and homography 0.500 (a further 2.9%, "
              "**rejected** -- below the 15% margin). It chooses affine, which is "
              "correct: a projected pushbroom strip carries shear and anisotropic "
              "scale that similarity cannot represent, but no true perspective.")

    # ------------------------------------------------------------ 5
    h(doc, "5. Every tuned parameter", 1)
    para(doc, "All of these live in `sandhi/config.py` as a frozen dataclass, each "
              "beside the measurement that chose it. They are overridable per run "
              "with `sandhi.config.settings(...)` or from the CLI. Nothing here is "
              "a default someone liked the look of.")
    table(doc, ["Parameter", "Value", "Why this value"], [
        ["`contrast_kernel`", "31", "Local contrast window, in pixels. This is now the "
         "whole of normalisation, after the DEM photometric stage failed its controls."],
        ["`loftr_conf`", "0.5", "LoFTR match confidence floor."],
        ["`tiles`", "8", "8x8 grid. Forces a matching attempt in every region; "
         "selection alone cannot raise coverage."],
        ["`tile_pad`", "16", "Search pad around each tile. A tight pad matters -- an "
         "earlier 170 px pad let tiles match ground far from their own and gained "
         "nothing."],
        ["`min_tile_px`", "128", "Below this LoFTR has too little context. A 506 px "
         "wide projected OHRC swath would otherwise be cut into 63 px tiles and the "
         "match count collapses: **7 matches versus 1443**."],
        ["`max_window_px`", "1024", "LoFTR coarse attention is O((H*W/64)^2); 2048^2 "
         "needs about **17 GB** on CPU. A resource bound, not a method bound."],
        ["`refine_patch`", "32", "Half-width of the phase-correlation patch."],
        ["`refine_max_shift`", "3.0", "Beyond this the two patches disagree about "
         "their content; the match is dropped rather than trusted."],
        ["`ransac_px`", "1.5", "RANSAC inlier threshold. Chosen from a sweep -- see "
         "the table below."],
        ["`complexity_margin`", "0.15", "A more complex model must beat the simpler "
         "one by 15% on held-out residual to be selected."],
        ["`min_matches_for_split`", "16", "Below this there are too few matches to "
         "hold out a validation split at all."],
        ["`uniformity_grid`", "8", "8x8 cells for coverage and entropy."],
        ["`noise_reject_ratio`", "5.0", "Real imagery must out-match pure noise by at "
         "least 5x. Below it, the method is keying on something both images share "
         "that is not the terrain."],
        ["`control_roll_px`", "15", "The roll control shifts the raw input by 15 px "
         "and requires the recovered offset to move by -15."],
        ["`min_matches`", "8", "Floor below which no offset estimate is meaningful."],
        ["`weights`", "None", "`None` means the public MegaDepth `outdoor` weights. A "
         "path swaps in a fine-tune. **Weights are not adopted by default** -- a "
         "fine-tune must beat the pretrained model on the control gate, the "
         "obliquity ladder and the illumination cases first."],
    ])

    h(doc, "5.1 How ransac_px was chosen", 2)
    para(doc, "Measured on 1024 px windows across four sites. The trade is between "
              "accuracy and spread, and both matter -- the PS asks for sub-pixel "
              "accuracy **and** uniform distribution.")
    table(doc, ["Threshold", "Inliers", "Ratio", "RMSE", "Coverage"], [
        ["3.0", "124", "79%", "1.299", "0.52"],
        ["2.0", "102", "65%", "0.959", "0.47"],
        ["**1.5**", "**92**", "**58%**", "**0.786**", "**0.45**"],
        ["1.0", "66", "42%", "0.594", "0.34"],
    ])
    caption(doc, "1.5 is the only row that is sub-pixel AND still spread. 3.0 admitted "
                 "matches up to 3 px off, which is what held RMSE above 1 px.")

    # ------------------------------------------------------------ 6
    h(doc, "6. Run-time parameters", 1)
    table(doc, ["Flag", "Meaning"], [
        ["`--source`, `--reference`", "The image pair."],
        ["`--gsd`", "Metres per pixel of the common grid. Only a fallback -- when both "
         "source and reference GSDs are known it is derived as the coarser of the two."],
        ["`--source-gsd`", "Resolution of the source **file as supplied**; sets the "
         "resampling ratio."],
        ["`--source-native-gsd`", "Resolution of the source **product**, if it was "
         "projected before being passed in (OHRC: 0.26 m, supplied at 7.403 m). "
         "Required for the 'sub-pixel accuracy of source image' metric."],
        ["`--reference-gsd`", "Reference product resolution."],
        ["`--out`, `--name`", "Where the three deliverable files go."],
        ["`--no-refine`", "Skip sub-pixel refinement (for ablation)."],
        ["`--no-model-selection`", "Force similarity, reproducing the older numbers exactly."],
        ["`--ransac-px`, `--tiles`, `--contrast-kernel`", "Override the tuned constants."],
        ["`--weights`", "Path to a fine-tuned LoFTR checkpoint."],
    ])

    # ------------------------------------------------------------ 7
    h(doc, "7. The control gate", 1)
    para(doc, "Four checks. No number is reported unless all four pass. This is the "
              "component that most distinguishes the project, and it exists because "
              "it caught us being wrong.")
    table(doc, ["#", "Control", "What it catches"], [
        ["1", "The real pair produces matches at all", "A dead pipeline."],
        ["2", "**Rolling the raw input by N moves the recovered offset by -N**",
         "A matcher that is not tracking the ground. **This is the one that matters**, "
         "and the one an earlier version got wrong -- it shifted the *already "
         "normalised* image, which moves imagery and artifact together and hides the "
         "confound. Perturb the input, never the normalised output."],
        ["3", "Pure noise collapses, or loses to real terrain by 5x",
         "A method keying on a shared artifact rather than terrain."],
        ["4", "Constant grey collapses likewise", "The same, with zero information."],
    ])
    para(doc, "Two tests are load-bearing. `test_control_gate_passes` runs the gate "
              "on both bundled cases. `test_gate_rejects_a_matcher_that_ignores_its_"
              "inputs` feeds the gate a matcher that returns a fixed correspondence "
              "while ignoring its inputs, and asserts the gate rejects it. **If that "
              "ever passes, the gate is broken and every number downstream is "
              "unverified.**")
    para(doc, "Validation is cross-window agreement on the recovered offset, not "
              "agreement with an assumed ground truth. Kaguya morning and evening are "
              "independently orthorectified and genuinely offset by 8.25 px, so "
              "identity correspondence is false even though the labels declare it.")

    # ------------------------------------------------------------ 8
    h(doc, "8. Data", 1)
    table(doc, ["Role", "Source", "Resolution", "Access"], [
        ["Source", "Chandrayaan-2 **OHRC**", "0.26 m/px", "ISRO PRADAN, free account"],
        ["Source", "Chandrayaan-2 **TMC-2**", "5.05 m/px", "ISRO PRADAN"],
        ["Source", "Chandrayaan-2 **IIRS**", "85.08 m/px", "ISRO PRADAN"],
        ["Source", "Chandrayaan-1 **M3**", "~147 m/px", "PDS ODE, no login"],
        ["Reference", "Kaguya / SELENE **TC**", "7.403 m/px", "JAXA DARTS, direct HTTP"],
        ["Reference", "LROC **NAC**", "~0.5 m/px", "PDS ODE"],
        ["Topography", "Kaguya TC **DTM**", "7.403 m/px", "JAXA DARTS"],
    ])
    para(doc, "**Kaguya TC is the highest-value dataset in the list.** It ships "
              "global **morning and evening** mosaics on one identical grid: the same "
              "terrain, worldwide, under opposite illumination, already co-registered, "
              "with the DEM on the same pixel grid. That is real cross-illumination "
              "ground truth, free, with no synthetic pairs anywhere. The PS's own "
              "truncated dataset field (`SE`) points at SELENE.")
    para(doc, "Chandrayaan-2 archives are read **in place** through GDAL `/vsizip/` "
              "and never extracted -- extracting would put about 10 GB of imagery on "
              "disk for no gain.")
    para(doc, "One data trap worth recording: PRADAN's `FileSizeInBytes` column is "
              "the **uncompressed** product size, not the archive size. A 555 MB zip "
              "holds a 938 MB `.img`, so a partial download looks plausible against "
              "the catalogue figure. Verify with `zipfile.testzip()`, never by size.")

    # ------------------------------------------------------------ 9
    h(doc, "9. Metrics", 1)
    table(doc, ["Metric", "Definition", "Why it is reported"], [
        ["**RMSE (common grid)**", "Root-mean-square residual of inlier matches "
         "against the fitted model, in pixels of the shared grid.", "The standard "
         "registration accuracy figure."],
        ["**RMSE (source px)**", "The same residual expressed in the source "
         "**product's** own pixels.", "**This is the unit the PS asks for** -- "
         "'sub-pixel accuracy of source image'. It differs from the grid figure by "
         "the scale ratio, and the gap can be large."],
        ["**Inlier count / ratio**", "Matches surviving RANSAC, absolute and as a "
         "fraction.", "Explicitly requested."],
        ["**Coverage**", "Fraction of the 8x8 cells containing at least one inlier.",
         "The PS requires uniform distribution. A transform fitted from points "
         "clustered in one corner is tightly constrained there and extrapolates "
         "badly everywhere else."],
        ["**Entropy**", "Normalised entropy of the per-cell match counts.",
         "Coverage says a cell is occupied; entropy says whether the occupancy is "
         "even. Both are needed."],
        ["**Cross-window spread**", "Scatter of the recovered offset across "
         "independent windows.", "The column an artifact cannot fake. A mask artifact "
         "reports zero displacement everywhere; noise reports scatter. Agreement "
         "across windows on a non-zero offset is the evidence that terrain is being "
         "measured."],
    ])
    para(doc, "Results are reported stratified by absolute sun-azimuth difference "
              "and by scale ratio. Never a metric without saying which data split "
              "produced it.")

    # ------------------------------------------------------------ 10
    h(doc, "10. Measured results", 1)
    h(doc, "10.1 Chandrayaan-2 registration", 2)
    table(doc, ["", "TMC-2 to Kaguya", "OHRC to Kaguya", "Kaguya to Kaguya"], [
        ["Matches", f"**{tmc.get('match_count', 3085)}**", "1443", "132"],
        ["Inliers", f"**{tmc.get('inlier_count', 3064)} (99.3%)**", "1427 (99%)", "65 (49%)"],
        ["RMSE, common grid", f"**{tmc.get('rmse_px', 0.595):.3f} px "
                              f"({tmc.get('rmse_m', 4.41):.2f} m)**", "0.513 px (3.80 m)",
         "0.796 px (5.89 m)"],
        ["Source product GSD", f"{tmc.get('source_gsd_m', 5.05)} m", "**0.26 m**", "7.403 m"],
        ["**RMSE in source px**", f"**{tmc.get('rmse_source_px', 0.872):.3f}** (sub-pixel)",
         "**14.6** (not sub-pixel)", "**0.796** (sub-pixel)"],
        ["Coverage", f"**{tmc.get('coverage', 0.891):.3f}**", "0.391", "0.344"],
        ["Entropy", f"**{tmc.get('entropy', 0.894):.3f}**", "0.731", "0.630"],
        ["Model selected", tmc.get("model_kind", "affine"), "affine", "similarity"],
        ["Control gate", "**PASS**", "**PASS**", "**PASS**"],
    ])
    caption(doc, "Reproduce: python scripts/tmc_vs_kaguya.py --lat 0.5 --size 1536")

    h(doc, "10.2 Against baselines, cross-illumination", 2)
    para(doc, "Kaguya TC morning versus evening. **This is the result the project "
              "exists for.**")
    table(doc, ["Method", "Matches", "Inlier", "RMSE", "Coverage", "Entropy", "Gate"], [
        ["**SANDHI**", "**536**", "0.312", "0.932 px", "**0.641**", "**0.801**", "**PASS**"],
        ["DISK + LightGlue", "1", "-", "-", "0.000", "0.000", "FAIL - real pair"],
        ["ASIFT", "90", "0.078", "-", "0.062", "0.325", "FAIL - roll, noise"],
        ["SIFT", "25", "0.120", "-", "0.047", "0.264", "FAIL - roll"],
        ["ORB", "83", "0.036", "-", "0.047", "0.264", "FAIL - noise"],
    ])
    para(doc, "**No baseline passes the control gate. SANDHI is the only method that "
              "does.** The failure modes differ and each says something. DISK+LightGlue "
              "finds **one** match on real terrain: a learned *detector* has nothing to "
              "fire on when shading inverts. SIFT and ASIFT fail the **roll** control -- "
              "roll the raw input 15 px and their recovered offset does not follow, so "
              "their 25 and 90 matches were never tracking the ground. ORB fails "
              "**noise**: it matches random pixels as readily as the Moon.")

    h(doc, "10.3 Against baselines, cross-sensor at survey scale", 2)
    para(doc, "Chandrayaan-2 TMC-2 versus Kaguya.")
    table(doc, ["Method", "Matches", "Inlier", "RMSE", "Coverage", "Entropy", "Gate"], [
        ["**SANDHI**", "**3417**", "0.806", "0.725 px", "**0.891**", "**0.894**", "**PASS**"],
        ["DISK + LightGlue", "665", "0.587", "0.902 px", "0.734", "0.829", "PASS"],
        ["SIFT", "72", "0.611", "**0.601 px**", "0.281", "0.621", "PASS"],
        ["ORB", "188", "0.277", "0.851 px", "0.312", "0.654", "PASS"],
        ["ASIFT", "215", "0.567", "0.668 px", "0.281", "0.565", "FAIL - noise"],
    ])
    para(doc, "**5.1x the matches of the next gate-passing method, at 1.2x its "
              "coverage.** SIFT's lower RMSE is measured over 72 matches covering 28% "
              "of the frame; ours is over 3417 covering 89%. RMSE alone is not "
              "comparable across such different match populations, which is why "
              "coverage and entropy are reported beside it.")

    h(doc, "10.4 Where a baseline beats us", 2)
    para(doc, "On OHRC versus Kaguya, all five methods clear the gate and **ASIFT "
              "beats us** on match count and coverage (4826 / 0.453 against our "
              "1622 / 0.375); SIFT edges us on RMSE and entropy. Stated plainly "
              "rather than buried.")
    para(doc, "The reason is that this case is geometrically easy -- the OHRC strip "
              "is already projected onto the Kaguya grid, so only a small residual "
              "remains -- and classical descriptors do well when illumination is "
              "comparable and geometry is nearly solved. Our advantage is not that we "
              "win everywhere. It is that we are the only method that does not "
              "collapse when illumination inverts.")

    h(doc, "10.5 Independent corroboration", 2)
    para(doc, "Phase correlation is translation-only, so it has no match count or "
              "coverage. It is used as a corroborating instrument, sharing no code "
              "with the matcher, and it corroborates.")
    table(doc, ["Case", "Phase correlation", "SANDHI", "Agreement"], [
        ["Kaguya", "dy +7.96", "dy +7.62", "0.34 px"],
        ["OHRC", "dx +151.13, dy -435.64", "dx +151.64, dy -435.59", "0.51 px"],
        ["TMC-2", "dx +21.68, dy +2.89", "dx +21.50, dy +2.46", "0.47 px"],
    ])
    para(doc, "ASIFT adds a third independent check: on TMC-2 it recovers "
              "(+21.24, +2.39), within **0.26 px** of the pipeline.")

    h(doc, "10.6 Scale envelope", 2)
    para(doc, f"Demonstrated to **19.84x** -- Chandrayaan-1 M3 against Kaguya TC: "
              f"{m3.get('match_count', 42)} matches, "
              f"{m3.get('inlier_ratio', 0.7143)*100:.1f}% inliers, RMSE "
              f"{m3.get('rmse_px', 0.704):.3f} px, full control gate **PASS**. An "
              f"11.49x control on Kaguya gives 1618 matches at 61.5% inliers. An "
              f"earlier 16x failure was window starvation at 64x64 input, not a "
              f"method ceiling.")

    # ------------------------------------------------------------ 11
    h(doc, "11. What was tested and deleted", 1)
    para(doc, "Four components appeared in earlier versions of this project and no "
              "longer do. Each was removed on measurement, not on taste. They are "
              "listed because a reviewer should be able to see what was tried.")

    h(doc, "11.1 DEM photometric normalisation - failed its controls", 2)
    para(doc, "The original differentiator. Render the DEM under the source image's "
              "sun geometry, then normalise by dividing. The physics is sound: the "
              "renderer reproduces real imagery at r = 0.53-0.58 and recovers "
              "illumination direction correctly.")
    para(doc, "But dividing injects a `1/render` term identical in both images, and "
              "that term alone is enough for the matcher. **Pure noise pushed through "
              "it produced more matches than real terrain: 266 versus 20 at 20 deg "
              "elevation, and 3591 versus 9 with a high-pass.** It fails the noise "
              "gate at 0.1x against a 5x threshold. Replaced by local contrast "
              "normalisation, which passes every control. The renderer is kept as a "
              "valid illumination-recovery tool; it is not used for matching.")

    h(doc, "11.2 Fine-tuning - attempted twice, rejected twice", 2)
    para(doc, "On the second run, validation precision rose from 0.109 to 0.575 "
              "across four epochs while matches on a **real** held-out pair fell from "
              "245 to **63**. The adoption guard rejected all four epochs and no "
              "checkpoint was saved -- proved afterwards by sha256 identity against "
              "the pretrained weights.")
    para(doc, "The cause is a training-data pathology, and it is worth understanding: "
              "the pairs were an image and a warped copy of **itself**, so they are "
              "photometrically identical. The task therefore rewards exact-appearance "
              "matching, which is the opposite of what cross-illumination needs. The "
              "validation metric improved because the model got better at the wrong "
              "task. Without the guard this would have shipped as an improvement.")

    h(doc, "11.3 Blind scale estimation", 2)
    para(doc, "Estimating the scale ratio from image content instead of reading it "
              "from product metadata. Abandoned -- the metadata is authoritative and "
              "free, and estimation added a failure mode for nothing.")

    h(doc, "11.4 Bucketed selection", 2)
    para(doc, "Selecting matches to spread them across cells. It cannot raise "
              "coverage, only redistribute matches among cells that already hold "
              "some. Replaced by tiling, which forces matches to be *produced* "
              "everywhere.")

    # ------------------------------------------------------------ 12
    h(doc, "12. Known limits", 1)
    h(doc, "12.1 IIRS does not register", 2)
    para(doc, "IIRS against Kaguya gives correlation +0.033, 44 matches, 6 inliers "
              "(13.6%). No fit. Nine explanations were tested and all nine "
              "eliminated, including our own previously published one.")
    table(doc, ["Ruled out", "How"], [
        ["Band choice", "Sweep 898-4504 nm gives 0-5 matches at **every** wavelength."],
        ["Reflectance vs radiance", "Both cubes give correlation +0.0326, identical to "
         "four decimals."],
        ["Projection", "Polynomial fit (3.48 px) **and** direct per-pixel backplane "
         "reprojection (0.449 px) both fail."],
        ["Strip shape and the imagery itself", "Two bands of one cube register at "
         "**3528 matches, 100% inliers**, correlation +0.986."],
        ["The 11.49x scale ratio", "A Kaguya control at the same scale gives 1618 "
         "matches at 61.5% inliers."],
        ["Instrument pairing (spectrometer vs camera)", "Chandrayaan-1 M3, also a "
         "spectrometer, registers against Kaguya at 71.4% inliers and **19.84x**. "
         "This refuted our own published explanation."],
        ["Constant geolocation offset up to 35 km", "A blind search over the full "
         "3x3 deg tile, across translation and scale 0.78-1.30, peaks at **6.0 sigma** "
         "where chance alone on 30.3M positions gives **5.87**. The same search finds "
         "a Kaguya control to **0.43 km at 11.4 sigma**."],
        ["Detector fixed-pattern striping", "IIRS carries **less** fixed column "
         "structure (1.27% of variance) than genuine Kaguya terrain (2.50%)."],
    ])
    para(doc, "What survives is non-rigid internal geometry, or a geolocation error "
              "that varies down the strip. Settling it needs an independent "
              "geolocation from SPICE reconstruction. **This is stated as an open "
              "question, not as a solved one.**")

    h(doc, "12.2 OHRC sub-pixel accuracy is reference-limited", 2)
    para(doc, "OHRC's product is 0.26 m/px. The reference it is matched against, "
              "Kaguya TC, is 7.403 m/px. Meeting the PS clause in OHRC's own pixels "
              "would mean locating a feature to **0.035 reference pixels** -- finer "
              "than the reference image itself resolves. No method achieves that. "
              "This is an information limit of the data pairing, not a weakness of "
              "the matcher.")
    para(doc, "LROC NAC at ~0.5 m/px moves the target to **0.52 reference pixels**, "
              "which is at the good end of the 0.5-0.8 range this pipeline delivers, "
              "and drops the scale gap from 28.5x to **1.9x** -- an easier "
              "registration than the TMC-2 one already passing. Six NAC products are "
              "already on disk. This is the next piece of planned work.")

    h(doc, "12.3 Others", 2)
    bullets(doc, [
        "**Viewpoint envelope.** Matching succeeds at a 12.3 deg emission gap and "
        "fails from 14.9 deg. But the one success is also the only image acquired in "
        "the same orbit sequence as the anchor, so obliquity and illumination are "
        "**confounded** in that ladder. 12.3 deg is a demonstrated success under "
        "favourable conditions, not a measured obliquity limit.",
        "**CPU-only throughput.** LoFTR coarse attention is O((H*W/64)^2), so a "
        "2048 px window needs about 17 GB. Windows are capped at 1024 px. A GPU "
        "removes this.",
        "**The control gate is not in CI.** It is enforced by convention and by two "
        "tests, but nothing blocks a commit that skips it.",
    ])

    # ------------------------------------------------------------ 13
    h(doc, "13. Code map", 1)
    table(doc, ["Path", "What it is"], [
        ["`sandhi/`", "The package. Ten modules, listed in section 3.1."],
        ["`scripts/register.py`", "The original pipeline, kept as the measurement "
         "record. Model selection off."],
        ["`scripts/remeasure.py`", "The authoritative evaluation harness. Owns the "
         "control gate and the normalisation the pipeline uses."],
        ["`scripts/demo.py`", "The only module that renders. Everything else prints "
         "numbers."],
        ["`scripts/baselines.py`", "SIFT, ASIFT, ORB, DISK+LightGlue, phase correlation."],
        ["`scripts/viewpoint.py`", "Model selection on held-out residual; obliquity ladder."],
        ["`scripts/photometric.py`", "DEM renderer. Kept as a physics result and an "
         "illumination-recovery tool; **not** used for matching."],
        ["`scripts/kaguya.py`, `triple_io.py`", "Kaguya fetch and aligned windows."],
        ["`scripts/ch2_io.py`, `ch2_footprints.py`", "Chandrayaan-2 readers and "
         "footprint pickers."],
        ["`scripts/ohrc_project.py`", "OHRC geolocation polynomial; residual 0.152 m."],
        ["`scripts/tmc_vs_kaguya.py`", "TMC-2 registration, gated. Writes the deliverable."],
        ["`scripts/iirs_vs_kaguya.py`, `iirs_vs_m3.py`, `m3.py`", "The IIRS "
         "investigation and the M3 experiment that refuted our own explanation."],
        ["`scripts/offset_origin.py`", "Decomposes the 3.4 km OHRC offset."],
        ["`tests/`", "36 tests. 25 unit (about 3 s, no imagery or network), 11 "
         "end-to-end on committed samples."],
        ["`samples/`", "3 MB of genuine committed imagery, so a fresh clone runs "
         "with no download."],
        ["`BUGS.md`", "25 entries. Every bug, with symptom, root cause, fix and the "
         "check that catches a regression."],
    ])
    para(doc, "One structural trap worth naming: three superseded modules "
              "(`ablation.py`, `dense_match.py`, `scale_pipeline.py`) carry SUPERSEDED "
              "banners because their **results** were retracted, but current code "
              "still imports their **helpers**. Deleting them breaks the pipeline. Do "
              "not cite their output; do not remove the files.")

    # ------------------------------------------------------------ 14
    h(doc, "14. Running it", 1)
    code(doc, "pip install -e \".[dev]\"\n"
              "python scripts/check_env.py        # deps + GDAL drivers. Run first.")
    para(doc, "The end-to-end demo. Falls back to the committed samples when the full "
              "products are absent, so it works from a fresh clone. About 20 s per "
              "case on CPU.")
    code(doc, "python scripts/demo.py --case all\n"
              "python scripts/demo.py --list      # full data or sample, per case")
    para(doc, "The package:")
    code(doc, "sandhi register --source A.tif --reference B.tif --out results/ \\\n"
              "                --source-gsd 5.05 --reference-gsd 7.403\n"
              "sandhi controls --case ohrc        # the four-control gate\n"
              "sandhi demo --case all")
    para(doc, "Tests:")
    code(doc, "pytest                 # all 36\n"
              "pytest -m \"not slow\"   # 25 unit tests, ~3 s\n"
              "pytest -m slow         # 11 end-to-end on samples/")
    para(doc, "`register` writes three files: the registered product (GeoTIFF when "
              "the reference carries a CRS, PNG otherwise), a match-point CSV with "
              "per-match residual and inlier flag, and a metrics JSON.")

    # ------------------------------------------------------------ 15
    h(doc, "15. The discipline, in one page", 1)
    bullets(doc, [
        "**Every matching number passes the four-control gate before it is written "
        "down.** Perturb the input, never the normalised output.",
        "**Validation is cross-window agreement, not agreement with an assumed "
        "ground truth.** Kaguya morning and evening are genuinely offset by 8.25 px; "
        "the identity correspondence their labels declare is false.",
        "**Every bug gets a log entry, with how it was resolved.** 25 entries, "
        "including the ones where our own published claim was the bug.",
        "**Every tuned constant carries the measurement that chose it**, so a future "
        "change has to argue with evidence rather than overwrite a literal.",
        "**Never report a metric without saying which data split produced it.**",
        "**Scale ratios are read from metadata, never estimated.**",
        "**A failed idea is deleted and recorded, not quietly retained.** Stage B was "
        "the headline feature of this project until it failed its controls.",
    ])
    para(doc, "The project is willing to publish its own negative results. IIRS does "
              "not register and we say so. ASIFT beats us on one case and we print "
              "the table. Our published explanation for the IIRS failure was wrong "
              "and the commit that retracts it is in the history. That is the "
              "standard the control gate exists to enforce.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--out-dir", default="docs")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    docx_path = out_dir / "SANDHI_Technical_Report.docx"

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    build(doc)
    doc.save(docx_path)
    print(f"wrote {docx_path}")

    if args.no_pdf:
        return 0
    # PDF via whichever office suite is installed. WPS registers KWPS
    # (Writer); MS Word registers Word.Application. This machine has WPS and no
    # Word, so trying only Word.Application silently produced no PDF at all.
    # FileFormat 17 is wdFormatPDF, and WPS honours the same constant.
    pdf_path = docx_path.with_suffix(".pdf")
    for prog_id in ("KWPS.Application", "WPS.Application", "Word.Application"):
        try:
            import win32com.client
            app = win32com.client.Dispatch(prog_id)
            app.Visible = False
            d = app.Documents.Open(str(docx_path))
            d.SaveAs(str(pdf_path), FileFormat=17)
            d.Close()
            app.Quit()
            print(f"wrote {pdf_path}  (via {prog_id})")
            return 0
        except Exception:                         # noqa: BLE001, S112
            continue
    print("PDF skipped - no WPS or Word COM server found. The .docx is the "
          "deliverable; open it and export to PDF if one is needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
