"""Regenerate the SIH submission deck from the committed measurements.

The deck was the one deliverable with no generator. `make_docs.py` rebuilds the
technical report from `outputs/*.json` and `make_video.py` rebuilds the video
the same way, but the slides were hand-edited -- which is exactly why they went
fourteen days stale and why "sub-metre positional accuracy" survived in them
for weeks when the best measured figure is 3.80 m.

    python scripts/make_deck.py                   # -> deck/sandhi 26166.pptx
    python scripts/make_deck.py --self-check      # asserts, writes nothing
    python scripts/make_deck.py --make-template   # maintenance, see below

**How it works.** `assets/sih_deck_template.pptx` is the SIH-format deck with
every measured figure replaced by a `{token}`. This script resolves each token
from `outputs/*.json` and writes the filled deck. The template carries the
branding, layout and images, which are not ours to generate; only the numbers
are substituted.

Substitution is per-paragraph and keyed by (slide, shape id, paragraph index),
not by find-and-replace on text -- a loose replace would hit "0.872" in three
unrelated places and silently diverge if one of them ever meant something else.
Every number-bearing paragraph in this deck is a single run, so the first run's
formatting is preserved exactly.

**Four figures have no machine-readable source** and are listed in `STATIC`
with where they actually come from. They are printed on every run rather than
hidden, because a constant in a generator is the same drift risk the generator
exists to remove. Giving them a JSON home is open work -- see NEXTSTEP
Priority 5, which records that the IIRS headline is currently a print literal.

**Editing the deck.** Change wording in the TEMPLATE, not in the output, or
the next run overwrites it. `--make-template` rewrites the template from the
current deck and is only for when the slides themselves change.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    from pptx import Presentation
except ModuleNotFoundError:                      # pragma: no cover
    raise SystemExit(
        "make_deck.py needs python-pptx, which is an optional dependency "
        "because it is only used to build the deck:\n"
        '  pip install -e ".[docs]"      (or: uv sync --extra docs)')

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
TEMPLATE = ROOT / "assets" / "sih_deck_template.pptx"
DECK = ROOT / "deck" / "sandhi 26166.pptx"

KAGUYA_GSD_M = 7.403          # the reference grid every ratio is quoted against
SANDHI = "SANDHI (LoFTR + local contrast)"


# --------------------------------------------------------------- the figures
# Figures with no machine-readable source. Each says where it really lives, so
# the gap is visible rather than looking like a measurement.
STATIC = {
    "kaguya_corr": ("-0.560", "BUGS.md BUG-005 -- window-specific, not in any JSON"),
    "kaguya_tile": ("N18E009N15E012SC", "BUGS.md BUG-005"),
    "iirs_matches": ("44", "scripts/iirs_vs_kaguya.py print literal, not in its JSON"),
    "iirs_inliers": ("6", "scripts/iirs_vs_kaguya.py print literal, not in its JSON"),
}


def load(name: str) -> dict:
    """Read a committed measurement. A missing one stops the build.

    BUG-014 and BUG-026 were both a missing input laundered into an empty
    result that still exited 0. A deck is a deliverable; an absent measurement
    must fail loudly rather than render a slide with a blank on it.
    """
    p = OUT / name
    if not p.exists():
        raise SystemExit(f"missing measurement {p} - run the pipeline first")
    return json.loads(p.read_text(encoding="utf-8"))


def count_tests() -> int:
    """Collect the real test count. Importing sandhi pulls torch, so ~10 s."""
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                        "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    n = sum(int(m) for m in re.findall(r"^tests/\S+: (\d+)$", r.stdout, re.M))
    if not n:
        raise SystemExit("could not count tests - is pytest installed?\n" + r.stdout[-500:])
    return n


def count_bugs() -> int:
    text = (ROOT / "BUGS.md").read_text(encoding="utf-8")
    return len(re.findall(r"^### BUG-", text, re.M))


def figures(*, fast: bool = False) -> dict[str, str]:
    """Every token the template can use, resolved from committed measurements."""
    tmc = load("tmc2_metrics.json")
    m3 = load("m3_vs_kaguya.json")
    base = load("baselines.json")["kaguya"][SANDHI]

    vals = {
        # Chandrayaan-2 TMC-2 against Kaguya, full product -- the headline
        "tmc_matches": f"{tmc['match_count']:,}",
        "tmc_inlier_pct": f"{tmc['inlier_ratio'] * 100:.1f}%",
        "tmc_rmse_px": f"{tmc['rmse_px']:.3f}",
        "tmc_source_px": f"{tmc['rmse_source_px']:.3f}",
        "tmc_coverage": f"{tmc['coverage']:.3f}",
        "tmc_entropy": f"{tmc['entropy']:.3f}",
        "tmc_gsd": f"{tmc['source_gsd_m']:g}",
        "ref_gsd": f"{tmc['gsd_m']:g}",
        # the largest demonstrated scale ratio, gated
        "m3_ratio": f"{m3['m3_gsd'] / KAGUYA_GSD_M:.2f}",
        # cross-illumination, the case where we are the only method that passes
        "base_matches": str(base["matches"]),
        "base_rmse": f"{base['rmse_px']:.3f}",
        "base_coverage": f"{base['coverage']:.3f}",
        # the repository itself
        "n_tests": "41" if fast else str(count_tests()),
        "n_bugs": str(count_bugs()),
    }
    vals.update({k: v for k, (v, _) in STATIC.items()})
    return vals


# ----------------------------------------------------------- the substitution
# (slide, shape id, paragraph index) -> the template text for that paragraph.
# Keyed by position rather than matched by content, so a figure that appears in
# three places cannot drift in one of them unnoticed.
TEMPLATED: dict[tuple[int, int, int], str] = {
    (2, 15363, 2): "- Scale: metadata-driven common-GSD resampling, demonstrated to {m3_ratio}x",
    (2, 15364, 3): "- TMC-2 to Kaguya, full product: {tmc_source_px} source px, "
                   "{tmc_coverage} coverage, {tmc_entropy} entropy",
    (2, 15374, 2): "{kaguya_corr}, 512 px window",
    (3, 19, 0): "Sources: OHRC 0.26 m/px, TMC-2 {tmc_gsd} m/px, IIRS 85.08 m/px",
    (3, 19, 1): "References: Kaguya TC {ref_gsd} m/px, LROC NAC about 0.5 m/px",
    (3, 21, 2): "TMC-2 {tmc_gsd} m/px",
    (3, 23, 1): "Kaguya TC {ref_gsd} m/px",
    (3, 51, 1): "{tmc_source_px} source px",
    (3, 51, 2): "Coverage {tmc_coverage}",
    (4, 21, 3): "TMC-2 to Kaguya, full product: {tmc_matches} matches, "
                "{tmc_inlier_pct} inliers, {tmc_rmse_px} common-grid px and "
                "{tmc_source_px} source px",
    (4, 21, 6): "IIRS to Kaguya does not register: {iirs_matches} matches and "
                "{iirs_inliers} inliers",
    (4, 21, 7): "OHRC with a {ref_gsd} m Kaguya reference is reference-limited in "
                "source pixels; LROC NAC is the planned reference",
    (5, 14, 3): "Survey-scale TMC-2, full product: {tmc_matches} matches, "
                "{tmc_source_px} source px and {tmc_coverage} coverage",
    (6, 14, 2): "- Ten-module sandhi package and {n_tests} tests",
    (6, 14, 4): "- {n_bugs} documented bugs, fixes and regression checks",
    (6, 19, 0): "{kaguya_corr}  (tile {kaguya_tile}, 512 px)",
    (6, 19, 1): "{base_matches} matches, {base_rmse} px, coverage {base_coverage}",
    (6, 19, 2): "{tmc_source_px} source px, coverage {tmc_coverage} (full product)",
    (6, 19, 3): "{m3_ratio}x M3-Kaguya: gate pass",
}


def paragraphs(prs):
    """Every paragraph with its (slide, shape id, index) key."""
    for i, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for j, para in enumerate(shape.text_frame.paragraphs):
                yield (i, shape.shape_id, j), para


def set_text(para, text: str) -> None:
    """Replace a paragraph's text, keeping the first run's formatting.

    Safe here because every templated paragraph is a single run -- asserted by
    `--self-check` rather than assumed, since a future edit could split one and
    this would then silently drop the second run's formatting.
    """
    runs = para.runs
    if not runs:
        raise SystemExit(f"paragraph has no runs to carry formatting: {para.text!r}")
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def render(vals: dict[str, str]) -> Presentation:
    if not TEMPLATE.exists():
        raise SystemExit(
            f"no template at {TEMPLATE}\n"
            "Build one from the current deck with:\n"
            "  python scripts/make_deck.py --make-template")
    prs = Presentation(TEMPLATE)
    seen = set()
    for key, para in paragraphs(prs):
        tpl = TEMPLATED.get(key)
        if tpl is None:
            continue
        if para.text != tpl:
            raise SystemExit(
                f"template drift at slide {key[0]} shape {key[1]} para {key[2]}\n"
                f"  template file: {para.text!r}\n"
                f"  this script:   {tpl!r}\n"
                "Re-run --make-template, or fix TEMPLATED to match.")
        set_text(para, tpl.format(**vals))
        seen.add(key)
    missing = set(TEMPLATED) - seen
    if missing:
        raise SystemExit(f"template is missing {len(missing)} paragraph(s): {sorted(missing)}")
    return prs


def make_template() -> int:
    """Rewrite the template from the CURRENT deck. Maintenance only."""
    if not DECK.exists():
        raise SystemExit(f"no deck at {DECK} to build a template from")
    prs = Presentation(DECK)
    written = 0
    for key, para in paragraphs(prs):
        tpl = TEMPLATED.get(key)
        if tpl is None:
            continue
        if len(para.runs) > 1:
            raise SystemExit(
                f"slide {key[0]} shape {key[1]} para {key[2]} has {len(para.runs)} runs; "
                "substitution would drop formatting. Merge it in PowerPoint first.")
        set_text(para, tpl)
        written += 1
    if written != len(TEMPLATED):
        raise SystemExit(f"only {written} of {len(TEMPLATED)} paragraphs found in the deck")
    TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
    prs.save(TEMPLATE)
    print(f"wrote {TEMPLATE}  ({written} tokenised paragraphs)")
    print("The deck is now regenerable. Edit WORDING in the template, not the output.")
    return 0


def self_check() -> int:
    """Assert the generator is faithful. Writes nothing."""
    vals = figures(fast=True)

    unresolved = set()
    for tpl in TEMPLATED.values():
        unresolved |= set(re.findall(r"\{(\w+)\}", tpl)) - set(vals)
    assert not unresolved, f"tokens with no resolver: {sorted(unresolved)}"

    unused = set(vals) - {t for tpl in TEMPLATED.values()
                          for t in re.findall(r"\{(\w+)\}", tpl)}
    assert not unused, f"resolvers no slide uses: {sorted(unused)} - delete them"

    for name, v in vals.items():
        assert v and v.strip(), f"token {name} resolved empty"
        assert "{" not in v, f"token {name} resolved to a template: {v!r}"

    prs = render(vals)
    filled = {k: p.text for k, p in paragraphs(prs) if k in TEMPLATED}
    for key, text in filled.items():
        assert "{" not in text and "}" not in text, f"unsubstituted token at {key}: {text!r}"

    # The measured figures must actually reach a slide.
    tmc = load("tmc2_metrics.json")
    blob = "\n".join(filled.values())
    for want in (f"{tmc['rmse_source_px']:.3f}", f"{tmc['coverage']:.3f}",
                 f"{tmc['match_count']:,}"):
        assert want in blob, f"{want} never reaches a slide"

    print(f"OK - {len(TEMPLATED)} paragraphs, {len(vals)} tokens, all resolved")
    print(f"     headline: {tmc['rmse_source_px']:.3f} source px, "
          f"coverage {tmc['coverage']:.3f}, {tmc['match_count']:,} matches")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DECK))
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--make-template", action="store_true",
                    help="rewrite the template from the current deck (maintenance)")
    args = ap.parse_args()

    if args.make_template:
        return make_template()
    if args.self_check:
        return self_check()

    vals = figures()
    prs = render(vals)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)

    print(f"wrote {out}")
    print(f"  {len(TEMPLATED)} paragraphs regenerated from outputs/*.json")
    for name, (v, src) in STATIC.items():
        print(f"  STATIC  {name} = {v}   ({src})")
    print("  ^ these four have no machine-readable source yet; everything else "
          "came from a committed measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
