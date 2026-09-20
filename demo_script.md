# Demo script — SANDHI, PS SIH26166

A script for **you** to record. Not the rendered video (`scripts/make_video.py`
makes that one). This is what to say, what to have on screen while you say it,
and what to do before you hit record.

Target: **4–5 minutes.** The evidence chain matters more than the algorithm
inventory — judges see a lot of pipelines and very few projects that can show
their own result being wrong and caught.

Every number below is checked against the committed measurements. If you change
a number here, change it because you re-measured, not because it sounds better.

---

## Before you record

Matching takes 15–30 s per pair on a laptop CPU. **Do not record dead air.**
Run everything once first, so the outputs exist and you are re-showing them.

```bash
cd D:\SIH
python scripts/demo.py --case all       # ~60 s total, writes outputs/demo_*.png
python scripts/check_gate_reports.py    # instant
pytest -m "not slow"                    # ~16 s
```

Have these open in separate windows or tabs, and rehearse switching:

| # | Window | What is in it |
|---|---|---|
| 1 | Terminal at `D:\SIH` | for the live commands |
| 2 | `outputs/demo_tmc.png` | the headline figure |
| 3 | `outputs/demo_ohrc.png` | the limit you state honestly |
| 4 | `outputs/tmc2_metrics.json` | the numbers, including the gate block |
| 5 | `BUGS.md` scrolled to **BUG-011** | the retraction |

Settings: 1080p, terminal font size up (judges may watch on a phone), close
notifications, and check your mic before the real take.

One honesty note that will save you a hard question: the figures in
`outputs/` were measured on the **full products**. A fresh clone with no
imagery runs on the committed 5 MB sample crops and gets slightly different
numbers (0.849 source px instead of 0.872). Say "full product" when you quote
the headline. The demo prints which one it used.

---

## The script

### 0. Open — 20 s

> "This is SANDHI, our entry for problem statement 26166 from ISRO. The task is
> to find correspondence between Chandrayaan-2 optical images and lunar
> reference imagery — at sub-pixel accuracy, with match points spread evenly
> across the image.
>
> I want to start with why this is hard, because the difficulty is not where
> people usually expect."

*On screen: window 1, terminal, nothing running yet.*

---

### 1. The problem, in one number — 45 s

*Switch to window 2 (`demo_tmc.png`) — or better, have the Kaguya
morning/evening pair visible.*

> "These are two images of the **same lunar ground**. One taken in lunar
> morning, one in lunar evening. Same terrain, same sensor, same map
> projection.
>
> Their raw correlation is **minus zero point five six zero**.
>
> Not weakly correlated — *anti*-correlated. The Moon has no atmosphere, so
> there is no fill light. A crater lit from the east looks almost exactly like a
> **dome** lit from the west. Shadows do not soften, they invert.
>
> And this is after the standard photometric correction has already been
> applied. Both products carry it. It normalises the reflectance function
> without a terrain model, so the shading and the cast shadows survive
> untouched."

**Say the window out loud.** It is measured on tile N18E009N15E012SC, a 512
pixel window. Other windows of the same pair give −0.45 to −0.72. If you say
"−0.560" flat and a judge runs the self-check and sees −0.627, you look like
you rounded something. Saying it first makes you look like you measured it.

---

### 2. What we do about it — 40 s

> "Our answer is deliberately not clever. We do **not** render the terrain and
> divide it out — we tried that, and I will come back to why we threw it away.
>
> We use local contrast normalisation: subtract a local mean, divide by a local
> standard deviation. No DEM. That leaves the terrain structure and removes the
> illumination, which is low-frequency.
>
> Then a detector-free dense matcher — LoFTR — because lunar terrain is
> texture-poor and corner detectors starve on it. Tiled, so every region of the
> image gets its own matching attempt, which is what gives us uniform coverage.
> Sub-pixel refinement by phase correlation. Then a robust fit, where we choose
> between similarity, affine and homography on **held-out** residual, not
> in-sample — because a higher-DOF model always wins in-sample."

---

### 3. The control gate — 60 s. **This is your differentiator.**

*Switch to window 5, `BUGS.md` at BUG-011.*

> "Here is the part I actually want to be judged on.
>
> An earlier version of this project reported **one hundred percent inlier
> rates**. It was completely false. We had written a shared zero-mask into both
> images, identically, and the matcher locked onto that pattern instead of the
> terrain. Two independent errors happened to agree, and the result looked
> beautiful.
>
> We caught it, retracted it, and it is bug eleven in our log — which is public.
>
> So now every number this project reports has to pass four controls before we
> are allowed to write it down."

*Switch to window 1. Run it live — this one is fast:*

```bash
python scripts/check_gate_reports.py
```

> "**One** — the real pair has to produce matches.
> **Two** — roll the *raw input* by fifteen pixels, and the recovered offset has
> to move by minus fifteen. That one is the important one. An earlier version of
> this test shifted the already-normalised image, and it passed — because that
> moves the imagery and the artifact together. You have to perturb the input.
> **Three** — pure noise has to collapse, or lose to real terrain by five times.
> **Four** — flat grey has to collapse.
>
> This runs in CI. There is a test that hands the gate a matcher which ignores
> its inputs entirely and returns a fixed answer, and asserts the gate rejects
> it. If that test ever passes, our gate is broken and every number downstream
> is unverified."

---

### 4. The result — 60 s

*Switch to window 2, `demo_tmc.png`.*

> "Chandrayaan-2 TMC-2 against Kaguya Terrain Camera. Cross-sensor,
> cross-illumination, and a scale difference read from the product metadata.
>
> Three thousand and eighty-five matches. Ninety-nine point three percent
> inliers. RMSE **zero point five nine five** pixels on the common grid.
>
> But that is not the number the problem statement asks for. It asks for
> sub-pixel accuracy **of the source image** — so it has to be stated in TMC-2's
> own pixels, not the shared grid. That is **zero point eight seven two source
> pixels**. Sub-pixel, in the unit the PS actually specifies."

*Point at the match-distribution panel, bottom right of the figure.*

> "And coverage zero point eight nine, entropy zero point eight nine. That panel
> is the match distribution on an eight-by-eight grid. 'Maintaining uniform
> distribution across the images' is an explicit requirement in the problem
> statement, and it is the one most teams skip — so we made it a first-class
> metric, not a footnote."

*Optional, if you have 20 s spare — this lands well:*

> "The offset we recover is independently confirmed by phase correlation, an
> algorithm that shares no code with our matcher, agreeing to within half a
> pixel. Two unrelated methods agreeing is what tells you it is terrain and not
> an artifact."

---

### 5. The limits, stated plainly — 50 s

*Switch to window 3, `demo_ohrc.png`. Do not skip this section.*

> "Two things do **not** work, and I would rather tell you than have you find
> them.
>
> **OHRC.** Our highest-resolution source, twenty-six centimetres per pixel. We
> register it, it passes the gate, RMSE zero point five one three on the grid.
> But in OHRC's *own* pixels that is **fourteen point six** — it fails the
> sub-pixel clause.
>
> That is not a matcher problem, and I want to be precise about why. The
> reference is Kaguya at seven point four metres. Locating a feature to better
> than twenty-six centimetres against it means zero point zero three five
> *reference* pixels — finer than the reference image itself resolves. No method
> achieves that. The fix is a finer reference, not a better algorithm: LROC NAC
> at half a metre puts the target at zero point five two reference pixels, which
> is inside the range we already deliver. The data is on disk. The pairing is
> the next piece of work.
>
> **IIRS.** It does not register. Forty-four matches, six inliers. We eliminated
> band choice across the whole spectrum, reflectance versus radiance, the
> projection method, the strip shape, and the scale ratio — each by measurement.
> We even brought in Chandrayaan-1 M3 to test whether a spectrometer can match a
> camera at all: it can, at nearly twenty times scale, passing the full gate.
> So the instrument pairing is not the explanation either. What survives is
> IIRS's own geolocation, and confirming that needs geometry from outside the
> product. We state it as open, not solved."

---

### 6. Close — 35 s

*Switch to window 1.*

> "Everything I have shown runs from a fresh clone. Five megabytes of real
> cropped imagery is committed, so there is no account, no download and no GPU
> needed to reproduce it."

```bash
python scripts/demo.py --case tmc
```

*While it runs (~20 s), keep talking — do not wait in silence:*

> "Forty-one tests, the control gate in continuous integration, thirty-one bugs
> logged with root cause and fix, and three approaches we tested and abandoned
> written down so nobody pays for them twice — including the DEM-based
> normalisation that was our original thesis and failed its own noise control.
>
> The thing I would point at is not the RMSE. It is that when this project was
> wrong, the project caught it. That is what we built."

*Let the figure appear. End on it.*

---

## If a judge asks

**"Why LoFTR and not SIFT?"**
> We measured it. On cross-illumination, SIFT and ASIFT fail the roll control —
> roll the input fifteen pixels and their recovered offset does not follow, so
> their matches were never tracking the ground. DISK with LightGlue finds
> exactly one match on real terrain. ORB matches noise as readily as the Moon.
> SANDHI is the only method we tested that passes the gate on that case.

**"Where does SIFT beat you?"**
> On OHRC against Kaguya, ASIFT beats us on match count and coverage, and SIFT
> edges us on RMSE. That case is geometrically easy — the strip is already
> projected onto the reference grid, so only a small residual remains, and
> classical descriptors do well when illumination is comparable. It is in our
> README. Our claim is not that we win everywhere; it is that we are the only
> one that does not collapse when illumination inverts.

**"Did you train the model?"**
> No, and that is a finding rather than an omission. We fine-tuned on warped
> lunar crops twice. Both runs improved their own training metric several times
> over and destroyed real cross-illumination matching — the second collapsed
> within one epoch. A warped self-pair is photometrically identical, so the
> cheapest solution is exact appearance matching, which is precisely the crutch
> a matcher must give up to survive an illumination change. Pretrained weights
> are the default and the checkpoints were never adopted.

**"How fast is it?"**
> Fifteen to thirty seconds per pair on a laptop CPU. It is already PyTorch, so
> a GPU is device placement rather than a rewrite — we would expect one to two
> seconds, and it also removes the memory bound on large windows.

**"Can it do a whole strip?"**
> Not yet, and the blocker is data rather than code. A full TMC-2 strip spans
> about fifty-seven degrees of latitude and we have reference tiles covering
> five percent of it. The tiling and the per-tile consistency check are about a
> day of work; the sixteen gigabytes of reference is the real cost.

**"What is the scale limit?"**
> Nearly twenty times, demonstrated and gated, on M3 against Kaguya. An earlier
> table said eight, but that was window starvation — we were shrinking the
> output window as we raised the ratio, not hitting a limit of the method. We
> corrected it in the bug log rather than quietly editing the table.

---

## Do not say

- **"Sub-metre accuracy."** Our best is 3.80 m. The only sub-metre figure in the
  project is the OHRC geolocation *fit residual*, which is a different quantity.
  An earlier version of the deck said this and it was wrong.
- **"Works at any scale / any illumination."** Say the measured envelope.
- **"Sub-pixel"** without saying *which* pixel. Grid pixels and source pixels
  differ by the scale ratio, and on OHRC that is the difference between passing
  and failing.
- **"100% accurate", "fully automated", "production ready."** None are true, and
  the project's whole credibility rests on not saying things like that.
- Any number without its data split. Full product and sample crop differ.

---

## Recording checklist

- [ ] All five windows open and ordered
- [ ] `outputs/` populated (run `demo.py --case all` first)
- [ ] Terminal font readable at 1080p on a phone screen
- [ ] Notifications off
- [ ] Mic tested — bad audio sinks a good demo
- [ ] Said "full product" wherever you quoted a headline number
- [ ] Stated the OHRC limit and the IIRS failure out loud
- [ ] Under 5 minutes
- [ ] Uploaded **unlisted**, link pasted into the SIH portal
