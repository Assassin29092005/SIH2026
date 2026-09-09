"""The four-control gate. No number is reported unless it passes.

This exists because an earlier version of this project reported **100% inlier
rates that were entirely artifact**. A shared zero-mask written identically into
both images gave the matcher a perfectly aligned, high-contrast pattern to lock
onto, and the ground truth was independently wrong by 8.25 px. Two errors
pointing the same way produced a beautiful, false result (BUGS.md BUG-011).

The four checks:

1. the real pair produces matches at all
2. **rolling the RAW input by N moves the recovered offset by -N**
3. pure noise collapses, or loses to real terrain by `noise_reject_ratio`
4. constant grey collapses, or likewise

Rule 2 is the one that matters, and the one an earlier version got wrong. It
shifted the *already-normalised* image, which moves imagery and artifact
together and hides the confound. **Perturb the input, never the output.**

Rules 3 and 4 are graded rather than binary: a handful of chance matches from
noise is not disqualifying, but noise matching nearly as well as terrain means
the method is keying on something that is not the terrain. Stage B fails this at
0.1x — noise passed through it matched *better* than real ground.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config, metrics


@dataclass
class ControlReport:
    passed: bool
    checks: list = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{'PASS' if self.passed else 'FAIL'}  control gate"]
        for c in self.checks:
            lines.append(f"  {c['name']:<18} {c['detail']:<44} "
                         f"{'PASS' if c['passed'] else 'FAIL'}")
        return "\n".join(lines)


def run(match_fn, src: np.ndarray, ref: np.ndarray, *,
        roll_px: int | None = None, seed: int = 0) -> ControlReport:
    """Run the gate.

    `match_fn(src, ref)` must accept two RAW images and return (pts_a, pts_b) —
    it should include whatever normalisation the method under test uses, so the
    controls exercise the real path.
    """
    cfg = config.get()
    roll = cfg.control_roll_px if roll_px is None else roll_px
    rng = np.random.default_rng(seed)
    checks = []

    pa, pb = match_fn(src, ref)
    base = metrics.offset_of(pa, pb)
    n_real = len(pa)
    if base is None:
        checks.append({"name": "real pair", "passed": False,
                       "detail": f"only {n_real} matches"})
        return ControlReport(False, checks)
    checks.append({"name": "real pair", "passed": True,
                   "detail": f"n={n_real}  dx={base[0]:+.2f} dy={base[1]:+.2f}"})

    # 2 -- shift the RAW input
    rpa, rpb = match_fn(np.roll(src, roll, axis=1), ref)
    rolled = metrics.offset_of(rpa, rpb)
    if rolled is None:
        checks.append({"name": f"roll +{roll} raw", "passed": False,
                       "detail": "collapsed, cannot verify"})
    else:
        delta = rolled[0] - base[0]
        ok = abs(delta + roll) < 4.0
        checks.append({"name": f"roll +{roll} raw", "passed": ok,
                       "detail": f"dx moved {delta:+.2f} (want {-roll:+d})"})

    # 3, 4 -- degenerate inputs must not match
    finite = src[np.isfinite(src)]
    mu = float(finite.mean()) if finite.size else 0.0
    sd = float(finite.std()) if finite.size else 1.0
    for name, fake in (("noise", rng.normal(mu, sd, src.shape)),
                       ("constant", np.full(src.shape, mu))):
        fpa, _ = match_fn(fake.astype(np.float32), ref)
        n_fake = len(fpa)
        ratio = n_real / max(n_fake, 1)
        ok = n_fake < cfg.min_matches or ratio >= cfg.noise_reject_ratio
        checks.append({"name": name, "passed": ok,
                       "detail": f"n={n_fake} vs real {n_real}, ratio {ratio:.1f}x"})

    return ControlReport(all(c["passed"] for c in checks), checks)


def default_match_fn(tiled: bool = True):
    """A `match_fn` for the production pipeline, for use with `run`."""
    from . import matching, normalize

    def fn(src, ref):
        a8, b8 = normalize.prepare(src), normalize.prepare(ref)
        if tiled:
            pa, pb, _, _, _ = matching.match_aligned(a8, b8)
        else:
            pa, pb, _ = matching.match(a8, b8)
        return pa, pb

    return fn
