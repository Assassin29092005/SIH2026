"""Tuned constants, each with the measurement that chose it.

These were module-level constants scattered across `scripts/`. Several were
picked by experiment, and that reasoning previously lived only in comments next
to the number. Keeping them here with their provenance means a future change
has to argue with the evidence rather than just overwrite a literal.

Override at runtime with `sandhi.config.settings(...)`, or from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

MOON_RADIUS_M = 1737400.0


@dataclass(frozen=True)
class Config:
    # -- normalisation -----------------------------------------------------
    # Local contrast kernel. DEM-based photometric normalisation (Stage B) was
    # tested and failed its controls: noise pushed through it matched better
    # than real terrain. See BUGS.md BUG-011 and ROADMAP "Deliberately not
    # planned". This kernel is the whole of normalisation now.
    contrast_kernel: int = 31

    # -- matching ----------------------------------------------------------
    loftr_conf: float = 0.5
    # Tile the frame so every region gets its own matching attempt. Bucketed
    # SELECTION cannot raise coverage -- it only redistributes matches among
    # cells that already hold some. Coverage must be forced where matches are
    # produced.
    tiles: int = 8
    # A tight pad matters: an earlier 170 px search pad let tiles match ground
    # far from their own and gained nothing.
    tile_pad: int = 16
    # Below this LoFTR has too little context. A 506 px wide projected OHRC
    # swath would otherwise be cut into 63 px tiles and the match count
    # collapses (7 matches vs 1443).
    min_tile_px: int = 128
    # LoFTR coarse attention is O((H*W/64)^2); 2048^2 needs ~17 GB on CPU.
    max_window_px: int = 1024

    # -- sub-pixel refinement ---------------------------------------------
    refine_patch: int = 32        # half-width of the phase-correlation patch
    refine_max_shift: float = 3.0  # beyond this the patches disagree; drop the match

    # -- robust fit --------------------------------------------------------
    # Measured on 1024 windows, four sites:
    #   threshold  inliers  ratio   RMSE   coverage
    #      3.0        124     79%   1.299    0.52
    #      2.0        102     65%   0.959    0.47
    #      1.5         92     58%   0.786    0.45   <- sub-pixel AND spread
    #      1.0         66     42%   0.594    0.34
    # 3.0 admitted matches up to 3 px off, which is what held RMSE above 1 px.
    ransac_px: float = 1.5
    # A higher-DOF model always wins in-sample, so selection uses held-out
    # residual -- and even then homography won 2 of 3 nadir pairs by 5-9% noise,
    # where no perspective exists. Parsimony is the default; perspective must be
    # earned by this margin.
    complexity_margin: float = 0.15
    min_matches_for_split: int = 16

    # -- metrics -----------------------------------------------------------
    uniformity_grid: int = 8

    # -- control gate ------------------------------------------------------
    # Real imagery must out-match noise by at least this factor. Below it, the
    # method is keying on something both images share that is not the terrain.
    noise_reject_ratio: float = 5.0
    control_roll_px: int = 15
    min_matches: int = 8

    # -- weights -----------------------------------------------------------
    # None means the public MegaDepth `outdoor` weights, which have never seen
    # lunar imagery. A path here swaps in a fine-tune. Weights are NOT adopted
    # by default: a fine-tune has to beat the pretrained model on the control
    # gate, the obliquity ladder AND the illumination cases before this default
    # changes, so the comparison is always available by flipping one flag.
    weights: str | None = None


DEFAULT = Config()
_active = DEFAULT


def get() -> Config:
    return _active


def settings(**overrides) -> Config:
    """Replace the active config. Returns the new one."""
    global _active
    _active = replace(_active, **overrides)
    return _active


def reset() -> Config:
    global _active
    _active = DEFAULT
    return _active
