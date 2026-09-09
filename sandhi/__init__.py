"""SANDHI — Sun-Angle aNd scale-invariant Detection of Homologous Imagery.

Registers a Chandrayaan-2 optical image to a controlled lunar reference frame
and returns sub-pixel match points plus the registered product.

    from sandhi import register, controls
    result = register(src, ref, gsd_m=7.403)
    print(result["metrics"]["rmse_px"])

Every reported number is expected to pass `sandhi.controls.run` first. See
CLAUDE.md for the conventions and BUGS.md for why that gate exists.
"""

from . import config, controls, matching, metrics, models, normalize, outputs, refine
from .pipeline import decimate, register, to_common_gsd, warp

__version__ = "0.1.0"

__all__ = [
    "config", "controls", "matching", "metrics", "models", "normalize",
    "outputs", "refine",
    "register", "warp", "decimate", "to_common_gsd",
]
