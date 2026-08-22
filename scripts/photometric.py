"""Stage B: render a DEM under a chosen sun geometry.

This is the project's central idea. Rather than hunting for illumination-
invariant descriptors, we render the reference terrain under the *source
image's* sun angles, so the matcher sees two images lit the same way.

Conventions, stated because they are the usual source of silent sign errors:

- `sun_azimuth_deg` is measured **clockwise from north**, where north is
  decreasing row index (row 0 is the northern edge of a north-up map product).
  So 0 = light from the north, 90 = from the east, 180 = from the south.
- `sun_elevation_deg` is the angle above the local horizon. Incidence angle,
  which is what PDS labels report, is `90 - elevation`.
- Elevations and pixel spacing are both in metres.

    python scripts/photometric.py          # analytic self-checks
"""

import argparse

import numpy as np

# Kaguya TC products are SIMPLE CYLINDRICAL. MAP_SCALE is the spacing at the
# equator; east-west spacing shrinks by cos(latitude) away from it. Ignoring
# this tilts every computed normal and quietly biases the shading.
def pixel_spacing(map_scale_m: float, latitude_deg: float) -> tuple[float, float]:
    """Return (east_west_m, north_south_m) ground spacing at a given latitude."""
    ew = map_scale_m * float(np.cos(np.radians(latitude_deg)))
    return ew, map_scale_m


def sun_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector pointing from the surface toward the sun, in (x_east, y_north, z_up)."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    return np.array(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], dtype=np.float64
    )


def surface_normals(dem: np.ndarray, spacing: tuple[float, float]) -> np.ndarray:
    """Per-pixel unit normals from an elevation grid. Returns (H, W, 3)."""
    ew, ns = spacing
    # np.gradient returns d/drow, d/dcol. Row increases southward, so the
    # north-facing derivative picks up a minus sign.
    dz_drow, dz_dcol = np.gradient(dem.astype(np.float64))
    dz_dx = dz_dcol / ew           # slope toward east
    dz_dy = -dz_drow / ns          # slope toward north

    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(dz_dx)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals


def cos_incidence(
    dem: np.ndarray, spacing: tuple[float, float], azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    """cos of the angle between each surface normal and the sun. Clipped at 0."""
    normals = surface_normals(dem, spacing)
    s = sun_vector(azimuth_deg, elevation_deg)
    return np.clip(normals @ s, 0.0, None)


def cast_shadows(
    dem: np.ndarray,
    spacing: tuple[float, float],
    azimuth_deg: float,
    elevation_deg: float,
    max_steps: int = 128,
) -> np.ndarray:
    """Boolean mask, True where terrain blocks the sun.

    Marches toward the sun in image space and asks whether any sample rises
    above the straight line from the origin pixel to the sun. Hard cast shadows
    are the dominant lunar signature -- shading alone will not reproduce them.
    """
    ew, ns = spacing
    az = np.radians(azimuth_deg)
    tan_el = np.tan(np.radians(elevation_deg))

    # Step one pixel at a time toward the sun.
    step_col = np.sin(az)
    step_row = -np.cos(az)
    step_ground = np.hypot(step_col * ew, step_row * ns)

    height, width = dem.shape
    rows, cols = np.mgrid[0:height, 0:width]
    shadowed = np.zeros(dem.shape, dtype=bool)
    origin = dem.astype(np.float64)

    for step in range(1, max_steps + 1):
        r = np.rint(rows + step_row * step).astype(int)
        c = np.rint(cols + step_col * step).astype(int)
        inside = (r >= 0) & (r < height) & (c >= 0) & (c < width)
        if not inside.any():
            break
        # Height the sun ray has reached above the origin pixel at this distance.
        ray = origin + step * step_ground * tan_el
        sampled = dem[np.clip(r, 0, height - 1), np.clip(c, 0, width - 1)]
        shadowed |= inside & (sampled > ray)

    return shadowed


def render(
    dem: np.ndarray,
    map_scale_m: float,
    latitude_deg: float,
    azimuth_deg: float,
    elevation_deg: float,
    model: str = "lommel_seeliger",
    shadows: bool = True,
) -> np.ndarray:
    """Render the DEM as it would appear under the given sun geometry.

    Returns reflectance in [0, 1]. `lommel_seeliger` suits the Moon better than
    plain Lambert at the low phase angles typical of mapping orbits.
    """
    spacing = pixel_spacing(map_scale_m, latitude_deg)
    mu0 = cos_incidence(dem, spacing, azimuth_deg, elevation_deg)

    if model == "lambert":
        refl = mu0
    elif model == "lommel_seeliger":
        # Nadir-ish viewing, so cos(emission) ~ 1.
        mu = 1.0
        refl = mu0 / (mu0 + mu)
        refl = refl / 0.5  # normalise: mu0=1 maps to 1.0
    else:
        raise ValueError(f"unknown model {model!r}")

    if shadows:
        refl = np.where(cast_shadows(dem, spacing, azimuth_deg, elevation_deg), 0.0, refl)

    return np.clip(refl, 0.0, 1.0)


def _slope_facing_east(height: int, width: int, spacing_m: float,
                       tilt_deg: float = 20.0) -> np.ndarray:
    """Plane whose outward normal points east.

    Careful: a surface that *rises* toward the east *faces* west. The normal is
    (-dz/dx, -dz/dy, 1), so to face east we need dz/dx < 0, i.e. elevation
    DECREASING with column index. Getting this backwards is the classic sign
    error here -- see BUGS.md BUG-003.
    """
    drop = spacing_m * np.tan(np.radians(tilt_deg))
    return np.tile(-np.arange(width, dtype=np.float64) * drop, (height, 1))


def _slope_facing_north(height: int, width: int, spacing_m: float,
                        tilt_deg: float = 20.0) -> np.ndarray:
    """Plane whose outward normal points north (elevation decreasing northward).

    Row 0 is north, so facing north means dz/dy_north < 0, i.e. elevation
    INCREASING with row index.
    """
    rise = spacing_m * np.tan(np.radians(tilt_deg))
    return np.tile((np.arange(height, dtype=np.float64) * rise)[:, None], (1, width))


def demo() -> int:
    """Analytic self-checks. These test the maths, not any dataset."""
    scale = 7.403  # metres/pixel, Kaguya TC map products
    spacing = (scale, scale)

    print("1. flat terrain -> cos(incidence) == sin(elevation)")
    flat = np.zeros((32, 32))
    for elevation in (10.0, 45.0, 80.0):
        mu0 = cos_incidence(flat, spacing, azimuth_deg=90.0, elevation_deg=elevation)
        expected = np.sin(np.radians(elevation))
        print(f"   el={elevation:4.1f}  got={mu0.mean():.6f}  expect={expected:.6f}")
        assert np.allclose(mu0, expected, atol=1e-9), "flat-surface shading is wrong"

    print("\n2. an east-facing slope is brightest under an eastern sun (azimuth 90)")
    east_facing = _slope_facing_east(32, 32, scale)
    lit_from_east = cos_incidence(east_facing, spacing, 90.0, 45.0).mean()
    lit_from_west = cos_incidence(east_facing, spacing, 270.0, 45.0).mean()
    print(f"   sun from east={lit_from_east:.4f}  from west={lit_from_west:.4f}")
    assert lit_from_east > lit_from_west, "east/west azimuth convention is inverted"

    print("\n3. a north-facing slope is brightest under a northern sun (azimuth 0)")
    north_facing = _slope_facing_north(32, 32, scale)
    from_north = cos_incidence(north_facing, spacing, 0.0, 45.0).mean()
    from_south = cos_incidence(north_facing, spacing, 180.0, 45.0).mean()
    print(f"   sun from north={from_north:.4f}  from south={from_south:.4f}")
    assert from_north > from_south, "north/south row convention is inverted"

    print("\n4. a wall casts its shadow away from the sun")
    step = np.zeros((32, 64))
    step[:, 32:] = 300.0  # tall cliff on the eastern half
    shadow_sun_east = cast_shadows(step, spacing, 90.0, 20.0)
    west_side = shadow_sun_east[:, 20:31].mean()
    east_side = shadow_sun_east[:, 40:60].mean()
    print(f"   sun from east: west-of-cliff shadowed={west_side:.2f} "
          f"east-of-cliff={east_side:.2f}")
    assert west_side > east_side, "shadows fall on the wrong side"

    print("\n5. lower sun casts longer shadows")
    long_shadow = cast_shadows(step, spacing, 90.0, 5.0).mean()
    short_shadow = cast_shadows(step, spacing, 90.0, 60.0).mean()
    print(f"   el=5 -> {long_shadow:.3f}   el=60 -> {short_shadow:.3f}")
    assert long_shadow > short_shadow, "shadow length does not track sun elevation"

    print("\n6. crater/dome inversion is reproduced (the effect Stage B defeats)")
    yy, xx = np.mgrid[-16:16, -16:16]
    bowl = (xx**2 + yy**2).astype(np.float64) * 0.5     # crater: rises outward
    dome = -bowl                                        # dome: falls outward
    crater_east = render(bowl, scale, 0.0, 90.0, 30.0, shadows=False)
    dome_west = render(dome, scale, 0.0, 270.0, 30.0, shadows=False)
    corr = float(np.corrcoef(crater_east.ravel(), dome_west.ravel())[0, 1])
    print(f"   corr(crater lit from east, dome lit from west) = {corr:+.3f}")
    assert corr > 0.9, "expected the classic crater/dome ambiguity to appear"

    print("\n7. latitude correction shrinks east-west spacing")
    ew_eq, _ = pixel_spacing(scale, 0.0)
    ew_60, _ = pixel_spacing(scale, 60.0)
    print(f"   equator={ew_eq:.4f} m   60deg={ew_60:.4f} m")
    assert abs(ew_60 - scale * 0.5) < 1e-6, "cos(latitude) correction is wrong"

    print("\nAll photometric self-checks passed.")
    return 0


def main() -> int:
    argparse.ArgumentParser().parse_args()
    return demo()


if __name__ == "__main__":
    raise SystemExit(main())
