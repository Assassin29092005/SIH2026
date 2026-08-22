"""Environment self-check for the SIH26166 pipeline.

Run this first on any new machine. It fails loudly if a dependency the project
actually relies on is missing or built without the planetary GDAL drivers.

    python scripts/check_env.py
"""

import sys

# Drivers we genuinely need. PDS4 reads Chandrayaan-2 (OHRC/TMC-2/IIRS);
# ISIS3 + PDS read LROC products. Everything else is convenience.
REQUIRED_DRIVERS = ["PDS4", "ISIS3", "PDS", "GTiff"]

# Present in most builds, useful, but we have fallbacks if absent.
OPTIONAL_DRIVERS = ["ISIS2", "VICAR", "JP2OpenJPEG", "HDF5", "HDF4"]

REQUIRED_MODULES = [
    "numpy",
    "scipy",
    "rasterio",
    "pyproj",
    "spiceypy",
    "cv2",
    "torch",
    "kornia",
]


def check_modules() -> list[str]:
    """Import each required module. Returns list of failures."""
    import importlib

    failures = []
    for name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"  {name:12s} OK    {version}")
        except Exception as exc:  # noqa: BLE001 - we want any import failure
            failures.append(name)
            print(f"  {name:12s} FAIL  {exc}")
    return failures


def check_drivers() -> list[str]:
    """Probe GDAL (via rasterio) for the planetary drivers. Returns missing required."""
    import rasterio

    with rasterio.Env() as env:
        available = {d.upper() for d in env.drivers()}

    print(f"  GDAL {rasterio.__gdal_version__} - {len(available)} drivers")

    missing = []
    for driver in REQUIRED_DRIVERS:
        ok = driver.upper() in available
        print(f"  {driver:12s} {'OK' if ok else 'MISSING (required)'}")
        if not ok:
            missing.append(driver)

    for driver in OPTIONAL_DRIVERS:
        ok = driver.upper() in available
        print(f"  {driver:12s} {'OK' if ok else 'absent (optional)'}")

    return missing


def check_sift() -> bool:
    """SIFT is a required baseline in the evaluation. Confirm it constructs."""
    import cv2

    try:
        cv2.SIFT_create()
        print("  SIFT         OK    (baseline available)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  SIFT         FAIL  {exc}")
        return False


def main() -> int:
    print(f"Python {sys.version.split()[0]}\n")

    print("Modules:")
    bad_modules = check_modules()

    print("\nGDAL drivers:")
    bad_drivers = check_drivers()

    print("\nBaselines:")
    sift_ok = check_sift()

    print()
    problems = []
    if bad_modules:
        problems.append(f"missing modules: {', '.join(bad_modules)}")
    if bad_drivers:
        problems.append(f"missing GDAL drivers: {', '.join(bad_drivers)}")
    if not sift_ok:
        problems.append("cv2.SIFT_create() failed")

    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        print("\nFix: pip install -r requirements.txt")
        print("If PDS4/ISIS3 drivers are still missing, the rasterio wheel was built")
        print("without them — install GDAL via conda-forge or use WSL instead.")
        return 1

    print("Environment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
