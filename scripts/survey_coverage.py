"""Survey what imagery is actually available for a region, before committing to it.

Answers two questions the project plan depends on:

1. Kaguya/SELENE TC -- does a complete (morning, evening, DEM) tile triple exist?
   These three share one 3x3-degree grid and differ only in the product-id stem,
   so a triple gives co-registered same-terrain imagery under opposite
   illumination PLUS the DEM needed for Stage-B photometric normalization.
   No login required.

2. LROC NAC -- how much multi-illumination coverage exists, binned by incidence
   angle? Incidence is our sun-elevation proxy for curriculum tier 3.

Chandrayaan-2 (OHRC/TMC-2/IIRS) is NOT indexed by ODE -- it is only available
through PRADAN, which requires a login. See docs note in CLAUDE.md.

    python scripts/survey_coverage.py --site ch3
    python scripts/survey_coverage.py --bbox 9 12 18 21
    python scripts/survey_coverage.py --self-check
"""

import argparse
import sys

import requests

ODE = "https://oderest.rsl.wustl.edu/live2/"

# ODE ignores unrecognised query parameters instead of erroring, so these exact
# spellings matter. Source: ODE REST V2.1.6 User's Manual. See BUGS.md BUG-002.
INC_MIN = "mininangle"
INC_MAX = "maxinangle"

# Named regions of interest. OHRC targeted the Chandrayaan-3 landing site, so
# that is where a CH-2/CH-3-relevant evaluation pair is most likely to exist.
SITES = {
    # name: (west_lon, east_lon, min_lat, max_lat)
    "ch3": (31.8, 32.9, -69.9, -68.9),      # Chandrayaan-3 landing site ~69.37S 32.32E
    "apollo15": (2.5, 4.5, 25.5, 26.5),
    "tycho": (348.0, 350.5, -43.5, -42.0),
    "equatorial": (9.0, 12.0, 18.0, 21.0),  # mid-latitude sanity region
}

INCIDENCE_BINS = [(0, 30), (30, 50), (50, 70), (70, 90)]


def ode_count(pt: str, bbox: tuple, ihid: str, iid: str, **extra) -> int | None:
    """Return ODE's product count for a bounding box, or None on error."""
    west, east, south, north = bbox
    params = {
        "query": "product",
        "results": "c",
        "output": "JSON",
        "target": "moon",
        "ihid": ihid,
        "iid": iid,
        "pt": pt,
        "westernlon": west,
        "easternlon": east,
        "minlat": south,
        "maxlat": north,
        **extra,
    }
    resp = requests.get(ODE, params=params, timeout=90)
    resp.raise_for_status()
    result = resp.json().get("ODEResults", {})
    if result.get("Status") == "ERROR":
        print(f"  ODE error: {result.get('Error', '')[:140]}")
        return None
    count = result.get("Count")
    return int(count) if count is not None else None


def survey_kaguya(bbox: tuple) -> dict:
    """Check morning / evening / DEM tile availability for the bbox."""
    print("Kaguya/SELENE TC (no login required):")
    counts = {}
    for pt, label in [("TCMORM", "morning"), ("TCEVEM", "evening"), ("TCDTMM", "DEM")]:
        n = ode_count(pt, bbox, ihid="SLN", iid="TC")
        counts[label] = n
        print(f"  {label:<8} {pt:<8} tiles: {n}")

    values = [v for v in counts.values() if v is not None]
    if values and len(set(values)) == 1 and values[0] > 0:
        print(f"  -> complete triples: {values[0]} tile(s). Usable for Stage B.")
    elif values and min(values) > 0:
        print(f"  -> PARTIAL: counts differ {counts}. Intersect by tile id before use.")
    else:
        print("  -> no coverage in this box.")
    return counts


def survey_nac(bbox: tuple) -> dict:
    """Histogram LROC NAC products by incidence angle."""
    print("\nLROC NAC (CDRNAC4), by incidence angle:")
    total = ode_count("CDRNAC4", bbox, ihid="LRO", iid="LROC")
    print(f"  unfiltered: {total}")

    hist = {}
    for lo, hi in INCIDENCE_BINS:
        n = ode_count(
            "CDRNAC4", bbox, ihid="LRO", iid="LROC", **{INC_MIN: lo, INC_MAX: hi}
        )
        hist[(lo, hi)] = n
        bar = "#" * min(40, (n or 0) // max(1, (total or 1) // 40 or 1))
        print(f"  {lo:>3}-{hi:<3} deg: {str(n):>6}  {bar}")

    populated = [b for b, n in hist.items() if (n or 0) >= 2]
    if len(populated) >= 2:
        print(f"  -> {len(populated)} populated illumination bins. Tier-3 pairs viable.")
    else:
        print("  -> too few illumination bins here for cross-illumination pairs.")
    return hist


def self_check() -> int:
    """Regression guard for BUG-002: confirm the incidence filter actually bites.

    ODE silently drops unknown parameters, so a filter that stops working looks
    exactly like no filter. This asserts the filtered count is strictly smaller.
    """
    bbox = SITES["ch3"]
    print("self-check: incidence filter must reduce the count (BUG-002 guard)")

    unfiltered = ode_count("CDRNAC4", bbox, ihid="LRO", iid="LROC")
    filtered = ode_count(
        "CDRNAC4", bbox, ihid="LRO", iid="LROC", **{INC_MIN: 30, INC_MAX: 50}
    )
    print(f"  unfiltered={unfiltered}  incidence30-50={filtered}")

    assert unfiltered is not None, "ODE returned no count for the unfiltered query"
    assert filtered is not None, "ODE returned no count for the filtered query"
    assert unfiltered > 0, "expected non-zero NAC coverage at the Chandrayaan-3 site"
    assert filtered < unfiltered, (
        f"incidence filter did not bite ({filtered} == {unfiltered}). "
        f"Parameter name '{INC_MIN}'/'{INC_MAX}' may have changed - see BUGS.md BUG-002."
    )

    print("  OK - filter is applied.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=sorted(SITES), help="named region")
    parser.add_argument(
        "--bbox", nargs=4, type=float, metavar=("W", "E", "S", "N"), help="custom bbox"
    )
    parser.add_argument("--self-check", action="store_true", help="run BUG-002 guard")
    args = parser.parse_args()

    if args.self_check:
        return self_check()

    if args.bbox:
        bbox = tuple(args.bbox)
        label = f"bbox {bbox}"
    else:
        site = args.site or "ch3"
        bbox = SITES[site]
        label = f"{site} {bbox}"

    print(f"Region: {label}\n")
    survey_kaguya(bbox)
    survey_nac(bbox)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
