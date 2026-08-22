"""List what the PDS ODE REST API actually indexes for the Moon.

ODE (Orbital Data Explorer, Washington University) is a public, key-free REST
API over the PDS Geosciences archive. We use it to discover which product types
are queryable, and crucially whether their footprints and incidence angles are
indexed -- incidence angle is the sun-elevation proxy we need to build
multi-illumination training pairs.

    python scripts/ode_catalog.py            # LROC only
    python scripts/ode_catalog.py --all      # every Moon instrument
"""

import argparse
import json
import sys
from pathlib import Path

import requests

ODE = "https://oderest.rsl.wustl.edu/live2/"
CACHE = Path(__file__).resolve().parent.parent / "data" / "interim" / "iipt.json"


def fetch_catalog(force: bool = False) -> dict:
    """Fetch (and cache) the instrument/product-type catalog for the Moon."""
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text(encoding="utf-8"))

    resp = requests.get(
        ODE, params={"query": "iipy", "output": "JSON", "target": "moon"}, timeout=90
    )
    resp.raise_for_status()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(resp.text, encoding="utf-8")
    return resp.json()


def iter_product_types(catalog: dict):
    """Yield each product-type record, regardless of ODE's nesting depth."""
    def walk(node):
        if isinstance(node, dict):
            if "PT" in node and "IID" in node:
                yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    yield from walk(catalog)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="show every instrument")
    parser.add_argument("--force", action="store_true", help="ignore cache")
    parser.add_argument("--grep", default=None, help="filter on instrument/product text")
    args = parser.parse_args()

    records = list(iter_product_types(fetch_catalog(force=args.force)))
    if not records:
        print("FAIL: no product types parsed - ODE response shape changed?")
        return 1

    instruments = sorted({r.get("IID", "?") for r in records})
    print(f"{len(records)} product types across {len(instruments)} instruments\n")
    print("Instruments:", ", ".join(instruments), "\n")

    rows = records
    if not args.all:
        rows = [r for r in rows if r.get("IID", "").upper() == "LROC"]
    if args.grep:
        needle = args.grep.lower()
        rows = [r for r in rows if needle in json.dumps(r).lower()]

    header = f"{'IID':<10} {'PT':<14} {'#products':>10}  {'fp':<3} {'inc':<4} name"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: -int(x.get("NumberProducts", 0) or 0)):
        print(
            f"{r.get('IID','?'):<10} {r.get('PT','?'):<14} "
            f"{int(r.get('NumberProducts', 0) or 0):>10}  "
            f"{r.get('ValidFootprints','?'):<3} {r.get('ValidIncidenceAngles','?'):<4} "
            f"{r.get('PTName','')[:52]}"
        )

    print("\nfp = footprints indexed, inc = incidence angle indexed.")
    print("We need fp=T AND inc=T to query multi-illumination pairs by geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
