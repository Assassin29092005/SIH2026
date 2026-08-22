"""Resolve, inspect and download Kaguya/SELENE TC tile triples.

A "triple" is the same 3x3-degree lunar tile in three products that share one
grid and differ only in the product-id stem:

    TCO_MAPM04_<TILE>   morning  ortho image
    TCO_MAPE04_<TILE>   evening  ortho image
    DTM_MAP_02_<TILE>   DEM

Morning + evening give real cross-illumination pairs of identical terrain with
correspondence known from the shared map projection -- no invented warps. The
DEM is what Stage B renders to normalise illumination. All public, no login.

    python scripts/kaguya.py list --site equatorial
    python scripts/kaguya.py labels --site equatorial      # ~12 KB, validates the triple
    python scripts/kaguya.py fetch  --site equatorial      # ~864 MB per triple

Product ids come from ODE, not from reimplementing the tile-naming grid.
"""

import argparse
import re
import sys
from pathlib import Path

import requests

ODE = "https://oderest.rsl.wustl.edu/live2/"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "kaguya"

# product type -> (ODE pt, product-id stem that precedes the tile code)
KINDS = {
    "morning": ("TCMORM", "TCO_MAPM04_"),
    "evening": ("TCEVEM", "TCO_MAPE04_"),
    "dem": ("TCDTMM", "DTM_MAP_02_"),
}

SITES = {
    "equatorial": (9.0, 12.0, 18.0, 21.0),
    "ch3": (31.8, 32.9, -69.9, -68.9),
    "apollo15": (2.5, 4.5, 25.5, 26.5),
}


def ode_products(pt: str, bbox: tuple) -> list[dict]:
    """Fetch product records (with file lists) for a bbox."""
    west, east, south, north = bbox
    params = {
        "query": "product",
        "results": "fmp",
        "output": "JSON",
        "target": "moon",
        "ihid": "SLN",
        "iid": "TC",
        "pt": pt,
        "westernlon": west,
        "easternlon": east,
        "minlat": south,
        "maxlat": north,
    }
    resp = requests.get(ODE, params=params, timeout=120)
    resp.raise_for_status()
    result = resp.json().get("ODEResults", {})
    if result.get("Status") == "ERROR":
        raise RuntimeError(f"ODE error: {result.get('Error', '')[:200]}")
    products = result.get("Products", {}).get("Product", [])
    return [products] if isinstance(products, dict) else products


def tile_code(pdsid: str) -> str | None:
    """Strip the product-id stem to get the bare tile code, e.g. N21E009N18E012SC."""
    for _, stem in KINDS.values():
        if pdsid.upper().startswith(stem):
            return pdsid[len(stem):].upper()
    return None


def product_files(product: dict) -> list[dict]:
    files = product.get("Product_files", {}).get("Product_file", [])
    return [files] if isinstance(files, dict) else files


def pick_file(product: dict, suffix: str) -> dict | None:
    """Pick the data-directory file with the given suffix (skip browse copies)."""
    candidates = [
        f
        for f in product_files(product)
        if (f.get("FileName", "").upper().endswith(suffix.upper()))
        and "/browse/" not in (f.get("URL") or "")
    ]
    if not candidates:
        return None
    # If several, take the largest - browse thumbnails are tiny.
    return max(candidates, key=lambda f: int(f.get("KBytes", 0) or 0))


def find_triples(bbox: tuple) -> dict[str, dict]:
    """Return {tile_code: {kind: product}} for tiles complete in all three kinds."""
    by_kind: dict[str, dict[str, dict]] = {}
    for kind, (pt, _) in KINDS.items():
        by_kind[kind] = {}
        for product in ode_products(pt, bbox):
            code = tile_code(product.get("pdsid", ""))
            if code:
                by_kind[kind][code] = product

    common = set.intersection(*(set(v) for v in by_kind.values())) if by_kind else set()
    return {
        code: {kind: by_kind[kind][code] for kind in KINDS} for code in sorted(common)
    }


def true_size(url: str) -> int | None:
    """Authoritative byte size from the server, or None if it will not say.

    ODE's reported KBytes is approximate -- it under-reports the Kaguya .IMG
    files by ~2% -- so it must never be used to decide whether a download
    finished. See BUGS.md BUG-004.
    """
    try:
        resp = requests.head(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except requests.RequestException:
        return None


def download(url: str, dest: Path, expect_kb: int | None = None) -> Path:
    """Stream a file to disk, resuming a partial download if one exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0

    # Exact comparison against the server's own Content-Length. A file is only
    # complete when the byte counts match exactly; anything else resumes.
    expected = true_size(url)
    if expected is not None and have == expected:
        print(f"    have {dest.name} ({have/1e6:.1f} MB, verified)")
        return dest
    if expected is not None and have > expected:
        print(f"    {dest.name} is larger than expected - refetching")
        dest.unlink()
        have = 0
    if expected is None and expect_kb and have >= expect_kb * 1024:
        # No Content-Length available; ODE's estimate is a floor, not a target.
        print(f"    have {dest.name} ({have/1e6:.1f} MB, size UNVERIFIED)")
        return dest

    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
        if have and resp.status_code == 416:
            return dest  # already complete
        resp.raise_for_status()
        mode = "ab" if have and resp.status_code == 206 else "wb"
        if mode == "wb":
            have = 0
        total = int(resp.headers.get("Content-Length", 0)) + have

        done = have
        with open(dest, mode) as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r    {dest.name} {pct:5.1f}%  {done/1e6:7.1f} MB",
                          end="", flush=True)
        print()

    final = dest.stat().st_size
    if expected is not None and final != expected:
        raise IOError(
            f"{dest.name}: got {final} bytes, server said {expected}. Truncated."
        )
    return dest


def parse_pds_label(text: str) -> dict[str, str]:
    """Minimal PDS3 ODL key = value parser. Good enough for header inspection."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Z0-9_:]+)\s*=\s*(.+?)\s*$", line)
        if m:
            key, value = m.group(1), m.group(2).strip('"')
            out.setdefault(key, value)
    return out


# Header fields worth eyeballing before trusting a pair.
LABEL_KEYS = [
    "PRODUCT_ID",
    "MAP_PROJECTION_TYPE",
    "LINE_PROJECTION_OFFSET",
    "SAMPLE_PROJECTION_OFFSET",
    "MAP_RESOLUTION",
    "MAP_SCALE",
    "WESTERNMOST_LONGITUDE",
    "EASTERNMOST_LONGITUDE",
    "MINIMUM_LATITUDE",
    "MAXIMUM_LATITUDE",
    "LINES",
    "LINE_SAMPLES",
    "SAMPLE_BITS",
    "SAMPLE_TYPE",
]


def cmd_list(bbox: tuple) -> int:
    triples = find_triples(bbox)
    print(f"complete triples: {len(triples)}\n")
    for code, kinds in triples.items():
        img = pick_file(kinds["morning"], ".IMG")
        size = int(img.get("KBytes", 0) or 0) / 1e6 if img else 0
        print(f"  {code}   ~{size*3:.2f} GB for the triple")
    return 0 if triples else 1


def cmd_labels(bbox: tuple, limit: int) -> int:
    triples = find_triples(bbox)
    if not triples:
        print("no complete triples in this box")
        return 1

    for code, kinds in list(triples.items())[:limit]:
        print(f"\n=== tile {code} ===")
        parsed = {}
        for kind in KINDS:
            lbl = pick_file(kinds[kind], ".LBL")
            if not lbl:
                print(f"  {kind:<8} no .LBL found")
                continue
            dest = RAW / code / Path(lbl["FileName"]).name
            download(lbl["URL"], dest, int(lbl.get("KBytes", 0) or 0))
            # PDS3 labels are ASCII/UTF-8; be explicit (see BUGS.md BUG-001).
            parsed[kind] = parse_pds_label(dest.read_text(encoding="utf-8",
                                                          errors="replace"))

        if not parsed:
            continue
        width = max(len(k) for k in LABEL_KEYS) + 2
        print(f"\n  {'field'.ljust(width)}" + "".join(k.ljust(26) for k in parsed))
        for key in LABEL_KEYS:
            row = "".join(str(parsed[k].get(key, "-")).ljust(26) for k in parsed)
            print(f"  {key.ljust(width)}{row}")

        # The whole approach rests on these agreeing. Check, do not assume.
        grid_keys = ["MAP_PROJECTION_TYPE", "MAP_SCALE", "LINES", "LINE_SAMPLES",
                     "WESTERNMOST_LONGITUDE", "MINIMUM_LATITUDE"]
        mismatched = [
            k for k in grid_keys
            if len({str(p.get(k, "?")) for p in parsed.values()}) > 1
        ]
        print()
        if mismatched:
            print(f"  MISMATCH on {mismatched} - tiles are NOT co-registered.")
            return 1
        print("  grid agrees across morning/evening/DEM - co-registered.")
    return 0


def cmd_fetch(bbox: tuple, limit: int) -> int:
    triples = find_triples(bbox)
    if not triples:
        print("no complete triples in this box")
        return 1
    for code, kinds in list(triples.items())[:limit]:
        print(f"\n=== tile {code} ===")
        for kind in KINDS:
            for suffix in (".LBL", ".IMG"):
                f = pick_file(kinds[kind], suffix)
                if not f:
                    print(f"  {kind} {suffix}: missing")
                    continue
                dest = RAW / code / Path(f["FileName"]).name
                print(f"  {kind} {suffix}")
                download(f["URL"], dest, int(f.get("KBytes", 0) or 0))
    print(f"\nfiles under {RAW}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["list", "labels", "fetch"])
    parser.add_argument("--site", choices=sorted(SITES), default="equatorial")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "E", "S", "N"))
    parser.add_argument("--limit", type=int, default=1, help="tiles to process")
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else SITES[args.site]
    print(f"region: {bbox}")

    if args.command == "list":
        return cmd_list(bbox)
    if args.command == "labels":
        return cmd_labels(bbox, args.limit)
    return cmd_fetch(bbox, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
