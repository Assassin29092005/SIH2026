"""Find and fetch Chandrayaan-1 M3 products from PDS ODE. Public, no login.

WHY M3
------
IIRS does not register against Kaguya (see `iirs_vs_kaguya.py`): correlation
+0.033, and three controls rule out the projection, the strip shape and the
11.49x scale ratio. What is left is the instrument pairing -- an imaging
spectrometer against a visible framing camera.

M3 is the test of that explanation. It is also an imaging spectrometer, at
~140 m/px against IIRS's 85.08 m/px, so IIRS-to-M3 is spectrometer-to-
spectrometer at **1.6x** rather than spectrometer-to-camera at 11.5x. If the
instrument-pairing explanation is right, this pairing should work; if it fails
too, the explanation is wrong and the problem is IIRS itself.

Unlike Chandrayaan-2, M3 **is indexed by ODE** and needs no PRADAN account.

TWO PRODUCT TYPES ARE NEEDED, FROM THE SAME OBSERVATION
------------------------------------------------------
`REFIMG` (L2, V01) carries the reflectance cube but no geolocation backplane.
`CALIV3` (L1B, V03) carries `LOC` -- per-pixel latitude, longitude and
elevation. They share an observation id, so the V03 LOC georeferences the V01
reflectance. Fetching only the cube leaves it unlocatable.

    python scripts/m3.py list --bbox 9.8 11.8 9.2 41.7
    python scripts/m3.py fetch --id M3G20090204T233457
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "m3"
BASE = "https://oderest.rsl.wustl.edu/live2/"

# BUG-002 and again on 2026-09-10: ODE silently returns nothing useful for an
# unknown parameter VALUE, not just an unknown parameter name. M3 lives under
# ihid "CH1-ORB"; querying "CH1" returns a result object with no Count and no
# products, which looks exactly like "no coverage here".
IHID, IID = "CH1-ORB", "M3"

# The files worth having, by product type. Cubes are large; backplanes are not.
WANTED = {
    "REFIMG": ("RFL.IMG", "RFL.HDR", "L2.LBL"),
    "CALIV3": ("LOC.IMG", "LOC.HDR", "OBS.IMG", "OBS.HDR", "L1B.LBL"),
}

# An observation id maps to a different product id per type: the L2 reflectance
# is "<obs>_V01_RFL", the L1B calibrated set "<obs>_V03_RDN". Filtering the
# unbounded product list client-side does not work -- ODE caps the response, so
# the target simply is not in the first page.
PID_SUFFIX = {"REFIMG": "_V01_RFL", "CALIV3": "_V03_RDN"}


def query(pt: str, bbox: tuple | None = None, pid: str | None = None) -> list[dict]:
    p = {"query": "product", "results": "fmp", "output": "JSON", "target": "moon",
         "ihid": IHID, "iid": IID, "pt": pt, "limit": 100}
    if bbox:
        w, e, s, n = bbox
        p |= {"westernlon": w, "easternlon": e, "minlat": s, "maxlat": n}
    if pid:
        p["pdsid"] = pid
    d = requests.get(BASE, params=p, timeout=180).json()["ODEResults"]
    prods = d.get("Products")
    # ODE returns a bare string here when nothing matches, not an empty list.
    if not isinstance(prods, dict):
        return []
    prods = prods.get("Product", [])
    return [prods] if isinstance(prods, dict) else prods


def files_of(prod: dict) -> list[dict]:
    fs = prod.get("Product_files", {}).get("Product_file", [])
    return [fs] if isinstance(fs, dict) else fs


def cmd_list(bbox: tuple, limit: int) -> int:
    prods = query("REFIMG", bbox=bbox)
    w0, e0, s0, n0 = bbox
    rows = []
    for pr in prods:
        w, e = float(pr["Westernmost_longitude"]), float(pr["Easternmost_longitude"])
        s, n = float(pr["Minimum_latitude"]), float(pr["Maximum_latitude"])
        ow, oh = min(e, e0) - max(w, w0), min(n, n0) - max(s, s0)
        if ow <= 0 or oh <= 0:
            continue
        mb = sum(int(f.get("KBytes", 0)) for f in files_of(pr)
                 if f.get("FileName", "").upper().endswith("RFL.IMG")) / 1024
        rows.append((ow * oh, pr["pdsid"], w, e, s, n, mb,
                     pr.get("UTC_start_time")))
    rows.sort(reverse=True)
    print(f"{len(rows)} M3 reflectance products overlap {bbox}\n")
    for area, pid, w, e, s, n, mb, t in rows[:limit]:
        print(f"  {pid}")
        print(f"    overlap {area:.2f} deg^2   RFL {mb:.0f} MB   {t}")
        print(f"    footprint lon {w:.2f}..{e:.2f}  lat {s:.2f}..{n:.2f}")
    if rows:
        print(f"\n  best size/overlap tradeoff is usually NOT the largest overlap;")
        print(f"  M3 strips run up to 90 deg of latitude, so area tracks length.")
    return 0


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    total = int(head.headers.get("Content-Length", 0))
    # Resume only against the SERVER's length, never a catalogue figure --
    # PRADAN's under-reported sizes made that mistake once already (BUG-004).
    if dest.exists() and total and dest.stat().st_size == total:
        print(f"    have {dest.name} ({total/1e6:.0f} MB, verified)")
        return
    print(f"    {dest.name} ({total/1e6:.0f} MB)")
    got = 0
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r      {100*got/total:5.1f}%  {got/1e6:7.0f} MB",
                          end="", flush=True)
    print()
    if total and got != total:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"truncated: got {got} of {total} bytes")


def cmd_fetch(obs_id: str) -> int:
    """Fetch the reflectance cube and the LOC/OBS backplanes for one observation."""
    for pt, suffixes in WANTED.items():
        prods = query(pt, pid=obs_id + PID_SUFFIX[pt])
        if not prods:
            print(f"  {pt}: no product matching {obs_id}")
            continue
        pr = prods[0]
        print(f"  {pt}: {pr['pdsid']}")
        for f in files_of(pr):
            name = f.get("FileName", "")
            if name.upper().endswith(suffixes):
                download(f["URL"], RAW / obs_id / name)
    print(f"\nfiles under {RAW / obs_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["list", "fetch"])
    ap.add_argument("--bbox", nargs=4, type=float, default=[9.8, 11.8, 9.2, 41.7],
                    metavar=("W", "E", "S", "N"))
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--id", help="observation id, e.g. M3G20090204T233457")
    args = ap.parse_args()

    if args.command == "list":
        return cmd_list(tuple(args.bbox), args.limit)
    if not args.id:
        print("fetch needs --id (run `list` first)")
        return 1
    return cmd_fetch(args.id)


if __name__ == "__main__":
    raise SystemExit(main())
