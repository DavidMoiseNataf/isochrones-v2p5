#!/usr/bin/env python3
"""
download_mist_v25.py  --  Fetch MIST v2.5 (alpha-enhanced) data products and
lay them out in a directory structure analogous to MIST v1.2, so that the
`isochrones` package can read them with minimal code changes.

WHAT CHANGED vs. the first version
----------------------------------
MIST v2.5 EEP tarballs unpack to   feh_m025_afe_p2_vvcrit0.4/eeps/*.track.eep
i.e. an integer-coded directory name with the tracks one level down. v1.2 uses
MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_EEPS/*.track.eep  (decimal, tracks flat).

This script now normalizes v2.5 into the v1.2 convention:

    MIST_v2.5_feh_m0.25_afe_p0.2_vvcrit0.4_EEPS/*.track.eep   (tracks flat)

That exact name is what the package's own `get_file_basename` format string
produces for version="2.5" with a variable [a/Fe], so directory resolution and
the `glob(dir/*.track.eep)` loader keep working untouched.

Layout created (ROOT = $ISOCHRONES or ~/.isochrones):

    ROOT/mist/tracks/MIST_v2.5_feh_<f>_afe_<a>_vvcrit<v>_EEPS/*.track.eep
    ROOT/BC/mist/v2/feh<+f>_afe<+a>.<SYSTEM>                  # BC tables (already flat)

Common workflows
----------------
    # You already downloaded tracks in the OLD layout -> migrate in place,
    # no re-download (safe, idempotent):
    python download_mist_v25.py --reorganize-existing

    # Fresh download of new systems/vvcrit, written straight into v1.2 layout:
    python download_mist_v25.py --what tracks bc --bc-systems GALEX

    # The ~7 GB theoretical isochrone grid (needed by the isochrone interpolator):
    python download_mist_v25.py --what full_isos

    # Preview any action without writing:
    python download_mist_v25.py --reorganize-existing --dry-run
"""

import argparse
import json
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://mist.science"
EEP_BASE = f"{BASE}/data/tarballs_v2.5/eeps"
ISO_BASE = f"{BASE}/data/tarballs_v2.5/isos"
BC_BASE = f"{BASE}/BC_tables/v2"

VERSION = "2.5"

# Full [Fe/H] and [alpha/Fe] grids of MIST v2.5 (matches the mist.science listing).
FEH_ALL = [-4.0, -3.5, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5,
           -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
AFE_ALL = [-0.2, 0.0, 0.2, 0.4, 0.6]
MISSING = {(0.5, 0.6)}  # combos not published on the server

BC_SYSTEMS_ALL = [
    "CFHTugriz", "DECam", "Euclid", "GALEX", "HSC", "HST_ACS_HRC", "HST_ACS_SBC",
    "HST_ACS_WFC", "HST_WFC3", "HST_WFPC2", "IPHAS", "JWST", "NIRISS", "PanSTARRS",
    "RoboAO", "Roman", "LSST", "SDSSugriz", "SPITZER", "SPLUS", "SkyMapper", "Swift",
    "UBVRIplus", "UKIDSS", "UVIT", "VISTA", "WashDDOuvby", "WISE",
]
ISO_SYSTEMS_ALL = [
    "CFHTugriz", "DECam", "GALEX", "HST_ACSHR", "HST_ACSWF", "HST_WFC3", "HST_WFPC2",
    "JWST", "NIRISS", "LSST", "PanSTARRS", "SDSSugriz", "SkyMapper", "SPITZER", "SPLUS",
    "HSC", "IPHAS", "Swift", "UBVRIplus", "UKIDSS", "UVIT", "VISTA", "WashDDOuvby",
    "Roman", "WISE",
]


# ---- naming helpers --------------------------------------------------------

def feh_tag_int(feh):
    """Server filename token for [Fe/H], e.g. -0.25 -> 'm025'."""
    return f"{'m' if feh < 0 else 'p'}{int(round(abs(feh) * 100)):03d}"


def afe_tag_int(afe):
    """Server filename token for [a/Fe], e.g. +0.2 -> 'p2'."""
    return f"{'m' if afe < 0 else 'p'}{int(round(abs(afe) * 10)):d}"


def server_eep_dirname(feh, afe, vvcrit):
    """Directory name the v2.5 EEP tarball unpacks into (integer-coded)."""
    return f"feh_{feh_tag_int(feh)}_afe_{afe_tag_int(afe)}_vvcrit{vvcrit:.1f}"


def server_eep_tarbase(feh, afe, vvcrit):
    """Tarball base name on the server (sans .txz)."""
    return f"MIST_v{VERSION}_feh_{feh_tag_int(feh)}_afe_{afe_tag_int(afe)}_vvcrit{vvcrit:.1f}_EEPS"


def v12_style_dirname(feh, afe, vvcrit):
    """Target directory name in the v1.2 (decimal) convention.

    Reproduces exactly what isochrones' get_file_basename emits for
    version='2.5', e.g. (-0.25, +0.2, 0.4) -> MIST_v2.5_feh_m0.25_afe_p0.2_vvcrit0.4_EEPS
    """
    fs = "m" if feh < 0 else "p"
    as_ = "m" if afe < 0 else "p"
    return (f"MIST_v{VERSION}_feh_{fs}{abs(feh):.2f}"
            f"_afe_{as_}{abs(afe):.1f}_vvcrit{vvcrit:.1f}_EEPS")


# Parse an existing integer-coded dir back to (feh, afe, vvcrit).
_INT_DIR_RE = re.compile(r"^feh_([mp])(\d{3})_afe_([mp])(\d)_vvcrit([\d.]+)$")


def parse_int_dir(name):
    m = _INT_DIR_RE.match(name)
    if not m:
        return None
    feh = (1 if m.group(1) == "p" else -1) * int(m.group(2)) / 100.0
    afe = (1 if m.group(3) == "p" else -1) * int(m.group(4)) / 10.0
    return feh, afe, float(m.group(5))


# ---- misc ------------------------------------------------------------------

def isochrones_root():
    root = os.environ.get("ISOCHRONES")
    return Path(root).expanduser() if root else Path.home() / ".isochrones"


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def download(url, dest: Path, retries=5, dry_run=False):
    """Stream `url` to `dest` with HTTP-range resume and exponential backoff."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    [have] {dest.name} ({human(dest.stat().st_size)})")
        return dest
    if dry_run:
        print(f"    [GET ] {url}\n           -> {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "mist-v25-downloader"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = r.length + have if r.length is not None else None
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as f:
                    t0, last = time.time(), have
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - t0 > 1:
                            rate = (have - last) / (time.time() - t0)
                            tot = f" / {human(total)}" if total else ""
                            print(f"\r    {dest.name}: {human(have)}{tot} "
                                  f"({human(rate)}/s)   ", end="", flush=True)
                            t0, last = time.time(), have
            print()
            part.rename(dest)
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code == 404:
                print(f"    [404 ] not on server, skipping: {url}")
                return None
            wait = min(2 ** attempt, 30)
            print(f"\n    [warn] {e} (attempt {attempt}/{retries}); retry in {wait}s")
            time.sleep(wait)
    print(f"    [FAIL] giving up on {url}")
    return None


def is_valid_xz_tar(path: Path):
    try:
        with tarfile.open(path, "r:xz") as t:
            t.next()
        return True
    except Exception:
        return False


def extract(tarball: Path, into: Path, dry_run=False):
    if dry_run:
        print(f"    [tar ] would extract {tarball.name} -> {into}/")
        return True
    if not tarball or not tarball.exists():
        return False
    if not is_valid_xz_tar(tarball):
        print(f"    [bad ] corrupt tarball, deleting: {tarball.name}")
        tarball.unlink(missing_ok=True)
        return False
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:xz") as t:
        try:
            t.extractall(into, filter="data")
        except TypeError:
            t.extractall(into)
    print(f"    [ok  ] extracted {tarball.name}")
    return True


def reorganize(tracks_dir: Path, feh, afe, vvcrit, dry_run=False):
    """Move/flatten the server layout into the v1.2-analogous layout.

        tracks_dir/feh_<int>_afe_<int>_vvcrit<v>/[eeps/]*.track.eep
            ->  tracks_dir/MIST_v2.5_feh_<dec>_afe_<dec>_vvcrit<v>_EEPS/*.track.eep
    """
    src = tracks_dir / server_eep_dirname(feh, afe, vvcrit)
    target = tracks_dir / v12_style_dirname(feh, afe, vvcrit)

    # Idempotent: already migrated.
    if target.is_dir() and any(target.glob("*.track.eep")):
        print(f"    [skip] already in v1.2 layout: {target.name}")
        # If a stale integer dir somehow remains, clean it.
        if src.is_dir() and not dry_run:
            shutil.rmtree(src, ignore_errors=True)
        return True
    if not src.exists():
        print(f"    [miss] nothing to reorganize for {src.name}")
        return False

    eeps = sorted(src.rglob("*.track.eep"))
    if not eeps:
        print(f"    [warn] no *.track.eep under {src.name}")
        return False
    print(f"    [move] {src.name}  ->  {target.name}  ({len(eeps)} tracks)")
    if dry_run:
        return True
    target.mkdir(parents=True, exist_ok=True)
    for f in eeps:
        shutil.move(str(f), str(target / f.name))
    # Also carry over any per-feh HDF caches if present (none on first run).
    for h5 in src.rglob("*.h5"):
        shutil.move(str(h5), str(target / h5.name))
    shutil.rmtree(src, ignore_errors=True)
    return True


# ---- actions ---------------------------------------------------------------

def do_tracks(root, vvcrit, fehs, afes, keep, dry_run, manifest):
    tracks_dir = root / "mist" / "tracks"
    print(f"\n== EEP tracks (vvcrit={vvcrit}) -> {tracks_dir}")
    for feh in fehs:
        for afe in afes:
            if (round(feh, 2), round(afe, 2)) in MISSING:
                print(f"    [n/a ] feh={feh:+.2f} afe={afe:+.1f} not published; skipping")
                continue
            target = tracks_dir / v12_style_dirname(feh, afe, vvcrit)
            if target.is_dir() and any(target.glob("*.track.eep")):
                print(f"    [have] {target.name}")
                manifest["tracks"].append(target.name)
                continue
            base = server_eep_tarbase(feh, afe, vvcrit)
            tarball = tracks_dir / f"{base}.txz"
            got = download(f"{EEP_BASE}/{base}.txz", tarball, dry_run=dry_run)
            if got and extract(got, tracks_dir, dry_run=dry_run):
                reorganize(tracks_dir, feh, afe, vvcrit, dry_run=dry_run)
                manifest["tracks"].append(v12_style_dirname(feh, afe, vvcrit))
                if not keep and not dry_run and tarball.exists():
                    tarball.unlink()


def do_reorganize_existing(root, dry_run, manifest):
    """Migrate any already-extracted integer-coded dirs into v1.2 layout."""
    tracks_dir = root / "mist" / "tracks"
    if not tracks_dir.is_dir():
        print(f"    [miss] {tracks_dir} does not exist")
        return
    candidates = sorted(d for d in tracks_dir.iterdir()
                        if d.is_dir() and _INT_DIR_RE.match(d.name))
    print(f"\n== Reorganizing existing tracks -> v1.2 layout ({len(candidates)} dirs)")
    for d in candidates:
        feh, afe, vvcrit = parse_int_dir(d.name)
        if reorganize(tracks_dir, feh, afe, vvcrit, dry_run=dry_run):
            manifest["tracks"].append(v12_style_dirname(feh, afe, vvcrit))


def do_bc(root, systems, keep, dry_run, manifest):
    dest_dir = root / "BC" / "mist" / "v2"
    print(f"\n== Bolometric corrections (v2) -> {dest_dir}")
    for sysname in systems:
        tarball = dest_dir / f"{sysname}.txz"
        got = download(f"{BC_BASE}/{sysname}.txz", tarball, dry_run=dry_run)
        if got and extract(got, dest_dir, dry_run=dry_run):
            manifest["bc"].append(sysname)
            if not keep and not dry_run and tarball.exists():
                tarball.unlink()


def do_isos(root, systems, keep, dry_run, manifest):
    dest_dir = root / "mist" / "isos_v2.5"
    print(f"\n== Per-system synthetic isochrones -> {dest_dir}")
    for sysname in systems:
        tarball = dest_dir / f"{sysname}.txz"
        got = download(f"{ISO_BASE}/{sysname}.txz", tarball, dry_run=dry_run)
        if got and extract(got, dest_dir, dry_run=dry_run):
            manifest["isos"].append(sysname)
            if not keep and not dry_run and tarball.exists():
                tarball.unlink()


def do_full_isos(root, vvcrit, keep, dry_run, manifest):
    """Fetch the single large theoretical-isochrone grid (~7 GB).

    This is the file the isochrone interpolator (MIST_Isochrone /
    MISTIsochroneGrid) parses -- distinct from the per-system isos above. It
    lands in $ISOCHRONES/mist/ extracted as MIST_v2.5_vvcrit{v}_full_isos/,
    mirroring the v1.2 layout.
    """
    dest_dir = root / "mist"
    base = f"MIST_v2.5_vvcrit{vvcrit:.1f}_full_isos"
    extracted = dest_dir / base
    print(f"\n== FULL theoretical isochrones (~7 GB) -> {dest_dir}/{base}/")
    if extracted.is_dir() and any(extracted.iterdir()):
        print(f"    [skip] already extracted: {base}/")
        manifest["full_isos"].append(base)
        return
    tarball = dest_dir / f"{base}.txz"
    got = download(f"{ISO_BASE}/{base}.txz", tarball, dry_run=dry_run)
    if got and extract(got, dest_dir, dry_run=dry_run):
        # The v2.5 tarball lays the .iso files flat into mist/ rather than inside
        # a MIST_v2.5_..._full_isos/ wrapper. Normalize to the layout the
        # isochrone grid expects so nobody has to move files by hand.
        if not dry_run:
            loose = [p for p in dest_dir.glob("*_full.iso") if p.is_file()]
            if loose:
                extracted.mkdir(parents=True, exist_ok=True)
                for f in loose:
                    f.rename(extracted / f.name)
                print(f"    [tidy] moved {len(loose)} .iso files into {base}/")
        manifest["full_isos"].append(base)
        if not keep and not dry_run and tarball.exists():
            tarball.unlink()


def main():
    p = argparse.ArgumentParser(description="Download/organize MIST v2.5 data products.")
    p.add_argument("--reorganize-existing", action="store_true",
                   help="Migrate already-extracted integer-coded track dirs into the "
                        "v1.2-analogous layout, without downloading anything.")
    p.add_argument("--what", nargs="+", default=[],
                   choices=["tracks", "bc", "isos", "full_isos"],
                   help="Products to fetch (e.g. tracks bc full_isos). "
                        "'full_isos' is the ~7 GB theoretical isochrone grid used "
                        "by the isochrone interpolator. Omit if only reorganizing.")
    p.add_argument("--vvcrit", type=float, default=0.4, choices=[0.0, 0.4])
    p.add_argument("--feh", nargs="+", type=float, default=None,
                   help="Subset of [Fe/H] for tracks (default: all 17).")
    p.add_argument("--afe", nargs="+", type=float, default=None,
                   help="Subset of [alpha/Fe] for tracks (default: all 5).")
    p.add_argument("--bc-systems", nargs="+", default=["JWST"],
                   help="BC systems (default: JWST). Use 'ALL' for everything.")
    p.add_argument("--iso-systems", nargs="+", default=["JWST"],
                   help="Per-system iso sets (default: JWST). Use 'ALL' for everything.")
    p.add_argument("--keep-tarballs", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned actions and exit without writing.")
    args = p.parse_args()

    if not args.reorganize_existing and not args.what:
        p.error("nothing to do: pass --reorganize-existing and/or --what ...")

    root = isochrones_root()
    fehs = args.feh if args.feh is not None else FEH_ALL
    afes = args.afe if args.afe is not None else AFE_ALL
    bc = BC_SYSTEMS_ALL if args.bc_systems == ["ALL"] else args.bc_systems
    isos = ISO_SYSTEMS_ALL if args.iso_systems == ["ALL"] else args.iso_systems

    print(f"ISOCHRONES root : {root}")
    print(f"vvcrit          : {args.vvcrit}")
    if args.dry_run:
        print("** DRY RUN - nothing will be written **")

    manifest = {"version": VERSION, "vvcrit": args.vvcrit, "root": str(root),
                "layout": "v1.2-analogous",
                "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tracks": [], "bc": [], "isos": [], "full_isos": []}

    if args.reorganize_existing:
        do_reorganize_existing(root, args.dry_run, manifest)
    if "tracks" in args.what:
        do_tracks(root, args.vvcrit, fehs, afes, args.keep_tarballs, args.dry_run, manifest)
    if "bc" in args.what:
        do_bc(root, bc, args.keep_tarballs, args.dry_run, manifest)
    if "isos" in args.what:
        do_isos(root, isos, args.keep_tarballs, args.dry_run, manifest)
    if "full_isos" in args.what:
        do_full_isos(root, args.vvcrit, args.keep_tarballs, args.dry_run, manifest)

    if not args.dry_run:
        # de-duplicate while preserving order
        manifest["tracks"] = list(dict.fromkeys(manifest["tracks"]))
        mpath = root / f"manifest_mist_v2.5_vvcrit{args.vvcrit:.1f}.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest written: {mpath}")
        print("Done.")


if __name__ == "__main__":
    main()
