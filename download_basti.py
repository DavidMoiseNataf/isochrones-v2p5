#!/usr/bin/env python3
"""
download_basti.py  --  Fetch BaSTI-IAC precomputed isochrones and lay them out
in a directory structure analogous to the MIST v2.5 products, so the
`isochrones.basti` extension can read them with minimal code, and the readme /
tutorial commands parallel download_mist_v25.py.

Physics case: overshooting + diffusion + eta = 0.3, Y_BBN = 0.247 ("O1D1E1"),
served at  http://basti-iac.oa-abruzzo.inaf.it/PREISOCS/<AFE>O1D1E1Y247/
as tarballs  isocz<zy><afe>o1d1e1.isc_<system>.tar.gz
(one tarball = all ages for one composition + photometric system).

Layout created (ROOT = $ISOCHRONES or ~/.isochrones), parallel to
ROOT/mist/MIST_v2.5_vvcrit0.4_full_isos/ :

    ROOT/basti/BaSTI_O1D1E1_isos/<age>z<Z>y<Y><AFE>O1D1E1.isc_<system>

i.e. ONE flat directory for all [a/Fe], all compositions, all systems --
BaSTI filenames encode the alpha tag, so nothing collides. A manifest is
written to ROOT/manifest_basti_O1D1E1.json (parallel to the MIST manifest).

BaSTI-specific necessity: the composition (Z, Y) strings per [Fe/H] node have
CHANGED between BaSTI releases, so the authoritative file list comes from the
live server directory listing (--scrape, recommended), with a 2023-era seed
manifest as offline fallback.

Common workflows (parallel to download_mist_v25.py)
----------------------------------------------------
    # See what the server has and check every URL, downloading nothing:
    python download_basti.py --scrape --probe

    # Fresh download, everything (all 3 alphas, 4 default systems):
    python download_basti.py --what isos --scrape

    # Subsets, MIST-style flags:
    python download_basti.py --what isos --scrape --afe 0.0 0.4 \\
        --iso-systems JWST WFC3

    # Migrate files from the earlier per-alpha layout (basti/raw/<TAG>O1D1E1/),
    # no re-download (safe, idempotent):
    python download_basti.py --reorganize-existing

    # Preview any action without writing:
    python download_basti.py --what isos --scrape --dry-run
"""

import argparse
import json
import os
import re
import shutil
import tarfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

BASE = "http://basti-iac.oa-abruzzo.inaf.it/PREISOCS"
CASE = "O1D1E1"
CASE_LOWER = "o1d1e1"
YDIR = "Y247"

AFE_ALL = [-0.2, 0.0, 0.4]
AFE_TAGS = {-0.2: "M02", 0.0: "P00", 0.4: "P04"}
AFE_LOWER = {-0.2: "m02", 0.0: "p00", 0.4: "p04"}
TAG_TO_LOWER = {AFE_TAGS[a]: AFE_LOWER[a] for a in AFE_ALL}

# Short system names (MIST-style tokens) -> BaSTI .isc extensions.
# The first four are VERIFIED against downloaded data. The rest are candidate
# extension strings following BaSTI's naming style and are UNVERIFIED --
# confirm each with:   python download_basti.py --probe --iso-systems 2MASS ...
# A [MISSING] result means the guess is wrong: download one file for that
# system from the BaSTI web interface, read its extension, and correct the
# entry here (please also update PHOT_SYSTEMS in isochrones/basti/models.py).
ISO_SYSTEMS = {
    # verified
    "JWST": "isc_jwst-nircam_zp_vega-sirius",
    "WFC3": "isc_wfc3",
    "ACS": "isc_acs",
    "GAIA": "isc_gaia-dr3",
    # tarball URLs probed OK 2026-07-11
    "2MASS": "isc_2mass",
    "DECAM": "isc_decam",
    "EUCLID": "isc_euclid",
    "GALEX": "isc_galex",
    "HAWKI": "isc_hawki",
    "TESS": "isc_tess",
    "VISTA": "isc_vista",
    "WISE": "isc_wise",
    # extensions read from real BaSTI files 2026-07-11 (note 'panstrss1' is
    # BaSTI's own spelling); tarball URLs follow the standard pattern --
    # confirm once with --probe before the bulk download
    "JC": "isc_john",              # Johnson-Cousins
    "PANSTARRS": "isc_panstrss1",
    "SKYMAPPER": "isc_skym",
    # Roman deliberately EXCLUDED (2026-07-11): the server carries three
    # product generations with inconsistent tarball coverage per alpha
    # (isc_roman_vega P04-only, isc_roman_ab P00+P04, isc_wfirst all three
    # but unverified content), even though the per-file web interface serves
    # current isc_roman_vega files for all alphas. Revisit if BaSTI posts
    # uniform roman_vega tarballs.
}
ISO_SYSTEMS_ALL = list(ISO_SYSTEMS)

# NB: extracted MEMBER extensions can differ from the tarball name -- the
# gaia-dr3 tarballs extract files ending '.isc_gaia-dr3-new'. The grid code
# handles known aliases; if a newly downloaded system's members carry an
# unexpected extension, add it to SYSTEM_ALIASES in isochrones/basti/models.py.

# ---------------------------------------------------------------------------
# SEED manifest (2023-era zy strings per nominal [Fe/H]); offline fallback
# only -- known partially stale (e.g. a current P04 file has z=0.00392, not in
# this list). --scrape supersedes it.
# ---------------------------------------------------------------------------
SEED_MANIFEST = {
    "M02": ["105y247", "405y247", "705y247", "154y247", "204y247", "304y247",
            "474y248", "604y248", "704y248", "103y248", "153y249", "203y250",
            "303y251", "503y253", "603y255", "703y257", "102y260", "132y264",
            "162y268", "222y276", "302y287"],
    "P00": ["105y247", "505y247", "104y247", "204y247", "304y247", "444y248",
            "604y248", "804y248", "103y248", "143y249", "203y250", "303y251",
            "403y252", "603y255", "803y257", "102y260", "132y264", "172y269",
            "202y274", "302y284"],
    "P04": ["205y247", "104y247", "204y247", "404y248", "604y248", "904y248",
            "103y249", "163y249", "203y250", "303y251", "403y252", "603y255",
            "803y257", "122y263", "152y267", "192y272", "242y279", "332y290"],
}


def isochrones_root():
    root = os.environ.get("ISOCHRONES")
    return Path(root) if root else Path.home() / ".isochrones"


def isos_dir(root):
    return root / "basti" / "BaSTI_{}_isos".format(CASE)


# ---------------------------------------------------------------------------
# server interaction
# ---------------------------------------------------------------------------

class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.hrefs.append(v)


def alpha_dir_url(tag):
    return "{}/{}{}{}".format(BASE, tag, CASE, YDIR)


def tarball_name(tag, zy, system_ext):
    return "isocz{}{}{}.{}.tar.gz".format(zy, TAG_TO_LOWER[tag], CASE_LOWER, system_ext)


def tarball_url(tag, zy, system_ext):
    return "{}/{}".format(alpha_dir_url(tag), tarball_name(tag, zy, system_ext))


def scrape_manifest(tags, system_exts):
    """{tag: {zy: set(system_ext)}} from live server listings; seed fallback."""
    # NB: assembled by concatenation, NOT str.format -- the {3} quantifiers
    # in the regex would be parsed as positional format fields.
    pat = re.compile(
        r"isocz([0-9]{3}y[0-9]{3})(m02|p00|p04)"
        + re.escape(CASE_LOWER)
        + r"\.(isc_[A-Za-z0-9_\-]+)\.tar\.gz$"
    )
    manifest = {}
    for tag in tags:
        url = alpha_dir_url(tag) + "/"
        print("Scraping {} ...".format(url))
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                html = r.read().decode("utf-8", "replace")
        except Exception as e:
            print("  !! listing failed ({}); falling back to SEED manifest "
                  "for {}".format(e, tag))
            manifest[tag] = {zy: set(system_exts) for zy in SEED_MANIFEST[tag]}
            continue
        p = _LinkParser()
        p.feed(html)
        found = {}
        for href in p.hrefs:
            m = pat.search(os.path.basename(href))
            if m:
                zy, _, ext = m.groups()
                if ext in system_exts:
                    found.setdefault(zy, set()).add(ext)
        print("  {} compositions, {} tarballs matching requested systems".format(
            len(found), sum(len(v) for v in found.values())))
        manifest[tag] = found
    return manifest


def probe(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code in (403, 405, 501):  # server rejects HEAD; try 1-byte GET
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.status in (200, 206)
            except Exception:
                return False
        return False
    except Exception:
        return False


def do_guess_ext(tags, candidates):
    """Probe candidate isc_* extensions against ONE composition per alpha.

    Used to discover tarball naming when it differs between alpha
    directories (e.g. Roman: 'isc_roman_vega' exists only under P04).
    """
    print("\n== Extension guessing (one composition per alpha)")
    for tag in tags:
        zy = SEED_MANIFEST[tag][0]
        print("  {} (composition {}):".format(tag, zy))
        for cand in candidates:
            ext = cand if cand.startswith("isc_") else "isc_" + cand
            url = tarball_url(tag, zy, ext)
            print("    [{}] {}".format("ok" if probe(url) else "--", ext))


def fetch(url, dest):
    tmp = str(dest) + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, str(dest))


# ---------------------------------------------------------------------------
# resume detection via the manifest. Filename-based detection is unreliable:
# the tarball's zy string encodes a NOMINAL Z that can differ from the exact
# Z in the extracted filenames (e.g. tarball 403y252 -> files z0039200y252),
# and member names may differ in case from web-interface downloads. The
# manifest written at the end of each run lists every successfully extracted
# tarball, so re-runs skip exactly those.
# ---------------------------------------------------------------------------

def load_previous_isos(root):
    mpath = root / "manifest_basti_{}.json".format(CASE)
    if mpath.exists():
        try:
            with open(mpath) as f:
                return set(json.load(f).get("isos", []))
        except Exception:
            return set()
    return set()


def extract_isc(tarball, dest, dry_run=False):
    n = 0
    with tarfile.open(tarball) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if ".isc" not in name:
                continue
            out = dest / name
            n += 1
            if dry_run or out.exists():
                continue
            with tar.extractfile(member) as src, open(out, "wb") as dst:
                dst.write(src.read())
    return n


# ---------------------------------------------------------------------------
# actions (do_* naming, mirroring download_mist_v25.py)
# ---------------------------------------------------------------------------

def do_reorganize_existing(root, dry_run, manifest):
    """Migrate basti/raw/<TAG>O1D1E1/ (earlier layout) -> flat isos dir."""
    dest = isos_dir(root)
    print("\n== Reorganize existing -> {}".format(dest))
    moved = 0
    for tag in AFE_TAGS.values():
        src = root / "basti" / "raw" / (tag + CASE)
        if not src.is_dir():
            continue
        files = [f for f in src.iterdir() if ".isc_" in f.name]
        if not files:
            continue
        print("    [move] {} files from {}".format(len(files), src))
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                target = dest / f.name
                if not target.exists():
                    shutil.move(str(f), str(target))
                    moved += 1
    print("    {} files migrated.".format(moved))
    manifest["reorganized"] = moved


def do_isos(root, tags, system_exts, use_scrape, do_probe, keep_tarballs,
            dry_run, manifest):
    dest = isos_dir(root)
    print("\n== BaSTI {} isochrones -> {}".format(CASE, dest))

    if use_scrape:
        listing = scrape_manifest(tags, system_exts)
    else:
        print("Using SEED manifest (2023-era zy strings) -- prefer --scrape.")
        listing = {t: {zy: set(system_exts) for zy in SEED_MANIFEST[t]}
                   for t in tags}
    manifest["compositions"] = {t: sorted(listing[t]) for t in listing}

    jobs = []
    for tag in tags:
        for zy, exts in sorted(listing[tag].items()):
            for ext in sorted(set(exts) & set(system_exts)):
                jobs.append((tag, zy, ext, tarball_url(tag, zy, ext)))
    print("{} tarballs in work list.".format(len(jobs)))

    if do_probe:
        n_ok = 0
        for tag, zy, ext, url in jobs:
            ok = probe(url)
            n_ok += ok
            print("  [{}] {}".format("ok" if ok else "MISSING", url))
        print("{}/{} present.".format(n_ok, len(jobs)))
        return

    tb_dir = root / "basti" / "tarballs"
    already = load_previous_isos(root)
    n_new = n_have = n_fail = 0
    for tag, zy, ext, url in jobs:
        if tarball_name(tag, zy, ext) in already:
            print("    [have] {}".format(tarball_name(tag, zy, ext)))
            n_have += 1
            manifest["isos"].append(tarball_name(tag, zy, ext))
            continue
        if dry_run:
            print("    [get ] {}".format(url))
            n_new += 1
            continue
        dest.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb = tb_dir / tarball_name(tag, zy, ext)
        try:
            if not tb.exists():
                print("    [get ] {}".format(url))
                fetch(url, tb)
            n = extract_isc(tb, dest)
            print("    [ok  ] {}  ({} isochrone files)".format(tb.name, n))
            n_new += 1
            manifest["isos"].append(tb.name)
            if not keep_tarballs:
                tb.unlink()
        except (urllib.error.URLError, tarfile.TarError, EOFError, OSError) as e:
            print("    [FAIL] {}  ({})".format(url, e))
            if tb.exists():
                tb.unlink()
            n_fail += 1
    print("    {} fetched, {} already present, {} failed.".format(n_new, n_have, n_fail))
    manifest["failed"] = n_fail


def main():
    p = argparse.ArgumentParser(description="Download/organize BaSTI-IAC O1D1E1 isochrones.")
    p.add_argument("--reorganize-existing", action="store_true",
                   help="Migrate files from the earlier per-alpha layout "
                        "(basti/raw/<TAG>O1D1E1/) into the flat isos dir, "
                        "without downloading anything.")
    p.add_argument("--what", nargs="+", default=[], choices=["isos"],
                   help="Products to fetch. BaSTI currently offers one kind: "
                        "'isos' (precomputed isochrones with magnitudes; the "
                        "analogue of MIST's full_isos + per-system isos in one).")
    p.add_argument("--afe", nargs="+", type=float, default=None,
                   choices=AFE_ALL,
                   help="Subset of [alpha/Fe] (default: all 3).")
    p.add_argument("--iso-systems", nargs="+", default=["ALL"],
                   help="Photometric systems: JWST WFC3 ACS GAIA, a raw "
                        "isc_* extension, or 'ALL' (default).")
    p.add_argument("--scrape", action="store_true",
                   help="Build the file list from the live server directory "
                        "listing (recommended; the built-in seed list is "
                        "2023-era and partially stale).")
    p.add_argument("--probe", action="store_true",
                   help="HEAD-check every URL in the work list and report; "
                        "download nothing.")
    p.add_argument("--guess-ext", nargs="+", default=None,
                   help="Probe candidate isc_* extensions against one "
                        "composition per alpha, then exit. For discovering "
                        "tarball names that differ between alpha directories "
                        "(e.g. Roman).")
    p.add_argument("--keep-tarballs", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned actions and exit without writing.")
    args = p.parse_args()

    if not args.reorganize_existing and not args.what and not args.probe \
            and not args.guess_ext:
        args.what = ["isos"]   # single-product downloader: default to it

    root = isochrones_root()
    afes = args.afe if args.afe is not None else AFE_ALL
    tags = [AFE_TAGS[a] for a in afes]
    if args.iso_systems == ["ALL"]:
        exts = [ISO_SYSTEMS[s] for s in ISO_SYSTEMS_ALL]
    else:
        exts = [ISO_SYSTEMS.get(s.upper(), s if s.startswith("isc_") else "isc_" + s)
                for s in args.iso_systems]

    print("ISOCHRONES root : {}".format(root))
    print("case            : {} (overshooting, diffusion, eta=0.3, Y=0.247)".format(CASE))
    print("[alpha/Fe]      : {}".format(afes))
    print("systems         : {}".format(exts))
    if args.dry_run:
        print("** DRY RUN - nothing will be written **")

    manifest = {"case": CASE, "root": str(root), "layout": "MIST-v2.5-analogous flat",
                "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "afe": afes, "systems": exts, "isos": []}

    if args.guess_ext:
        do_guess_ext(tags, args.guess_ext)
        return

    if args.reorganize_existing:
        do_reorganize_existing(root, args.dry_run, manifest)
    if "isos" in args.what or args.probe:
        do_isos(root, tags, exts, args.scrape, args.probe,
                args.keep_tarballs, args.dry_run, manifest)

    if not args.dry_run and not args.probe:
        manifest["isos"] = list(dict.fromkeys(manifest["isos"]))
        mpath = root / "manifest_basti_{}.json".format(CASE)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        print("\nManifest written: {}".format(mpath))
        print("Done.")


if __name__ == "__main__":
    main()
  
