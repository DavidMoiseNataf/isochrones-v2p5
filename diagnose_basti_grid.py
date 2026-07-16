#!/usr/bin/env python3
"""Validate the pseudo-EEP assumptions across a downloaded BaSTI grid.

Run AFTER download_basti.py and BEFORE any fitting. Checks, per
([a/Fe], system) directory:

  1. every block in every file has Np == 2100;
  2. the MS->RGB / He-burning segment boundary sits at row 1289/1290
     (detected as the largest single-row |dlogL| jump);
  3. initial mass is strictly non-decreasing within every isochrone;
  4. theoretical columns (Mini, Mfin, logL, logTe) agree row-for-row across
     photometric systems at the same (age, composition);
  5. the (age x [Fe/H]) coverage matrix, so holes are known before the
     DFInterpolator pads them with NaNs;
  6. drift of the MSTO row (argmax logTe among rows < 1290) across the grid,
     to quantify turnoff interpolation smearing.

Exit code 0 = all hard checks pass (1, 2, 3, 4); coverage and MSTO drift are
reported informationally.

Usage:
    python diagnose_basti_grid.py [--afe -0.2 0.0 0.4] [--isochrones-dir PATH]
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AFE_TAGS = {-0.2: "M02", 0.0: "P00", 0.4: "P04"}
CASE = "O1D1E1"
EXPECTED_NP = 2100
EXPECTED_TIP = 1289


def parse(fn):
    """Lightweight standalone parser (mirrors basti/models.py)."""
    import re
    hdr = re.compile(
        r"#\s*Np\s*=\s*(\d+)\s+\[M/H\]\s*=\s*([\-\+\d\.]+)\s+Z\s*=\s*([\d\.]+)"
        r"\s+Y\s*=\s*([\d\.]+)\s+Age\s*\(Myr\)\s*=\s*([\d\.]+)")
    blocks, meta, rows = [], None, []

    def close():
        if meta is not None:
            blocks.append((meta, np.array(rows, dtype=float)))

    with open(fn, encoding="latin-1") as f:
        for line in f:
            if line.startswith("#"):
                m = hdr.search(line)
                if m:
                    close()
                    rows = []
                    meta = dict(np=int(m.group(1)), mh=float(m.group(2)),
                                z=float(m.group(3)), y=float(m.group(4)),
                                age=float(m.group(5)))
                continue
            if line.strip():
                rows.append(line.split())
    close()
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afe", nargs="+", type=float, default=[-0.2, 0.0, 0.4])
    ap.add_argument("--isochrones-dir",
                    default=os.getenv("ISOCHRONES", os.path.expanduser("~/.isochrones")))
    ap.add_argument("--systems", nargs="+", default=None,
                    help="Restrict to these .isc_ extensions (e.g. "
                         "jwst-nircam_zp_vega-sirius). Default: all found. "
                         "NB: full-grid runs parse every file and can take a "
                         "while; one system suffices for structure checks, "
                         "two or more for the cross-system consistency check.")
    args = ap.parse_args()

    failures = 0
    d = os.path.join(args.isochrones_dir, "basti", "BaSTI_{}_isos".format(CASE))
    if not os.path.isdir(d):
        print("No directory {} -- run download_basti.py first.".format(d))
        sys.exit(1)
    all_files = sorted(f for f in os.listdir(d) if ".isc_" in f.lower())
    if args.systems is not None:
        wanted = tuple(".isc_" + s.replace("isc_", "").lower() for s in args.systems)
        all_files = [f for f in all_files if f.lower().endswith(wanted)]
    for afe in args.afe:
        tag = AFE_TAGS[afe]
        # case-insensitive: tarball members may use lowercase tags
        # (m02o1d1e1) while web-interface downloads use uppercase.
        token = (tag + CASE + ".").lower()
        files = [f for f in all_files if token in f.lower()]
        print("== {}: {} files".format(tag, len(files)))

        coverage = defaultdict(set)          # (z, mh) -> set(ages)
        young_boundary = {}                  # detected drop row -> count (age < 2 Gyr)
        systems_at = defaultdict(dict)       # (z, age) -> {system: theory array}
        msto_rows = []
        for fname in files:
            system = fname.lower().split(".isc_", 1)[1]
            for meta, data in parse(os.path.join(d, fname)):
                key = "z={:.5f} [M/H]={:+.3f}".format(meta["z"], meta["mh"])
                coverage[key].add(meta["age"])

                if meta["np"] != EXPECTED_NP or len(data) != meta["np"]:
                    print("  FAIL Np: {} age {} (Np={}, rows={})".format(
                        fname, meta["age"], meta["np"], len(data)))
                    failures += 1
                    continue

                mini, logL, logTe = data[:, 0], data[:, 2], data[:, 3]
                # Segment-boundary verification, two regimes:
                #  * age >= 2 Gyr (degenerate He ignition): the TRGB is the
                #    unambiguous logL max of the first segment ([:1500]
                #    excludes the AGB). HARD check: must be exactly 1289.
                #  * age < 2 Gyr: INFORMATIONAL ONLY. The boundary is a
                #    construction anchor (Hidalgo+18 Table 4: line 1290 =
                #    TRGB, line 1300 = quiescent He-burning start), not a
                #    morphological feature: near the He-flash transition the
                #    ZAHB descent moves the largest logL drop a few rows into
                #    segment 2, and for very young isochrones (turnoff masses
                #    of several Msun) phase 11 is not a luminosity maximum at
                #    all, so the drop lands wherever the loop sampling puts
                #    it. The detection histogram is reported; nothing fails.
                if meta["age"] >= 2000.0:
                    tip = int(np.argmax(logL[:1500]))
                    if tip != EXPECTED_TIP:
                        print("  FAIL segment boundary: {} age {} -> logL max "
                              "of first segment at row {} (expected {})".format(
                                  fname, meta["age"], tip, EXPECTED_TIP))
                        failures += 1
                else:
                    drops = logL[1250:1350] - logL[1251:1351]
                    tip = int(np.argmax(drops)) + 1250
                    young_boundary[tip] = young_boundary.get(tip, 0) + 1
                if np.any(np.diff(mini) < 0):
                    print("  FAIL mass monotonicity: {} age {}".format(fname, meta["age"]))
                    failures += 1

                msto_rows.append(int(np.argmax(logTe[:EXPECTED_TIP + 1])))

                skey = (meta["z"], meta["age"])
                theory = data[:, :4]
                for other, arr in systems_at[skey].items():
                    if not np.allclose(arr, theory, atol=5e-5, rtol=0):
                        print("  FAIL cross-system theory mismatch: z={} age={} "
                              "{} vs {}".format(meta["z"], meta["age"], system, other))
                        failures += 1
                systems_at[skey][system] = theory

        # coverage matrix
        all_ages = sorted({a for s in coverage.values() for a in s})
        print("  ages found ({}): {} .. {} Myr".format(
            len(all_ages), min(all_ages, default=0), max(all_ages, default=0)))
        n_holes = 0
        for key in sorted(coverage):
            missing = [a for a in all_ages if a not in coverage[key]]
            if missing:
                n_holes += 1
                if n_holes <= 12:
                    print("  coverage hole: {} missing {} of {} ages "
                          "(first few: {})".format(key, len(missing),
                                                   len(all_ages), missing[:5]))
        if n_holes > 12:
            print("  ... and {} more compositions with holes (dense BaSTI age "
                  "grids differ per composition; NaN-padded at build)".format(
                      n_holes - 12))
        # holes BELOW 15 Gyr are the ones that matter for science builds
        n_sci_holes = sum(
            1 for key in coverage
            for a in all_ages
            if a not in coverage[key] and a < 15000.0)
        print("  missing (age, composition) nodes below 15 Gyr: {}".format(n_sci_holes))
        if young_boundary:
            print("  young-age (<2 Gyr) boundary-drop detections by row: {}".format(
                dict(sorted(young_boundary.items()))))
        if msto_rows:
            print("  MSTO row (argmax logTe, pre-tip): min={} median={} max={}".format(
                min(msto_rows), int(np.median(msto_rows)), max(msto_rows)))

    print("\n{} hard-check failure(s).".format(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
