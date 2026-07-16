#!/usr/bin/env python3
"""
build_basti_grids.py -- Build (and cache) the production BaSTI interpolation
grids for the M31/M33 fitting configuration, one per [alpha/Fe].

Everything a fit needs is precomputed here, so the first fitter run starts
instantly instead of paying a many-minute grid build. Three cache layers are
written per (alpha, systems, age_range) configuration under
$ISOCHRONES/basti/, each tagged with the code version, alpha, systems, and
age range -- different configurations can never collide:

    basti_<tag>.h5        the assembled isochrone DataFrame
    dm_deep_<tag>.h5      d(initial mass)/d(EEP) per isochrone (IMF weight)
    full_grid_<tag>.npz   the regularized DFInterpolator array

Edit the CONFIG block, or override it on the command line:

    python build_basti_grids.py
    python build_basti_grids.py --bands F090W F162M --afes 0.0 --age-range 0.1 14.5

Sizing guidance (per alpha, JWST-only, age_range=(0.02, 14.5)):
  ~20 compositions x ~240 ages x 2100 rows ~= 10 M rows; expect a few GB of
  RAM at the DFInterpolator-regularization peak and ~1-3 GB of cache on disk.
  If memory is tight: narrow AGE_RANGE (rows scale linearly with the number
  of retained ages) or reduce BANDS (columns scale with bands). The peak is
  the NPZ build, which happens once per configuration.

Rerunning is cheap: existing caches are detected and only the sanity probes
run. To force a rebuild after a code change, bump BASTI_GRID_VERSION in
basti/models.py (renames all cache files) or delete the three files with the
old tag.
"""

import os
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

# ============================ CONFIG =======================================

# Canonical band tokens (bare NIRCam names; ACS_WFC_* / WFC3_UVIS_* /
# WFC3_IR_*; Gaia G/BP/RP; see readme_basti.md). The required BaSTI
# photometric systems are inferred from these automatically.
BANDS = ["F090W", "F162M", "F200W", "F300M", "F356W", "F460M"]

# One grid is built per entry. {-0.2, 0.0, +0.4} are the BaSTI offerings.
AFES = [0.0, 0.4, -0.2]

# (min, max) in Gyr. Coverage below 15 Gyr is complete (no NaN padding).
AGE_RANGE = (0.02, 14.5)

# Reference queries for the post-build sanity probe: (eep, log10 age, feh).
PROBES = [
    (1289, 10.0, -1.0),     # TRGB, 10 Gyr, metal-poor
    (1150, 9.9, -0.5),      # upper RGB
    (1400, 10.0, -1.0),     # core He burning / clump
]
PROBE_DIST_PC = 776e3       # M31
PROBE_AV = 0.3

# ===========================================================================


def fmt_bytes(path):
    try:
        n = os.path.getsize(path)
    except OSError:
        return "absent"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return "%.1f %s" % (n, unit)
        n /= 1024.0
    return "%.1f TB" % n


def main():
    global BANDS, AFES, AGE_RANGE
    ap = argparse.ArgumentParser(description="Pre-build BaSTI production grids")
    ap.add_argument("--bands", nargs="+", default=None,
                    help="canonical band tokens (default: CONFIG BANDS)")
    ap.add_argument("--afes", nargs="+", type=float, default=None,
                    help="[alpha/Fe] values to build (default: CONFIG AFES)")
    ap.add_argument("--age-range", nargs=2, type=float, default=None,
                    metavar=("MIN_GYR", "MAX_GYR"),
                    help="age window in Gyr (default: CONFIG AGE_RANGE)")
    args = ap.parse_args()
    if args.bands:
        BANDS = args.bands
    if args.afes:
        AFES = args.afes
    if args.age_range:
        AGE_RANGE = tuple(args.age_range)

    from isochrones.basti import get_ichrone_basti
    from isochrones.basti.extinction import load_extinction_table

    tab = load_extinction_table()
    if tab is None:
        print("NOTE: no extinction table found -- interp_mag will fall back "
              "to scalar placeholder coefficients (warned per session). "
              "Run build_extinction_table.py first for production fits.")
    else:
        missing = [b for b in BANDS if not tab.has(b)]
        print("extinction table: {} ({} bands){}".format(
            tab.curve, len(tab.bands),
            "" if not missing else " -- MISSING {}: will fall back to "
            "scalars".format(missing)))

    for afe in AFES:
        print("\n================ [alpha/Fe] = {:+.1f} ================".format(afe))
        ic = get_ichrone_basti(bands=BANDS, afe=afe, age_range=AGE_RANGE)
        grid = ic.model_grid
        print("systems inferred : {}".format(grid.systems))
        print("cache tag        : {}".format(grid.kwarg_tag))

        cached = os.path.exists(grid.hdf_filename)
        t0 = time.time()
        df = grid.df                       # builds/loads HDF + dm_deep
        t1 = time.time()
        if cached:
            print("(loaded from existing cache)")
        n_ages = len(df.index.levels[0])
        n_fehs = len(df.index.levels[1])
        print("grid dataframe   : {:,} rows  ({} ages x {} [Fe/H] x 2100 EEP)"
              "  [{:.1f} s]".format(len(df), n_ages, n_fehs, t1 - t0))
        feh_max = max(df.index.levels[1])
        print("  [Fe/H] nodes   : {} .. {}".format(
            round(min(df.index.levels[1]), 3), round(feh_max, 3)))
        if abs(afe - 0.4) < 1e-6:
            print("  NOTE: this ceiling ({:+.3f}) is feh_quad_max -- the "
                  "free-alpha regime switch".format(feh_max))
        print("  dm_deep NaNs   : {:,} / {:,}".format(
            int(df.dm_deep.isna().sum()), len(df)))

        t2 = time.time()
        _ = grid.interp                    # builds/loads the NPZ (memory peak)
        t3 = time.time()
        print("DFInterpolator   : ready  [{:.1f} s]".format(t3 - t2))

        for eep, age, feh in PROBES:
            T, logg, f, mags = ic.interp_mag(
                [eep, age, feh, PROBE_DIST_PC, PROBE_AV], ic.bands)
            print("  probe (eep={:>4}, age={:>5.2f}, feh={:+.1f}) -> "
                  "Teff={:6.0f} logg={:5.2f}  {}".format(
                      eep, age, feh, T, logg,
                      "  ".join("%s=%.3f" % (b, m)
                                for b, m in zip(ic.bands, mags))))

        print("cache files:")
        for p in [grid.hdf_filename, grid.interp_grid_npz_filename,
                  os.path.join(grid.datadir, "dm_deep{}.h5".format(grid.kwarg_tag))]:
            print("  {:>9}  {}".format(fmt_bytes(p), os.path.basename(p)))

    print("\nAll grids built. Fitters constructing get_ichrone_basti with "
          "the same BANDS / afe / AGE_RANGE will load these caches directly.")


if __name__ == "__main__":
    main()
