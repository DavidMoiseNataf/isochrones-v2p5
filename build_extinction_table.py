#!/usr/bin/env python3
"""
build_extinction_table.py -- Construct the BaSTI extinction table
R_X(band, A_V) from SVO filter transmission curves and a chosen extinction
law, for use by isochrones/basti at fit time:

    m_X = M_X + mu + R_X(A_V) * A_V

The table captures the A_V-dependence of broadband extinction (the effective
wavelength of a wide filter reddens as A_V grows, so A_X is not linear in
A_V), which a single scalar coefficient cannot.

Physics of each table entry (photon-counting bandpass integral):

    A_X(A_V) = -2.5 log10 [ INT T(l) l F(l) 10^(-0.4 A(l; A_V)) dl
                          / INT T(l) l F(l) dl ]
    R_X(A_V) = A_X(A_V) / A_V

with T from SVO (http://svo2.cab.inta-csic.es/theory/fps/), A(l) from the
`extinction` package (https://extinction.readthedocs.io/), and F(l) a source
SED -- default: a 4500 K blackbody, appropriate for red-giant work
(configurable; the choice matters most for wide and blue filters).

Curves: fm07 (Fitzpatrick & Massa 2007; R_V fixed at 3.1) [DEFAULT],
f99 (Fitzpatrick 1999), ccm89, odonnell94 (these take --rv). Each run writes
    ~/.isochrones/basti/extinction_table_<curve>[ _rv<val> ].npz
with full provenance; rebuilding with a different curve is just a re-run, and
isochrones/basti picks the table by curve name (see basti/extinction.py).

Filter curves are downloaded once and cached under
~/.isochrones/filters/svo/. SVO IDs marked VERIFIED below were confirmed
against real downloads; the rest follow SVO naming conventions and are
validated at runtime -- any 404/parse failure is listed at the end with the
band it affects, and that band is simply omitted from the table (falling back
to the scalar placeholder coefficients at fit time, with a warning). Fixing a
wrong ID is a one-line edit in SVO_IDS.

Usage:
    python build_extinction_table.py                       # fm07, all bands
    python build_extinction_table.py --curve f99 --rv 3.1
    python build_extinction_table.py --bands F090W F162M   # subset
    python build_extinction_table.py --teff 4000           # cooler SED
    python build_extinction_table.py --list                # show ID map
"""

import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timezone

import numpy as np

# np.trapezoid is the NumPy >= 2.0 name; older NumPy calls it np.trapz
try:
    _trapz = np.trapezoid          # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz              # NumPy < 2.0

ISOCHRONES_DIR = os.getenv("ISOCHRONES", os.path.expanduser("~/.isochrones"))
FILTER_CACHE = os.path.join(ISOCHRONES_DIR, "filters", "svo")
SVO_URL = ("https://svo2.cab.inta-csic.es/theory/fps/getdata.php"
           "?format=ascii&id={}")

# A_V grid: dense at low extinction where fits live, extending to heavy
# obscuration. R_X(A_V->0) is evaluated at a small positive A_V.
AV_GRID = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0,
                    4.0, 5.0, 6.0, 8.0, 10.0, 12.5, 15.0, 20.0])

# ---------------------------------------------------------------------------
# Canonical band token -> SVO filter ID.
# VERIFIED patterns (confirmed downloads): JWST/NIRCam2025.*,
# HST/WFC3_UVIS2.*, CTIO/DECam.* . Everything else follows SVO conventions
# and is validated when the script runs. None => no SVO curve exists
# (synthetic/leak bands); such bands are skipped with a notice.
# ---------------------------------------------------------------------------

_NIRCAM = ("F070W F090W F115W F140M F150W F150W2 F162M F182M F200W F210M "
           "F250M F277W F300M F322W2 F335M F356W F360M F410M F430M F444W "
           "F460M F480M").split()
_WFC3_UVIS = ("F218W F225W F275W F336W F390W F438W F475W F555W F606W F625W "
              "F775W F814W F200LP F300X F350LP F475X F600LP F850LP F390M "
              "F410M F467M F547M F621M F689M F763M F845M F395N").split()
_WFC3_IR = "F105W F110W F125W F140W F160W F098M F127M F139M F153M".split()
_ACS = "F435W F475W F555W F606W F625W F775W F814W".split()

SVO_IDS = {}
SVO_IDS.update({b: "JWST/NIRCam2025.{}".format(b) for b in _NIRCAM})
SVO_IDS.update({"WFC3_UVIS_" + b: "HST/WFC3_UVIS2.{}".format(b) for b in _WFC3_UVIS})
SVO_IDS.update({"WFC3_IR_" + b: "HST/WFC3_IR.{}".format(b) for b in _WFC3_IR})
SVO_IDS.update({"ACS_WFC_" + b: "HST/ACS_WFC.{}".format(b) for b in _ACS})
SVO_IDS.update({
    # Gaia DR3
    "G": "GAIA/GAIA3.G", "BP": "GAIA/GAIA3.Gbp", "RP": "GAIA/GAIA3.Grp",
    "G_RVS": "GAIA/GAIA3.Grvs",
    # DECam (VERIFIED pattern)
    "DECam_u": "CTIO/DECam.u", "DECam_g": "CTIO/DECam.g",
    "DECam_r": "CTIO/DECam.r", "DECam_i": "CTIO/DECam.i",
    "DECam_z": "CTIO/DECam.z", "DECam_Y": "CTIO/DECam.Y",
    # 2MASS
    "2MASS_J": "2MASS/2MASS.J", "2MASS_H": "2MASS/2MASS.H",
    "2MASS_Ks": "2MASS/2MASS.Ks",
    # PanSTARRS1
    "PS1_g": "PAN-STARRS/PS1.g", "PS1_r": "PAN-STARRS/PS1.r",
    "PS1_i": "PAN-STARRS/PS1.i", "PS1_z": "PAN-STARRS/PS1.z",
    "PS1_y": "PAN-STARRS/PS1.y", "PS1_w": "PAN-STARRS/PS1.w",
    # SkyMapper (u_leak has no SVO curve)
    "SkyMapper_u": "SkyMapper/SkyMapper.u", "SkyMapper_v": "SkyMapper/SkyMapper.v",
    "SkyMapper_g": "SkyMapper/SkyMapper.g", "SkyMapper_r": "SkyMapper/SkyMapper.r",
    "SkyMapper_i": "SkyMapper/SkyMapper.i", "SkyMapper_z": "SkyMapper/SkyMapper.z",
    "SkyMapper_u_leak": None,
    # Johnson-Cousins (JC_BX and JC_Lprime have no standard SVO curve)
    "JC_U": "Generic/Johnson.U", "JC_B": "Generic/Johnson.B",
    "JC_V": "Generic/Johnson.V", "JC_R": "Generic/Cousins.R",
    "JC_I": "Generic/Cousins.I",
    # JHK: use the 2MASS transmission curves as proxies (per D. Nataf).
    # The Johnson-Glass and 2MASS NIR bandpasses differ by only a few
    # percent in effective wavelength -- negligible for R_X at these
    # wavelengths, where the extinction curve is smooth and shallow.
    "JC_J": "2MASS/2MASS.J", "JC_H": "2MASS/2MASS.H",
    "JC_K": "2MASS/2MASS.Ks",
    # Generic/Johnson.L does not exist on SVO; WISE W1 (3.35 um) is an
    # excellent proxy for Johnson L (3.45 um) -- the extinction curve is
    # nearly flat and small there, so R_X differences are ~1e-3.
    "JC_L": "WISE/WISE.W1", "JC_M": "Generic/Johnson.M",
    "JC_BX": None, "JC_Lprime": None,
    # Euclid
    "Euclid_VIS": "Euclid/VIS.vis", "Euclid_Y": "Euclid/NISP.Y",
    "Euclid_J": "Euclid/NISP.J", "Euclid_H": "Euclid/NISP.H",
    # GALEX
    "GALEX_FUV": "GALEX/GALEX.FUV", "GALEX_NUV": "GALEX/GALEX.NUV",
    # VLT HAWK-I
    "HAWKI_Y": "Paranal/HAWKI.Y", "HAWKI_J": "Paranal/HAWKI.J",
    "HAWKI_H": "Paranal/HAWKI.H", "HAWKI_Ks": "Paranal/HAWKI.Ks",
    "HAWKI_CH4": "Paranal/HAWKI.CH4",
    # TESS
    "TESS_T": "TESS/TESS.Red",
    # VISTA
    "VISTA_Z": "Paranal/VISTA.Z", "VISTA_Y": "Paranal/VISTA.Y",
    "VISTA_J": "Paranal/VISTA.J", "VISTA_H": "Paranal/VISTA.H",
    "VISTA_Ks": "Paranal/VISTA.Ks",
    # WISE (BaSTI provides W1/W2 only)
    "WISE_W1": "WISE/WISE.W1", "WISE_W2": "WISE/WISE.W2",
})

CURVES_NEEDING_RV = {"f99", "ccm89", "odonnell94"}


def curve_function(name, rv):
    import extinction as extn
    if name == "fm07":
        return lambda wave, av: extn.fm07(wave, av)
    if name == "f99":
        return lambda wave, av: extn.fitzpatrick99(wave, av, rv)
    if name == "ccm89":
        return lambda wave, av: extn.ccm89(wave, av, rv)
    if name == "odonnell94":
        return lambda wave, av: extn.odonnell94(wave, av, rv)
    raise ValueError("Unknown curve '{}'".format(name))


def fetch_filter(svo_id):
    """Return (wave_angstrom, transmission), downloading + caching."""
    fname = os.path.join(FILTER_CACHE, re.sub(r"[/:]", "_", svo_id) + ".dat")
    if not os.path.exists(fname):
        os.makedirs(FILTER_CACHE, exist_ok=True)
        url = SVO_URL.format(urllib.request.quote(svo_id, safe=""))
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read().decode("utf-8", "replace")
        if not data.strip() or "<html" in data[:200].lower():
            raise ValueError("SVO returned no ascii data for '{}'".format(svo_id))
        with open(fname, "w") as f:
            f.write(data)
    arr = np.loadtxt(fname)
    if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 5:
        raise ValueError("Unparseable filter file for '{}'".format(svo_id))
    wave, trans = arr[:, 0].astype(float), arr[:, 1].astype(float)
    good = trans > 0
    return wave[good], trans[good]


def source_sed(wave_aa, kind, teff):
    """F_lambda (arbitrary normalization) on the filter wavelength grid."""
    if kind == "flat":
        return np.ones_like(wave_aa)
    if kind == "blackbody":
        # B_lambda ~ l^-5 / (exp(hc/lkT) - 1); constants drop out
        x = 1.4387769e8 / (wave_aa * teff)     # hc/(l k T), l in Angstrom
        with np.errstate(over="ignore"):
            return wave_aa ** -5.0 / np.expm1(np.clip(x, 1e-6, 700.0))
    raise ValueError("Unknown SED kind '{}'".format(kind))


def band_R_of_av(wave, trans, sed, ext_at, av_grid):
    """R_X(A_V) on av_grid via the photon-counting bandpass integral."""
    w = trans * wave * sed                     # photon-counting weight
    denom = _trapz(w, wave)
    out = np.empty(len(av_grid))
    for i, av in enumerate(av_grid):
        alam = ext_at(np.ascontiguousarray(wave, dtype=np.float64), float(av))
        num = _trapz(w * 10.0 ** (-0.4 * alam), wave)
        out[i] = (-2.5 * np.log10(num / denom)) / av
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--curve", default="fm07",
                    choices=["fm07", "f99", "ccm89", "odonnell94"])
    ap.add_argument("--rv", type=float, default=3.1,
                    help="R_V for curves that take it (fm07 is fixed at 3.1)")
    ap.add_argument("--sed", default="blackbody", choices=["blackbody", "flat"])
    ap.add_argument("--teff", type=float, default=4500.0,
                    help="blackbody temperature of the source SED [K]")
    ap.add_argument("--bands", nargs="+", default=None,
                    help="subset of canonical band tokens (default: all)")
    ap.add_argument("--list", action="store_true",
                    help="print the band -> SVO ID map and exit")
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()

    if args.list:
        for b, i in sorted(SVO_IDS.items()):
            print("%-22s %s" % (b, i if i else "(no SVO curve; skipped)"))
        return

    bands = args.bands if args.bands else sorted(SVO_IDS)
    ext_at = curve_function(args.curve, args.rv)
    rv_tag = ("_rv{:g}".format(args.rv) if args.curve in CURVES_NEEDING_RV
              else "")
    outfile = args.outfile or os.path.join(
        ISOCHRONES_DIR, "basti",
        "extinction_table_{}{}.npz".format(args.curve, rv_tag))

    print("curve = {}{}   SED = {}{}   {} bands   A_V grid: {} .. {}".format(
        args.curve, rv_tag, args.sed,
        " ({:g} K)".format(args.teff) if args.sed == "blackbody" else "",
        len(bands), AV_GRID[0], AV_GRID[-1]))

    R = {}
    failed, skipped = [], []
    for b in bands:
        svo_id = SVO_IDS.get(b, "__unknown__")
        if svo_id == "__unknown__":
            failed.append((b, "no entry in SVO_IDS"))
            continue
        if svo_id is None:
            skipped.append(b)
            continue
        try:
            wave, trans = fetch_filter(svo_id)
            sed = source_sed(wave, args.sed, args.teff)
            R[b] = band_R_of_av(wave, trans, sed, ext_at, AV_GRID)
            print("  [ok  ] %-22s %-28s R(0)=%.3f R(%g)=%.3f" % (
                b, svo_id, R[b][0], AV_GRID[-1], R[b][-1]))
        except Exception as e:
            failed.append((b, "{}: {}".format(svo_id, e)))
            print("  [FAIL] %-22s %s" % (b, e))

    if not R:
        raise SystemExit("No bands succeeded; nothing written.")

    band_list = sorted(R)
    matrix = np.array([R[b] for b in band_list])
    provenance = dict(
        curve=args.curve, rv=(args.rv if args.curve in CURVES_NEEDING_RV
                              else 3.1),
        sed=args.sed, teff=args.teff,
        integral="photon-counting: T(l) * l * F(l)",
        svo_ids={b: SVO_IDS[b] for b in band_list},
        built=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    np.savez_compressed(outfile, bands=np.array(band_list), av_grid=AV_GRID,
                        R=matrix, provenance=json.dumps(provenance))
    print("\nWrote {} bands x {} A_V nodes -> {}".format(
        len(band_list), len(AV_GRID), outfile))
    if skipped:
        print("Skipped (no SVO curve exists): {}".format(skipped))
    if failed:
        print("FAILED ({} bands) -- fix the SVO ID in SVO_IDS and re-run "
              "(cached filters are not refetched):".format(len(failed)))
        for b, why in failed:
            print("   {:<22s} {}".format(b, why))


if __name__ == "__main__":
    main()
