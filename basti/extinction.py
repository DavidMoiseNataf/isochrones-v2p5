"""Extinction coefficients R_X = A_X / A_V for BaSTI-based magnitudes.

BaSTI .isc files carry reddening-free absolute magnitudes, so extinction is
applied by the interpolator as   m_X = M_X + mu + R_X * A_V.

Three sources of extinction, in order of preference:

  0. An A_V-DEPENDENT TABLE R_X(band, A_V), built by
     ``build_extinction_table.py`` from SVO filter transmission curves and a
     chosen extinction law (default Fitzpatrick & Massa 2007 'fm07'; f99 /
     ccm89 / odonnell94 available via re-run). The table captures the
     nonlinearity of broadband extinction in A_V and is the production path:
     A_X = R_X(A_V) * A_V, with R_X linearly interpolated on the table's A_V
     grid. Selected via extinction_table_<curve>.npz under
     ~/.isochrones/basti/; the active curve is DEFAULT_CURVE below (or
     set_extinction_curve()).

Legacy scalar sources, used only when no table (or no table entry for a
band) exists:

  1. A derived table, ~/.isochrones/basti/extinction_coeffs.json, produced by
     ``derive_extinction_coeffs_from_mist_v25()`` below: for each band, the
     coefficient is d(BC)/d(Av) evaluated from the MIST v2.5 BC tables at a
     red-giant reference point. This makes the BaSTI extinction treatment
     numerically consistent with the MIST v2.5 fits (same underlying
     Fitzpatrick Rv=3.1 curve, same stellar SED), which is exactly what you
     want for a model-systematics comparison.

  2. The PLACEHOLDER table below (approximate Rv=3.1 values for a cool giant,
     assembled from standard literature curves). A logger warning is emitted
     once per session when these are used. Replace them by running the
     derivation on a machine with the MIST v2.5 BC data.

Note this is a single scalar per band: no Teff/logg/[Fe/H] dependence of the
effective extinction, and A_V-linearity is assumed. That is the deliberate
"phase 1" simplification; the derivation function stores the reference point
used so the approximation can be tightened later.
"""

import os
import json

import numpy as np

from ..config import ISOCHRONES
from ..logger import getLogger

COEFF_FILE = os.path.join(ISOCHRONES, "basti", "extinction_coeffs.json")

# Reference red giant for coefficient derivation (typical M31 RGB target)
REFERENCE_POINT = {"Teff": 4500.0, "logg": 1.5, "feh": -0.5}

# ---------------------------------------------------------------------------
# PLACEHOLDER coefficients (Rv = 3.1, cool-giant SED, approximate).
# >>> Regenerate with derive_extinction_coeffs_from_mist_v25() before any
#     production fitting. These are order-of-magnitude-correct only. <<<
# ---------------------------------------------------------------------------
PLACEHOLDER_COEFFS = {
    # --- JWST NIRCam (wide) ---
    "F070W": 0.746, "F090W": 0.552, "F115W": 0.386, "F150W": 0.242,
    "F200W": 0.152, "F277W": 0.088, "F356W": 0.058, "F444W": 0.043,
    "F150W2": 0.220, "F322W2": 0.070,
    # --- JWST NIRCam (medium) ---
    "F140M": 0.272, "F162M": 0.213, "F182M": 0.174, "F210M": 0.139,
    "F250M": 0.104, "F300M": 0.075, "F335M": 0.063, "F360M": 0.056,
    "F410M": 0.048, "F430M": 0.045, "F460M": 0.041, "F480M": 0.038,
    # --- Gaia DR3 (giant SED) ---
    "G": 0.83, "BP": 1.00, "RP": 0.63, "G_RVS": 0.59,
    # --- HST ACS/WFC ---
    "ACS_WFC_F435W": 1.30, "ACS_WFC_F475W": 1.18, "ACS_WFC_F555W": 1.03,
    "ACS_WFC_F606W": 0.92, "ACS_WFC_F625W": 0.87, "ACS_WFC_F775W": 0.65,
    "ACS_WFC_F814W": 0.59, "ACS_WFC_F850LP": 0.48,
    # --- HST WFC3/UVIS (subset; extend as needed) ---
    "WFC3_UVIS_F336W": 1.65, "WFC3_UVIS_F390W": 1.45, "WFC3_UVIS_F438W": 1.32,
    "WFC3_UVIS_F475W": 1.18, "WFC3_UVIS_F555W": 1.04, "WFC3_UVIS_F606W": 0.92,
    "WFC3_UVIS_F625W": 0.87, "WFC3_UVIS_F775W": 0.65, "WFC3_UVIS_F814W": 0.60,
    # --- HST WFC3/IR ---
    "WFC3_IR_F098M": 0.49, "WFC3_IR_F105W": 0.43, "WFC3_IR_F110W": 0.39,
    "WFC3_IR_F125W": 0.34, "WFC3_IR_F140W": 0.28, "WFC3_IR_F160W": 0.24,
    # --- Johnson-Cousins (giant SED, Rv=3.1) ---
    "JC_U": 1.56, "JC_BX": 1.32, "JC_B": 1.30, "JC_V": 1.00, "JC_R": 0.83,
    "JC_I": 0.60, "JC_J": 0.29, "JC_H": 0.18, "JC_K": 0.12,
    "JC_L": 0.06, "JC_Lprime": 0.05, "JC_M": 0.04,
    # --- PanSTARRS1 ---
    "PS1_g": 1.17, "PS1_r": 0.86, "PS1_i": 0.67, "PS1_z": 0.52,
    "PS1_y": 0.43, "PS1_w": 0.90,
    # --- SkyMapper ---
    "SkyMapper_u": 1.60, "SkyMapper_v": 1.50, "SkyMapper_g": 1.10,
    "SkyMapper_r": 0.86, "SkyMapper_i": 0.63, "SkyMapper_z": 0.49,
    "SkyMapper_u_leak": 1.55,
    # (Roman coefficients removed with Roman support -- see download_basti.py)
    # --- 2MASS / VISTA / HAWK-I NIR ---
    "2MASS_J": 0.29, "2MASS_H": 0.18, "2MASS_Ks": 0.12,
    "VISTA_Z": 0.50, "VISTA_Y": 0.42, "VISTA_J": 0.28, "VISTA_H": 0.18,
    "VISTA_Ks": 0.12,
    "HAWKI_J": 0.28, "HAWKI_H": 0.18, "HAWKI_Ks": 0.12, "HAWKI_Y": 0.42,
    "HAWKI_CH4": 0.14,
    # --- DECam / Euclid / GALEX / TESS / WISE ---
    "DECam_u": 1.57, "DECam_g": 1.20, "DECam_r": 0.84, "DECam_i": 0.63,
    "DECam_z": 0.48, "DECam_Y": 0.42,
    "Euclid_VIS": 0.72, "Euclid_Y": 0.40, "Euclid_J": 0.28, "Euclid_H": 0.19,
    "GALEX_FUV": 2.60, "GALEX_NUV": 2.85,
    "TESS_T": 0.63,
    "WISE_W1": 0.07, "WISE_W2": 0.05,   # BaSTI provides W1/W2 only
}

_warned = {"placeholder": False}


def get_extinction_coeffs(bands):
    """Return {band: R_X} for the requested bands.

    Loads the derived JSON if present; falls back to placeholders with a
    warning. Raises KeyError listing any bands with no coefficient at all.
    """
    coeffs = {}
    derived = {}
    if os.path.exists(COEFF_FILE):
        with open(COEFF_FILE) as f:
            derived = json.load(f).get("coeffs", {})

    missing = []
    used_placeholder = []
    for b in bands:
        if b in derived:
            coeffs[b] = float(derived[b])
        elif b in PLACEHOLDER_COEFFS:
            coeffs[b] = PLACEHOLDER_COEFFS[b]
            used_placeholder.append(b)
        else:
            missing.append(b)

    if missing:
        raise KeyError(
            "No extinction coefficient for band(s) {}. Add them to "
            "PLACEHOLDER_COEFFS or regenerate {} with "
            "derive_extinction_coeffs_from_mist_v25().".format(missing, COEFF_FILE)
        )
    if used_placeholder and not _warned["placeholder"]:
        _warned["placeholder"] = True
        getLogger().warning(
            "Using PLACEHOLDER extinction coefficients for {} -- approximate "
            "values only. Run basti.extinction.derive_extinction_coeffs_"
            "from_mist_v25() to derive consistent ones.".format(used_placeholder)
        )
    return coeffs


def derive_extinction_coeffs_from_mist_v25(
    bands=None, reference=None, av=1.0, afe=0.0, outfile=COEFF_FILE
):
    """Derive R_X = [BC_X(Av=0) - BC_X(Av)] / Av from the MIST v2.5 BC grid.

    Requires the MIST v2.5 BC tables (bc_v2p5 machinery + downloaded data).
    Since A_X = -[BC_X(Av) - BC_X(0)] by the isochrones sign convention,
    the coefficient per band is evaluated at a red-giant reference point and
    written, with full provenance, to ``outfile``.

    Run this once on the machine that has the v2.5 BC data; every BaSTI
    interpolator afterwards will pick the derived values up automatically.
    """
    from ..mist.bc_v2p5 import MISTBolometricCorrectionGridV2p5

    if bands is None:
        bands = [b for b in PLACEHOLDER_COEFFS if not b.startswith(("WFC3", "ACS", "G", "BP", "RP"))]
    ref = dict(REFERENCE_POINT)
    if reference:
        ref.update(reference)

    bc = MISTBolometricCorrectionGridV2p5(bands, afe=afe)
    p0 = [ref["Teff"], ref["logg"], ref["feh"], 0.0]
    p1 = [ref["Teff"], ref["logg"], ref["feh"], float(av)]
    bc0 = np.atleast_1d(bc.interp(p0, bands)).astype(float)
    bc1 = np.atleast_1d(bc.interp(p1, bands)).astype(float)
    coeffs = {b: float((c0 - c1) / av) for b, c0, c1 in zip(bands, bc0, bc1)}

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(
            {
                "coeffs": coeffs,
                "provenance": {
                    "source": "MIST v2.5 BC tables (bc_v2p5)",
                    "reference_point": ref,
                    "av": av,
                    "afe": afe,
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    getLogger().info("Wrote {} extinction coefficients to {}".format(len(coeffs), outfile))
    return coeffs


# ===========================================================================
# A_V-dependent extinction table (production path)
# ===========================================================================

DEFAULT_CURVE = "fm07"
_active_curve = None      # set via set_extinction_curve(); None -> DEFAULT_CURVE
_table_cache = {}


def set_extinction_curve(curve):
    """Select which extinction_table_<curve>.npz subsequent lookups use
    (e.g. 'fm07', 'f99_rv3.1'). Takes effect for interpolators created
    afterwards; existing interpolators cache their handle."""
    global _active_curve
    _active_curve = curve


def _table_path(curve=None):
    c = curve or _active_curve or DEFAULT_CURVE
    return os.path.join(ISOCHRONES, "basti",
                        "extinction_table_{}.npz".format(c))


class ExtinctionTable(object):
    """Loads R_X(band, A_V) and serves fast A_X lookups.

    ``handle(bands)`` returns a callable h(av) -> A_X array of shape
    (n_bands,) for scalar av or (n_bands, n) for array av, with R_X linearly
    interpolated (and clamped) on the table's A_V grid.
    """

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.bands = [str(b) for b in z["bands"]]
            self.av_grid = z["av_grid"].astype(float)
            self.R = z["R"].astype(float)
            self.provenance = json.loads(str(z["provenance"]))
        self._index = {b: i for i, b in enumerate(self.bands)}
        self.path = path

    @property
    def curve(self):
        return self.provenance.get("curve")

    def has(self, band):
        return band in self._index

    def handle(self, bands):
        missing = [b for b in bands if b not in self._index]
        if missing:
            raise KeyError(missing)
        Rsub = self.R[[self._index[b] for b in bands]]   # (n_bands, n_grid)
        grid = self.av_grid

        def h(av):
            av_arr = np.atleast_1d(np.asarray(av, dtype=float))
            avc = np.clip(av_arr, grid[0], grid[-1])
            j = np.clip(np.searchsorted(grid, avc), 1, len(grid) - 1)
            w = (avc - grid[j - 1]) / (grid[j] - grid[j - 1])
            Rv = Rsub[:, j - 1] * (1.0 - w) + Rsub[:, j] * w   # (n_bands, n)
            Ax = Rv * av_arr[None, :]
            return Ax[:, 0] if np.isscalar(av) or np.ndim(av) == 0 else Ax
        return h


def load_extinction_table(curve=None):
    """Return the ExtinctionTable for the active curve, or None if absent."""
    path = _table_path(curve)
    if path not in _table_cache:
        _table_cache[path] = ExtinctionTable(path) if os.path.exists(path) else None
    return _table_cache[path]


def get_Ax_handle(bands):
    """Return (callable av -> A_X array, source_description).

    Prefers the A_V-dependent table; any band absent from it falls back to
    the scalar coefficients (derived JSON, then placeholders) with a warning.
    """
    table = load_extinction_table()
    if table is not None:
        in_table = [b for b in bands if table.has(b)]
        if len(in_table) == len(bands):
            return table.handle(bands), "table:{}".format(table.curve)
        getLogger().warning(
            "Extinction table '{}' lacks band(s) {}; using scalar "
            "coefficients for all requested bands instead.".format(
                table.path, [b for b in bands if not table.has(b)]))
    coeffs = get_extinction_coeffs(list(bands))
    R = np.array([coeffs[b] for b in bands], dtype=float)

    def h(av):
        av_arr = np.atleast_1d(np.asarray(av, dtype=float))
        Ax = R[:, None] * av_arr[None, :]
        return Ax[:, 0] if np.isscalar(av) or np.ndim(av) == 0 else Ax
    return h, "scalar"
