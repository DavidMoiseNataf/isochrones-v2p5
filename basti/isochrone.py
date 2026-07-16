"""BaSTI isochrone interpolator -- params (EEP, age, [Fe/H], distance, AV).

Mirrors ``mist/isochrone_v2p5.py`` structurally, with one architectural
difference driven by the data: BaSTI's absolute magnitudes live IN the model
grid (per-system columns), so there is no bolometric-correction grid at all.
``interp_mag`` is overridden at the Python level:

    m_X = M_X(eep, age, feh)  +  5 log10(d / 10 pc)  +  R_X * A_V

with R_X from ``basti/extinction.py``. This matches the calling convention of
the v2.5 fitting path (StarModelV2p5.lnlike already calls ic.interp_mag in
Python rather than the compiled star_lnlike kernel), so a BaSTI interpolator
drops into that fitter unchanged apart from the [a/Fe] treatment.

[a/Fe] is a fixed discrete choice per interpolator: -0.2, 0.0, or +0.4
(the BaSTI-IAC O1D1E1 offerings), selected exactly like the fixed-alpha
"Option A" path of the MIST v2.5 extension:

    from isochrones.basti.isochrone import get_ichrone_basti
    ic = get_ichrone_basti(bands=["F090W", "F162M"], afe=0.4)
    Teff, logg, feh, mags = ic.interp_mag([eep, age, feh, distance, AV],
                                          ic.bands)
"""

import numpy as np
import pandas as pd

from ..models import IsochroneInterpolator
from .models import BastiIsochroneGrid, DEFAULT_SYSTEMS, BASTI_NP
from .extinction import get_extinction_coeffs, get_Ax_handle

# Which BaSTI photometric systems provide which canonical band tokens.
# Used to auto-select the systems to ingest from the requested bands.
_BAND_TO_SYSTEM_PREFIX = (
    ("WFC3_", "wfc3"),
    ("ACS_WFC_", "acs"),
    ("JC_", "john"),
    ("PS1_", "panstrss1"),
    ("SkyMapper_", "skym"),
    ("2MASS_", "2mass"),
    ("DECam_", "decam"),
    ("Euclid_", "euclid"),
    ("GALEX_", "galex"),
    ("HAWKI_", "hawki"),
    ("TESS_", "tess"),
    ("VISTA_", "vista"),
    ("WISE_", "wise"),
)
_GAIA_BANDS = {"G", "BP", "RP", "G_RVS"}


def systems_for_bands(bands):
    """Infer the minimal set of BaSTI systems needed for the given bands."""
    systems = set()
    for b in bands:
        matched = False
        for prefix, system in _BAND_TO_SYSTEM_PREFIX:
            if b.startswith(prefix):
                systems.add(system)
                matched = True
                break
        if matched:
            continue
        if b in _GAIA_BANDS:
            systems.add("gaia-dr3-new")
        else:
            # bare filter names default to NIRCam, matching the v2.5 convention
            systems.add("jwst-nircam_zp_vega-sirius")
    return tuple(sorted(systems))


class Basti_Isochrone(IsochroneInterpolator):
    """params: (EEP, age [log10 yr], [Fe/H], distance [pc], A_V)"""

    grid_type = BastiIsochroneGrid
    bc_type = None
    eep_bounds = (0, BASTI_NP)
    default_bands = ("F090W", "F150W", "F277W", "F444W")

    def __init__(self, bands=None, afe=0.0, systems=None, **kwargs):
        self.bands = list(bands) if bands is not None else list(self.default_bands)
        if systems is None:
            systems = systems_for_bands(self.bands)
        self._model_grid = None
        self._bc_grid = None
        self.param_index_order = list(self._param_index_order)
        self.kwargs = dict(kwargs, afe=afe, systems=systems)
        self._fehs = None
        self._ages = None
        self._masses = None
        self._Rx = None
        self._mag_props = None
        self._ax_handle = None

    # -- no BC grid -----------------------------------------------------------

    @property
    def bc_grid(self):
        return None

    @property
    def Rx(self):
        """A_X / A_V per requested band, as an array aligned with self.bands."""
        if self._Rx is None:
            coeffs = get_extinction_coeffs(self.bands)
            self._Rx = np.array([coeffs[b] for b in self.bands], dtype=float)
        return self._Rx

    def _check_bands(self):
        cols = set(self.model_grid.df.columns)
        missing = [b for b in self.bands if b not in cols]
        if missing:
            raise ValueError(
                "Band(s) {} not present in the BaSTI grid (systems={}). "
                "Available magnitude columns include: {}".format(
                    missing,
                    self.kwargs["systems"],
                    sorted(c for c in cols if c.isupper() or "_F" in c)[:40],
                )
            )

    # -- magnitudes -----------------------------------------------------------

    def interp_mag(self, pars, bands):
        """pars: [eep, age, feh, distance, AV], scalars or broadcastable arrays.

        Returns (Teff, logg, feh, mags) with mags shaped (n_bands,) for scalar
        input or (n_bands, n) for array input -- same contract as the stock
        compiled path.
        """
        if self._mag_props is None:
            self._check_bands()
            self._mag_props = True

        eep, age, feh, dist, AV = [np.atleast_1d(np.asarray(p, dtype=float))
                                   for p in np.broadcast_arrays(*pars)]
        scalar = eep.size == 1

        props = ["Teff", "logg", "feh"] + list(bands)
        # grid index order: (age, feh, eep)
        grid_pars = [age, feh, eep]
        vals = self.model_grid.interp(grid_pars, props)
        vals = np.atleast_2d(vals)              # (n, n_props)

        Teff = vals[:, 0]
        logg = vals[:, 1]
        feh_out = vals[:, 2]
        abs_mags = vals[:, 3:].T                # (n_bands, n)

        mu = 5.0 * np.log10(dist / 10.0)
        if self._ax_handle is None or self._ax_bands != tuple(bands):
            self._ax_handle, self._ax_source = get_Ax_handle(list(bands))
            self._ax_bands = tuple(bands)
        Ax = self._ax_handle(AV)                # (n_bands,) or (n_bands, n)
        mags = abs_mags + mu[None, :] + np.atleast_2d(Ax.T).T

        if scalar:
            return float(Teff[0]), float(logg[0]), float(feh_out[0]), mags[:, 0]
        return Teff, logg, feh_out, mags

    def __call__(self, p1, p2, p3, distance=10.0, AV=0.0):
        """Derived-quantity DataFrame at (eep, age, feh, distance, AV).

        Overrides IsochroneInterpolator.__call__, whose final concatenation
        assumes magnitudes shaped (n_samples, n_bands). This class's
        interp_mag returns (n_bands, n_samples) instead, so the stock method
        mismatches lengths (the StarModelV2p5._make_samples crash:
        "size N vs size n_bands"). Here magnitudes are transposed back to
        (n_samples, n_bands) before concatenation, and the output columns
        (theory columns + "<band>_mag") match the stock contract that
        _make_samples consumes. Strictly row-preserving: N samples in ->
        N rows out, NaN where the grid can't evaluate, never dropped.
        """
        eep, age, feh, dist, AV = [
            np.atleast_1d(a).astype(float)
            for a in np.broadcast_arrays(p1, p2, p3, distance, AV)
        ]
        prop_cols = list(self.model_grid.df.columns)
        props = np.atleast_2d(self.interp_value([eep, age, feh, dist, AV], prop_cols))
        if props.shape[0] != eep.size and props.shape[1] == eep.size:
            props = props.T                              # -> (n_samples, n_props)
        _, _, _, mags = self.interp_mag([eep, age, feh, dist, AV], self.bands)
        mags = np.atleast_2d(mags)
        if mags.shape[0] == len(self.bands) and mags.shape[1] == eep.size:
            mags = mags.T                                # (n_bands, N) -> (N, n_bands)
        values = np.concatenate([props, mags], axis=1)
        cols = prop_cols + ["{}_mag".format(b) for b in self.bands]
        return pd.DataFrame(values, columns=cols)


def get_ichrone_basti(bands=None, afe=0.0, systems=None, **kwargs):
    """BaSTI analogue of get_ichrone_v2p5_iso.

    Returns the isochrone interpolator (params: EEP, age, [Fe/H], distance,
    AV) built on the BaSTI-IAC O1D1E1 grid at a fixed [a/Fe] in
    {-0.2, 0.0, +0.4}.

    Parameters
    ----------
    bands : list(str)
        Canonical band tokens: bare NIRCam names (F090W, ...), qualified HST
        names (ACS_WFC_F814W, WFC3_UVIS_F390W, WFC3_IR_F160W), Gaia
        (G, BP, RP). The required BaSTI systems are inferred automatically
        unless ``systems`` is given explicitly.
    afe : float
        [alpha/Fe]: one of -0.2, 0.0, +0.4. Fixed, not interpolated.
    age_range : (float, float), optional keyword
        (min_gyr, max_gyr) -- forwarded to the grid; restricts which files
        are parsed at build time (BaSTI carries ~230-306 ages per
        composition). All (age, feh) nodes below 15 Gyr are complete, so any
        science age range yields a hole-free grid. The range is part of the
        cache tag.

    Fitter EEP bounds can be chosen from the official construction anchors
    (Hidalgo+18 Table 4), importable as basti.models.BASTI_EEP_ANCHORS
    (zams=99, msto=359, rgb_base=489, trgb=1289, zaheb=1299, eagb_end=2099).
    """
    return Basti_Isochrone(bands=bands, afe=afe, systems=systems, **kwargs)
