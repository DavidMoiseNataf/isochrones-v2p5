"""Wiring for MIST v2.5 (Option A): an EvolutionTrack interpolator that pairs
the v2.5 track grid with the v2.5 BC grid, with [a/Fe] forwarded to both.

Purely additive -- the original mist/isochrone.py and the top-level get_ichrone
are left untouched. Use either the class directly or the get_ichrone_v2p5
convenience factory below.

    from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5
    track = get_ichrone_v2p5(bands=["F090W", "F162M"], afe=0.0)
    Teff, logg, feh, mags = track.interp_mag([mass, eep, feh, distance, AV],
                                             track.bands)

Notes
-----
* This is the EVOLUTION-TRACK interpolator (params: mass, eep, feh, distance,
  AV). It needs only the EEP tracks you downloaded -- no isochrone/full_isos
  grids -- as long as you query by EEP. Age-based lookup (mass, age, feh)
  additionally requires the EEP-fitting precompute, which is the next step.
* eep_bounds is widened to 1721 because the v2.5 tracks extend a few EEPs past
  the v1.2 final EEP of 1710.
* bc_grid is overridden so the BC grid uses the SAME [a/Fe] as the tracks; the
  stock ModelGridInterpolator.bc_grid builds ``bc_type(self.bands)`` and would
  otherwise silently leave the BC grid at afe=0.0.
"""

from ..models import EvolutionTrackInterpolator, IsochroneInterpolator
from .models_v2p5 import MISTEvolutionTrackGridV2p5, MISTIsochroneGridV2p5
from .bc_v2p5 import MISTBolometricCorrectionGridV2p5


class MIST_EvolutionTrackV2p5(EvolutionTrackInterpolator):
    grid_type = MISTEvolutionTrackGridV2p5
    bc_type = MISTBolometricCorrectionGridV2p5
    eep_bounds = (0, 1721)

    # v2.5 isochrone sibling is now available (full_isos downloaded), enabling
    # .iso-based methods. Linked below the class definitions.
    _iso_type = None

    @property
    def bc_grid(self):
        if self._bc_grid is None:
            self._bc_grid = self.bc_type(self.bands, afe=self.kwargs.get("afe", 0.0))
        return self._bc_grid


class MIST_IsochroneV2p5(IsochroneInterpolator):
    """MIST v2.5 isochrone interpolator -- params (EEP, age, [Fe/H], distance, AV).

    This is the interpolator StarModel.fit() samples (it fits age, not mass).
    Pairs the v2.5 isochrone grid with the v2.5 BC grid at a fixed [a/Fe],
    forwarding alpha to the BCs the same way the track interpolator does.
    """

    grid_type = MISTIsochroneGridV2p5
    bc_type = MISTBolometricCorrectionGridV2p5
    eep_bounds = (0, 1721)

    @property
    def bc_grid(self):
        if self._bc_grid is None:
            self._bc_grid = self.bc_type(self.bands, afe=self.kwargs.get("afe", 0.0))
        return self._bc_grid


# Cross-link the v2.5 track/isochrone interpolators (both grids now exist).
MIST_EvolutionTrackV2p5._iso_type = MIST_IsochroneV2p5
MIST_IsochroneV2p5._track_type = MIST_EvolutionTrackV2p5


def get_ichrone_v2p5(bands=None, afe=0.0, vvcrit=0.4, **kwargs):
    """MIST v2.5 analogue of get_ichrone(models='mist', tracks=True).

    Returns the EVOLUTION-TRACK interpolator -- params (mass, EEP, [Fe/H], ...).
    For the isochrone interpolator used by StarModel.fit, see
    get_ichrone_v2p5_iso.

    Parameters
    ----------
    bands : list(str) or None
        Photometric bands (e.g. JWST NIRCam short names). Defaults to the
        BC grid's default_bands if None.
    afe : float
        [alpha/Fe] grid value to use for BOTH tracks and BCs (one of
        -0.2, 0.0, 0.2, 0.4, 0.6). Fixed (Option A), not interpolated.
    vvcrit : float
        Rotation grid (0.0 or 0.4).
    """
    return MIST_EvolutionTrackV2p5(
        bands=bands, version="2.5", afe=afe, vvcrit=vvcrit, **kwargs
    )


def get_ichrone_v2p5_iso(bands=None, afe=0.0, vvcrit=0.4, **kwargs):
    """MIST v2.5 analogue of get_ichrone(models='mist') -- the ISOCHRONE grid.

    Returns the isochrone interpolator (params: EEP, age, [Fe/H], distance, AV),
    which is what StarModel.fit() samples. Requires the v2.5 full_isos grid
    (download with: python download_mist_v25.py --what full_isos).
    """
    return MIST_IsochroneV2p5(
        bands=bands, version="2.5", afe=afe, vvcrit=vvcrit, kind="full_isos", **kwargs
    )


# ---------------------------------------------------------------------------
# Phase II: variable [a/Fe] by interpolating between fixed-alpha interpolators.
#
# The interpolation core has 4-D kernels, but mags.py:interp_mag hardcodes a
# 3-D model + 4-D BC lookup, and the v2.5 BCs depend on [a/Fe] as well -- a true
# 4-D model axis would force a 5-D BC and new numba kernels. Instead we bracket
# the requested alpha and linearly blend two fixed-alpha MIST_IsochroneV2p5
# interpolators. Each already uses matched alpha for BOTH structure and BC, so
# the blend captures both effects, reusing only validated code.
# ---------------------------------------------------------------------------

import numpy as _np
import pandas as _pd


class AlphaInterpIsochrone(object):
    """Variable-[a/Fe] MIST v2.5 isochrone interpolator.

    Presents the ModelGridInterpolator surface that StarModel needs, but takes
    an extra alpha coordinate. Query parameter order:

        interp_mag  : [EEP, age, [Fe/H], [a/Fe], distance, AV]
        interp_value: [EEP, age, [Fe/H], [a/Fe]]

    Sub-interpolators (one per grid alpha) are built lazily, so only the alpha
    values actually queried get their grids constructed.
    """

    _AFES = (-0.2, 0.0, 0.2, 0.4, 0.6)
    name = "mist"
    eep_replaces = "mass"
    eep_bounds = (0, 1721)

    def __init__(self, bands=None, vvcrit=0.4, afes=None):
        self.afes = _np.array(sorted(afes if afes is not None else self._AFES), dtype=float)
        self.afe_bounds = (float(self.afes[0]), float(self.afes[-1]))
        self.vvcrit = vvcrit
        # Construct (cheap; grids build lazily on first interp) one sub per alpha.
        self._subs = {
            float(a): MIST_IsochroneV2p5(
                bands=bands, version="2.5", afe=float(a), vvcrit=vvcrit, kind="full_isos"
            )
            for a in self.afes
        }
        # Resolve bands from any sub (constructing a sub does not build a grid).
        self.bands = self._subs[float(self.afes[0])].bands

    # -- alpha bracketing ---------------------------------------------------
    def _bracket(self, afe):
        afes = self.afes
        a = float(min(max(afe, afes[0]), afes[-1]))  # clamp to grid
        if a <= afes[0]:
            return float(afes[0]), float(afes[0]), 0.0
        if a >= afes[-1]:
            return float(afes[-1]), float(afes[-1]), 0.0
        i = int(_np.searchsorted(afes, a))
        lo, hi = afes[i - 1], afes[i]
        w = (a - lo) / (hi - lo)
        return float(lo), float(hi), float(w)

    def _sub(self, afe):
        return self._subs[float(afe)]

    def __getattr__(self, name):
        # Forward unknown (passive) attributes to a reference sub-interpolator.
        # Uses __dict__ directly to avoid recursion before _subs/afes are set.
        if name.startswith("_") or name == "afes":
            raise AttributeError(name)
        try:
            subs = self.__dict__["_subs"]
            afes = self.__dict__["afes"]
        except KeyError:
            raise AttributeError(name)
        ref = 0.0 if 0.0 in set(afes) else float(afes[len(afes) // 2])
        return getattr(subs[float(ref)], name)

    # -- interpolation surface used by StarModel ----------------------------
    def interp_mag(self, pars, bands):
        pars = list(pars)
        afe = pars[3]
        sub_pars = [pars[0], pars[1], pars[2], pars[4], pars[5]]  # eep, age, feh, dist, AV
        lo, hi, w = self._bracket(afe)
        if w == 0.0:
            return self._sub(lo).interp_mag(sub_pars, bands)
        if w == 1.0:
            return self._sub(hi).interp_mag(sub_pars, bands)
        T0, g0, f0, m0 = self._sub(lo).interp_mag(sub_pars, bands)
        T1, g1, f1, m1 = self._sub(hi).interp_mag(sub_pars, bands)
        blend = lambda x, y: x * (1.0 - w) + y * w
        return (blend(T0, T1), blend(g0, g1), blend(f0, f1),
                blend(_np.asarray(m0), _np.asarray(m1)))

    @property
    def _ref_afe(self):
        """Reference alpha for alpha-independent quantities (e.g. EEP prior)."""
        return 0.0 if 0.0 in set(self.afes) else float(self.afes[len(self.afes) // 2])

    def interp_value(self, pars, props):
        pars = list(pars)
        eep, age, feh = pars[0], pars[1], pars[2]
        if len(pars) < 4:
            # 3-vector [eep, age, feh] (e.g. from EEP_prior) -- the eep->mass /
            # dm_deep weighting is essentially alpha-independent; use ref alpha.
            return self._sub(self._ref_afe).interp_value([eep, age, feh], props)
        afe = pars[3]
        lo, hi, w = self._bracket(afe)
        if w == 0.0:
            return self._sub(lo).interp_value([eep, age, feh], props)
        if w == 1.0:
            return self._sub(hi).interp_value([eep, age, feh], props)
        v0 = _np.asarray(self._sub(lo).interp_value([eep, age, feh], props), dtype=float)
        v1 = _np.asarray(self._sub(hi).interp_value([eep, age, feh], props), dtype=float)
        return v0 * (1.0 - w) + v1 * w

    # -- proxies StarModel / priors reach for ------------------------------
    @property
    def model_grid(self):
        # get_limits(mass/feh/age) etc. -- alpha-independent, use any sub.
        return self._sub(float(self.afes[len(self.afes) // 2])).model_grid

    @property
    def bc_grid(self):
        return self._sub(float(self.afes[len(self.afes) // 2])).bc_grid

    def get_eep(self, *args, **kwargs):
        a = 0.0 if 0.0 in set(self.afes) else float(self.afes[len(self.afes) // 2])
        return self._sub(a).get_eep(*args, **kwargs)

    # -- attributes StarModel reads -----------------------------------------
    @property
    def param_names(self):
        # alpha inserted after feh, before distance
        return ("eep", "age", "feh", "afe", "distance", "AV")

    # Not used internally (interp_mag/interp_value/__call__ are overridden);
    # present only so code that reads ic.param_index_order doesn't break.
    param_index_order = (1, 2, 0, 4, 5)

    def __call__(self, p1, p2, p3, p4, distance=10.0, AV=0.0):
        """Derived-quantity DataFrame at (eep, age, feh, afe, distance, AV).

        Vectorized over arrays (used by StarModel._make_samples). Evaluates each
        grid-alpha sub-interpolator and linearly interpolates every output
        column in alpha per row.
        """
        eep, age, feh, afe, dist, AV = [
            _np.atleast_1d(a).astype(float)
            for a in _np.broadcast_arrays(p1, p2, p3, p4, distance, AV)
        ]
        n = len(eep)
        afes = self.afes
        dfs = [self._sub(float(a))(eep, age, feh, dist, AV) for a in afes]
        cols = dfs[0].columns
        planes = _np.stack([d.values for d in dfs], axis=0)  # (n_afe, n, ncols)

        afe_c = _np.clip(afe, afes[0], afes[-1])
        idx = _np.clip(_np.searchsorted(afes, afe_c), 1, len(afes) - 1)
        lo = afes[idx - 1]
        hi = afes[idx]
        w = _np.where(hi > lo, (afe_c - lo) / (hi - lo), 0.0)
        rows = _np.arange(n)
        lo_vals = planes[idx - 1, rows, :]
        hi_vals = planes[idx, rows, :]
        out = lo_vals * (1.0 - w)[:, None] + hi_vals * w[:, None]
        return _pd.DataFrame(out, columns=cols)


def get_ichrone_v2p5_alpha(bands=None, vvcrit=0.4, afes=None):
    """Variable-[a/Fe] MIST v2.5 isochrone interpolator (for fitting alpha).

    Returns an AlphaInterpIsochrone whose interp_mag/interp_value take an extra
    [a/Fe] coordinate after [Fe/H]. Requires the v2.5 full_isos grid.
    """
    return AlphaInterpIsochrone(bands=bands, vvcrit=vvcrit, afes=afes)
