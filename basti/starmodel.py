"""StarModel layer for BaSTI fits.

Two entry points, mirroring the two MIST v2.5 fitting modes:

FIXED [a/Fe] (recommended default for BaSTI -- see notch caveat below):

    from isochrones.basti import get_ichrone_basti
    from isochrones.basti.starmodel import BastiStarModel

    ic = get_ichrone_basti(bands=["F090W", "F162M", "F460M"], afe=0.4)
    model = BastiStarModel(ic, F090W=(m1, e1), F162M=(m2, e2), F460M=(m3, e3))
    model.fit(...)
    # parameter vector: [eep, age, feh, distance, AV]

FREE [a/Fe] over the three BaSTI nodes (-0.2, 0.0, +0.4):

    from isochrones.basti.starmodel import get_ichrone_basti_alpha
    from isochrones.mist.starmodel_v2p5 import StarModelV2p5   # unchanged!

    ic = get_ichrone_basti_alpha(bands=[...])
    model = StarModelV2p5(ic, **obs)
    # parameter vector: [eep, age, feh, afe, distance, AV]

The free-alpha wrapper presents exactly the surface StarModelV2p5 reads
(afe_bounds, interp_mag with the 6-vector, interp_value, param_names), so the
existing v2.5 star model is reused verbatim.

===========================================================================
THE (feh, afe) SUPPORT NOTCH -- read before free-alpha BaSTI fits
===========================================================================
BaSTI's [a/Fe] = +0.4 grid tops out near [Fe/H] = +0.06 while the -0.2 and
0.0 grids extend to ~+0.45. Under QUADRATIC alpha interpolation every afe
strictly inside (-0.2, +0.4) -- and the whole extrapolation zone above +0.4
-- touches the +0.4 grid, so for feh > feh_max(+0.4 grid) the likelihood is
-inf at every alpha except exactly on the -0.2/0.0 nodes (measure zero for a
continuous sampler). Metal-rich stars are therefore effectively excluded
from free-alpha BaSTI fits, not merely pushed to low alpha. This implicit
prior is left in deliberately for now (per discussion); prefer fixed-alpha
runs for super-solar targets and revisit before interpreting them.
``BastiAlphaInterpIsochrone.support_note`` carries this warning
programmatically.
===========================================================================
"""

import numpy as np
import pandas as pd

from ..starmodel import SingleStarModel
from ..likelihood import gauss_lnprob
from .isochrone import Basti_Isochrone


class BastiStarModel(SingleStarModel):
    """Single-star model on a fixed-alpha BaSTI isochrone interpolator.

    Parameter vector: [eep, age, feh, distance, AV] (stock ordering).

    Only lnlike is overridden: the stock version calls the compiled
    star_lnlike kernel against a model grid + BC grid pair, and the BaSTI
    interpolator has no BC grid (magnitudes live in the model grid). The
    Gaussian likelihood is recomputed in Python via ic.interp_mag, exactly
    as StarModelV2p5 does for the alpha-interpolating MIST case. The stock
    ``bands`` property is also overridden: it reads ic.bc_grid.bands, which
    doesn't exist here, so photometric bands are matched against ic.bands.
    """

    @property
    def bands(self):
        if self._bands is None:
            self._bands = [k for k in self.kwargs if k in self.ic.bands]
        return self._bands

    def lnlike(self, pars):
        eep, age, feh, dist, AV = [float(p) for p in pars[:5]]

        Teff, logg, feh_model, mags = self.ic.interp_mag(
            [eep, age, feh, dist, AV], self.bands
        )
        Teff = float(np.atleast_1d(Teff)[0])
        logg = float(np.atleast_1d(logg)[0])
        feh_model = float(np.atleast_1d(feh_model)[0])
        if not np.isfinite(Teff):
            return -np.inf

        lnl = 0.0
        for b, mod in zip(self.bands, np.atleast_1d(mags).ravel()):
            val, unc = self.kwargs[b]
            lnl += gauss_lnprob(val, unc, float(mod))

        for prop, mod in zip(["Teff", "logg", "feh"], [Teff, logg, feh_model]):
            if prop in self.kwargs:
                val, unc = self.kwargs[prop]
                lnl += gauss_lnprob(val, unc, mod)

        if "parallax" in self.kwargs:
            plax, plax_unc = self.kwargs["parallax"]
            lnl += gauss_lnprob(plax, plax_unc, 1000.0 / dist)

        return lnl if np.isfinite(lnl) else -np.inf


class BastiAlphaInterpIsochrone(object):
    """Variable-[a/Fe] BaSTI isochrone interpolator over (-0.2, 0.0, +0.4).

    Interpolation scheme (per D. Nataf's specification):

    METAL-POOR REGIME, [Fe/H] below the +0.4 grid's ceiling (currently +0.06):
      * -0.2 <= afe <= +0.4 : evaluate the model prediction at ALL THREE grid
        alphas and combine with the 3-point Lagrange (quadratic) weights;
      * +0.4 <  afe <= +0.6 : evaluate at afe = 0.0 and +0.4 and extrapolate
        linearly (clamped at +0.6);
      * afe <  -0.2         : clamped to -0.2.

    METAL-RICH REGIME, [Fe/H] >= the +0.4 grid's ceiling (only the -0.2 and
    0.0 grids exist there):
      * -0.2 <= afe <= +0.4 : LINEAR in alpha through the (-0.2, 0.0) nodes,
        extended across the full range (i.e. extrapolation for afe > 0);
      * afe > +0.4          : NaN (-inf likelihood);
      * afe <  -0.2         : clamped to -0.2.

    The regime threshold ``feh_quad_max`` is read from the highest-alpha
    sub-grid's actual [Fe/H] upper limit (so it adapts automatically if the
    BaSTI server gains metal-rich +0.4 nodes); it can be overridden by
    assigning the attribute.

    Weights always sum to 1 in both regimes. Sub-grids with weight exactly 0
    (queries exactly on a node) are never evaluated.

    Query parameter order (unchanged):
        interp_mag  : [EEP, age, [Fe/H], [a/Fe], distance, AV]
        interp_value: [EEP, age, [Fe/H], [a/Fe]]  (or 3-vector, ref alpha)
    """

    _AFES = (-0.2, 0.0, 0.4)
    AFE_EXTRAP_MAX = 0.6
    name = "basti"
    eep_replaces = "mass"
    eep_bounds = Basti_Isochrone.eep_bounds

    support_note = (
        "BaSTI free-alpha: for feh >= feh_quad_max (the +0.4 grid's [Fe/H] "
        "ceiling, currently ~+0.06), alpha is linear through the -0.2/0.0 "
        "grids over [-0.2, +0.4] and NaN-censored above +0.4."
    )

    def __init__(self, bands=None, afes=None, systems=None, **sub_kwargs):
        # sub_kwargs (e.g. age_range=(0.02, 14.5)) are forwarded to every
        # fixed-alpha sub-interpolator and hence into the grid cache tags --
        # REQUIRED to reuse grids prebuilt with an age_range, since the tag
        # encodes it.
        self.afes = np.array(sorted(afes if afes is not None else self._AFES), dtype=float)
        self.afe_bounds = (float(self.afes[0]),
                           float(self.AFE_EXTRAP_MAX if len(self.afes) >= 2
                                 else self.afes[-1]))
        self._subs = {
            float(a): Basti_Isochrone(bands=bands, afe=float(a), systems=systems,
                                      **sub_kwargs)
            for a in self.afes
        }
        self.bands = self._subs[float(self.afes[0])].bands
        self._feh_quad_max = None   # lazy; override by assigning feh_quad_max

    # -- regime threshold ------------------------------------------------------
    @property
    def feh_quad_max(self):
        """[Fe/H] ceiling of the highest-alpha grid; quadratic regime below,
        linear(-0.2, 0.0) regime at/above. Currently ~+0.06 for BaSTI O1D1E1."""
        if self._feh_quad_max is None:
            top = self._sub(float(self.afes[-1]))
            self._feh_quad_max = float(top.model_grid.get_limits("feh")[1])
        return self._feh_quad_max

    @feh_quad_max.setter
    def feh_quad_max(self, value):
        self._feh_quad_max = float(value)

    # -- alpha weights ---------------------------------------------------------
    def _alpha_weights(self, afe, feh):
        """Return [(node, weight), ...] with zero-weight nodes omitted, or
        None to signal NaN (metal-rich + afe > top node)."""
        afes = self.afes
        if len(afes) == 1:
            return [(float(afes[0]), 1.0)]
        x = float(afe)

        if float(feh) >= self.feh_quad_max and len(afes) >= 3:
            # metal-rich: linear through the two LOWEST (feh-complete) nodes,
            # valid over [afes[0], afes[-1]]; NaN above.
            if x > afes[-1]:
                return None
            x = max(x, float(afes[0]))
            a0, a1 = float(afes[0]), float(afes[1])
            w1 = (x - a0) / (a1 - a0)
            if w1 == 0.0:
                return [(a0, 1.0)]
            if w1 == 1.0:
                return [(a1, 1.0)]
            return [(a0, 1.0 - w1), (a1, w1)]

        if x <= afes[0]:
            return [(float(afes[0]), 1.0)]
        if x > afes[-1]:
            x = min(x, float(self.AFE_EXTRAP_MAX))
            a1, a2 = float(afes[-2]), float(afes[-1])
            w2 = (x - a1) / (a2 - a1)
            return [(a1, 1.0 - w2), (a2, w2)]
        # Lagrange weights over all nodes
        out = []
        for i, ai in enumerate(afes):
            w = 1.0
            for j, aj in enumerate(afes):
                if j != i:
                    w *= (x - aj) / (ai - aj)
            if abs(w) > 1e-12:
                out.append((float(ai), float(w)))
        return out

    def _sub(self, afe):
        return self._subs[float(afe)]

    @property
    def _ref_afe(self):
        return 0.0 if 0.0 in set(self.afes) else float(self.afes[len(self.afes) // 2])

    def __getattr__(self, name):
        if name.startswith("_") or name == "afes":
            raise AttributeError(name)
        try:
            subs = self.__dict__["_subs"]
        except KeyError:
            raise AttributeError(name)
        return getattr(subs[self._ref_afe], name)

    # -- interpolation surface ------------------------------------------------
    def interp_mag(self, pars, bands):
        pars = list(pars)
        sub_pars = [pars[0], pars[1], pars[2], pars[4], pars[5]]
        nodes = self._alpha_weights(pars[3], pars[2])
        if nodes is None:
            nan = np.nan
            return nan, nan, nan, np.full(len(bands), np.nan)
        T = g = f = None
        m = None
        for a, w in nodes:
            Ti, gi, fi, mi = self._sub(a).interp_mag(sub_pars, bands)
            mi = np.asarray(mi)
            if T is None:
                T, g, f, m = w * Ti, w * gi, w * fi, w * mi
            else:
                T, g, f, m = T + w * Ti, g + w * gi, f + w * fi, m + w * mi
        return T, g, f, m

    def interp_value(self, pars, props):
        pars = list(pars)
        eep, age, feh = pars[0], pars[1], pars[2]
        if len(pars) < 4:
            return self._sub(self._ref_afe).interp_value([eep, age, feh], props)
        nodes = self._alpha_weights(pars[3], feh)
        if nodes is None:
            return np.full(len(props) if not isinstance(props, str) else 1, np.nan)
        out = None
        for a, w in nodes:
            v = np.asarray(self._sub(a).interp_value([eep, age, feh], props), dtype=float)
            out = w * v if out is None else out + w * v
        return out

    # -- proxies StarModel / priors reach for ---------------------------------
    @property
    def model_grid(self):
        return self._sub(self._ref_afe).model_grid

    @property
    def bc_grid(self):
        # StarModelV2p5's inherited ``bands`` property reads ic.bc_grid.bands;
        # BaSTI has no BC grid, so expose a minimal shim carrying the band
        # list. Nothing else on the object is reachable from the Python
        # (interp_mag-based) likelihood path.
        class _BandsShim(object):
            def __init__(self, bands):
                self.bands = list(bands)
        return _BandsShim(self.bands)

    @property
    def param_names(self):
        return ("eep", "age", "feh", "afe", "distance", "AV")

    param_index_order = (1, 2, 0, 4, 5)

    def __call__(self, p1, p2, p3, p4, distance=10.0, AV=0.0):
        """Derived-quantity DataFrame at (eep, age, feh, afe, distance, AV).

        Vectorized: evaluates each grid-alpha sub once for all rows, then
        combines with per-row quadratic/extrapolation weights. Zero-weight
        contributions are masked so on-node rows never inherit NaNs from
        grids they don't use.
        """
        eep, age, feh, afe, dist, AV = [
            np.atleast_1d(a).astype(float)
            for a in np.broadcast_arrays(p1, p2, p3, p4, distance, AV)
        ]
        n = len(eep)
        afes = [float(a) for a in self.afes]
        dfs = [self._sub(a)(eep, age, feh, dist, AV) for a in afes]
        cols = dfs[0].columns
        planes = np.stack([d.values for d in dfs], axis=0)   # (n_afe, n, ncols)

        W = np.zeros((n, len(afes)))
        censored = np.zeros(n, dtype=bool)
        for i in range(n):
            nodes = self._alpha_weights(afe[i], feh[i])
            if nodes is None:
                censored[i] = True
                continue
            for a, w in nodes:
                W[i, afes.index(a)] = w

        contrib = np.where(W.T[:, :, None] != 0.0,
                           W.T[:, :, None] * planes, 0.0)
        out = contrib.sum(axis=0)
        out[censored, :] = np.nan
        return pd.DataFrame(out, columns=cols)


def get_ichrone_basti_alpha(bands=None, afes=None, systems=None, **kwargs):
    """Variable-[a/Fe] BaSTI isochrone interpolator (drop-in for the
    StarModelV2p5 fitter). Extra kwargs (notably age_range=(min_gyr, max_gyr))
    are forwarded to the underlying grids -- pass the SAME age_range used at
    grid-build time to reuse those caches. Read the module docstring's
    support-notch warning before using for metal-rich alpha-enhanced
    targets."""
    return BastiAlphaInterpIsochrone(bands=bands, afes=afes, systems=systems, **kwargs)
