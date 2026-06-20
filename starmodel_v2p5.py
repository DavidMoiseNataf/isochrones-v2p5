"""StarModel that fits [a/Fe] as a free parameter on MIST v2.5.

Additive: subclasses the stock SingleStarModel and pairs it with the
AlphaInterpIsochrone wrapper. The parameter vector gains [a/Fe] after [Fe/H]:

    [eep, age, feh, afe, distance, AV]

BasicStarModel drives mnest_prior / lnprior / n_params / bounds / _make_samples
off self.ic.param_names, so once the wrapper advertises an 'afe' name and we
register an 'afe' prior + bound, those all work unchanged. Only two things need
overriding:

  * lnlike -- the stock version calls the compiled star_lnlike against a single
    3-D model grid + 4-D BC grid, which the 5-sub-grid wrapper can't provide; we
    recompute the Gaussian likelihood in Python using the wrapper's interp_mag.
  * the hardcoded distance_index / AV_index (3, 4 for a 5-vector) shift to
    (4, 5) because alpha is inserted at index 3.

Usage:
    from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_alpha
    from isochrones.mist.starmodel_v2p5 import StarModelV2p5

    ic = get_ichrone_v2p5_alpha(bands=["F090W", "F162M", "F460M"])
    model = StarModelV2p5(ic, F090W=(m1, e1), F162M=(m2, e2), F460M=(m3, e3),
                          parallax=(plx, plx_err), Teff=(T, dT), feh=(f, df))
    model.fit(refit=True, n_live_points=LP, evidence_tolerance=ET,
              basename=base, verbose=False)
    model.samples            # now has an 'afe' column
"""

import numpy as np

from ..starmodel import SingleStarModel
from ..likelihood import gauss_lnprob
from ..priors import FlatPrior


class StarModelV2p5(SingleStarModel):
    def __init__(self, ic, **kwargs):
        super().__init__(ic, **kwargs)

        # alpha is inserted after feh (index 3); distance/AV shift up by one.
        self.afe_index = 3
        self.distance_index = 4
        self.AV_index = 5

        # register the [a/Fe] prior + bounds so the inherited mnest_prior /
        # lnprior / bounds machinery handles it like any other parameter.
        afe_bounds = tuple(float(x) for x in ic.afe_bounds)
        self._priors["afe"] = FlatPrior(bounds=afe_bounds)
        self._bounds["afe"] = afe_bounds

    def lnlike(self, pars):
        eep = float(pars[0])
        age = float(pars[1])
        feh = float(pars[2])
        afe = float(pars[3])
        dist = float(pars[4])
        AV = float(pars[5])

        Teff, logg, feh_model, mags = self.ic.interp_mag(
            [eep, age, feh, afe, dist, AV], self.bands
        )
        Teff = float(np.atleast_1d(Teff)[0])
        logg = float(np.atleast_1d(logg)[0])
        feh_model = float(np.atleast_1d(feh_model)[0])
        if not np.isfinite(Teff):
            return -np.inf

        lnl = 0.0

        # photometry
        for b, mod in zip(self.bands, np.atleast_1d(mags)):
            val, unc = self.kwargs[b]
            lnl += gauss_lnprob(val, unc, float(mod))

        # spectroscopy (Teff, logg, feh) if provided
        for prop, mod in zip(["Teff", "logg", "feh"], [Teff, logg, feh_model]):
            if prop in self.kwargs:
                val, unc = self.kwargs[prop]
                lnl += gauss_lnprob(val, unc, mod)

        # parallax
        if "parallax" in self.kwargs:
            plax, plax_unc = self.kwargs["parallax"]
            lnl += gauss_lnprob(plax, plax_unc, 1000.0 / dist)

        return lnl if np.isfinite(lnl) else -np.inf
