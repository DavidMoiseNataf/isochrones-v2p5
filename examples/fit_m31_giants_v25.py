#!/usr/bin/env python3
"""
fit_m31_giants_v25.py -- fit M31 red giants with MIST v2.5, [a/Fe] free.

Fits two Andromeda red giants from F475W(ACS) + 6 NIRCam bands, with alpha as a
free parameter (the F300M water band carries the oxygen/alpha information in cool
giants). For each star it prints best-fit parameters + fit time, and writes a
corner plot that ALSO shows the predicted apparent magnitudes in F390W(WFC3),
F115W, and F250M. All corner plots go into one multi-page PDF.

SETUP
  - models_v2p5.py, isochrone_v2p5.py, starmodel_v2p5.py in .../isochrones/mist/.
  - v2.5 full_isos + BC for JWST, HST_ACS_WFC, HST_WFC3 present.
  - conda activate isochrones ; cd ~ ; python ~/fit_m31_giants_v25.py

Band naming: NIRCam filters use bare names (resolve to JWST); HST filters must be
system-qualified ("ACS_WFC_F475W", "WFC3_UVIS_F390W"), since bare F475W/F390W are
ambiguous between ACS and WFC3.
"""

# ---- quiet header (before isochrones / pymultinest / MPI import) ----
import os
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("OMPI_MCA_shmem", "posix")

import warnings
warnings.filterwarnings("ignore", message="object name is not a valid Python identifier")

import numpy as np
np.seterr(divide="ignore", invalid="ignore")
# --------------------------------------------------------------------

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import corner

from isochrones.priors import GaussianPrior
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_alpha
from isochrones.mist.starmodel_v2p5 import StarModelV2p5

# observed bands (enter the likelihood)
OBS_BANDS = ["ACS_WFC_F475W", "F090W", "F162M", "F200W", "F300M", "F356W", "F460M"]
# predicted-only bands (shown in the corner, NOT fit)
PRED_BANDS = ["WFC3_UVIS_F390W", "F115W", "F250M"]
ALL_BANDS = OBS_BANDS + PRED_BANDS

ERR = 0.02

# M31: mu ~ 24.38 -> d ~ 752 kpc. Gaussian distance prior; age > 100 Myr.
M31_DIST_MU, M31_DIST_SIG = 752000.0, 20000.0
DIST_BOUNDS = (700000.0, 800000.0)
AGE_BOUNDS = (8.0, 10.13)   # log10 yr; >100 Myr
AV_BOUNDS = (0.0, 1.0)

N_LIVE = 500

STARS = {
    "M31_giant_1": [24.176, 21.0347, 19.203, 18.983, 18.9383, 18.8542, 19.1075],
    "M31_giant_2": [24.197, 22.0078, 20.6331, 20.503, 20.4402, 20.4224, 20.4927],
}

# corner: fit params + predicted apparent mags
CORNER_COLS = ["eep", "age", "feh", "afe", "AV",
               "WFC3_UVIS_F390W_mag", "F115W_mag", "F250M_mag"]
CORNER_LABELS = ["EEP", "log(age)", "[Fe/H]", "[a/Fe]", "AV",
                 "F390W", "F115W", "F250M"]

PDF_PATH = os.path.expanduser("~/m31_giant_alpha_fits.pdf")


def fit_one(name, mags):
    ic = get_ichrone_v2p5_alpha(bands=ALL_BANDS)
    obs = {b: (float(m), ERR) for b, m in zip(OBS_BANDS, mags)}
    model = StarModelV2p5(ic, **obs)

    model.set_bounds(distance=DIST_BOUNDS, age=AGE_BOUNDS, AV=AV_BOUNDS)
    model._priors["distance"] = GaussianPrior(M31_DIST_MU, M31_DIST_SIG, bounds=DIST_BOUNDS)

    print(f"\n[{name}] fitting ({model.n_params} params, n_live={N_LIVE}) ...")
    t0 = time.time()
    model.fit(n_live_points=N_LIVE, verbose=False, overwrite=True,
              basename=os.path.expanduser(f"~/chains_m31/{name}-"))
    dt = time.time() - t0
    return model, dt


def col(s, p):
    return p if p in s.columns else next((c for c in s.columns if c.startswith(p)), None)


def report(name, model, dt):
    s = model.samples
    ds = model.derived_samples
    print(f"\n=== {name}: best-fit median [16, 84] ===")
    for p in ["eep", "age", "feh", "afe", "distance", "AV"]:
        c = col(s, p)
        print(f"  {p:9s} {s[c].median():11.3f}  [{s[c].quantile(.16):.3f}, {s[c].quantile(.84):.3f}]")
    for q, lab in [("mass", "mass"), ("Teff", "Teff"), ("logg", "logg")]:
        if q in ds.columns:
            print(f"  {lab:9s} {ds[q].median():11.3f}  "
                  f"[{ds[q].quantile(.16):.3f}, {ds[q].quantile(.84):.3f}]")
    print("  predicted apparent mags:")
    for b in PRED_BANDS:
        mc = b + "_mag"
        if mc in ds.columns:
            print(f"    {b:18s} {ds[mc].median():8.3f}  "
                  f"[{ds[mc].quantile(.16):.3f}, {ds[mc].quantile(.84):.3f}]")
    print(f"  fit time: {dt:.1f} s")


def corner_frame(model):
    s, ds = model.samples, model.derived_samples
    data = {}
    for cname in CORNER_COLS:
        if cname.endswith("_mag"):
            data[cname] = ds[cname].values
        else:
            data[cname] = s[col(s, cname)].values
    return pd.DataFrame(data)


def main():
    pdf = PdfPages(PDF_PATH)
    for name, mags in STARS.items():
        model, dt = fit_one(name, mags)
        report(name, model, dt)

        cdf = corner_frame(model)
        fig = corner.corner(
            cdf.values, labels=CORNER_LABELS,
            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".2f",
            title_kwargs={"fontsize": 9}, label_kwargs={"fontsize": 9},
        )
        fig.suptitle(f"{name}  (MIST v2.5, [a/Fe] free)", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"\nWrote corner plots to {PDF_PATH}")


if __name__ == "__main__":
    main()
