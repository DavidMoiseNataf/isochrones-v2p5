#!/usr/bin/env python3
"""
fit_m31_sample_v25.py -- fit an M31 red-giant SAMPLE with MIST v2.5, [a/Fe] free,
and produce cross-star systematics diagnostics.

Same per-star fit as fit_m31_giants_v25.py (F475W + 6 NIRCam bands, alpha free,
F300M carrying the oxygen/alpha signal), scaled to many stars. Outputs:
  - per-star best-fit params + fit time to screen, then a summary table
  - ~/m31_sample_results.csv             (machine-readable results)
  - ~/m31_sample_alpha_fits.pdf          (per-star corners + summary pages)

Systematics pages at the end of the PDF:
  - best-fit residual (observed - model) per band across all stars. A band whose
    residuals are consistently off-zero flags a zero-point or model systematic;
    watch F300M, since that's where the alpha leverage lives.
  - [a/Fe] vs [Fe/H] and [a/Fe] vs Teff, to expose spurious trends (alpha that
    tracks temperature/extinction rather than real abundance).

Run:  conda activate isochrones ; cd ~ ; python ~/fit_m31_sample_v25.py
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
# short labels for printing / residual plot
OBS_LABELS = ["F475W", "F090W", "F162M", "F200W", "F300M", "F356W", "F460M"]
# predicted-only bands (shown in the corner, NOT fit)
PRED_BANDS = ["WFC3_UVIS_F390W", "F115W", "F250M"]
ALL_BANDS = OBS_BANDS + PRED_BANDS

ERR = 0.02

# M31: mu ~ 24.38 -> d ~ 752 kpc. Gaussian distance prior; age > 100 Myr.
M31_DIST_MU, M31_DIST_SIG = 752000.0, 20000.0
DIST_BOUNDS = (700000.0, 800000.0)
AGE_BOUNDS = (8.0, 10.13)   # log10 yr; >100 Myr
AV_BOUNDS = (0.0, 1.0)

N_LIVE = 500   # lower to ~300 to speed up large samples

# one row per star: [F475W, F090W, F162M, F200W, F300M, F356W, F460M]
STARS = [
    [24.659, 20.822,  18.8511, 18.6191, 18.5583, 18.4104, 18.7129],
    [24.718, 20.9774, 19.0236, 18.738,  18.7197, 18.571,  18.8846],
    [25.354, 20.7283, 18.636,  18.3794, 18.3811, 18.1262, 18.3956],
    [24.786, 21.203,  19.2265, 18.979,  18.9312, 18.8172, 19.0396],
    [24.662, 21.8523, 20.1585, 19.9517, 19.9128, 19.8277, 20.0364],
    [26.462, 24.5895, 23.3663, 23.2708, 23.2457, 23.2626, 23.2769],
    [24.763, 20.6988, 18.6388, 18.3821, 18.3538, 18.1888, 18.4341],
    [24.884, 21.3085, 19.4049, 19.1696, 19.1276, 19.037,  19.3383],
    [24.613, 21.2792, 19.3477, 19.1144, 19.0669, 18.966,  19.2373],
    [25.051, 21.8512, 20.0338, 19.8156, 19.7336, 19.6804, 19.9365],
    [24.232, 21.1707, 19.378,  19.174,  19.1237, 19.0519, 19.2753],
    [24.192, 21.4225, 19.6854, 19.4925, 19.4457, 19.3766, 19.6567],
    [25.548, 22.3608, 20.5466, 20.3468, 20.2905, 20.2165, 20.4836],
    [24.78,  20.7271, 18.6959, 18.437,  18.4295, 18.2622, 18.5622],
]

CORNER_COLS = ["eep", "age", "feh", "afe", "AV",
               "WFC3_UVIS_F390W_mag", "F115W_mag", "F250M_mag"]
CORNER_LABELS = ["EEP", "log(age)", "[Fe/H]", "[a/Fe]", "AV", "F390W", "F115W", "F250M"]

CSV_PATH = os.path.expanduser("~/m31_sample_results.csv")
PDF_PATH = os.path.expanduser("~/m31_sample_alpha_fits.pdf")


def col(s, p):
    return p if p in s.columns else next((c for c in s.columns if c.startswith(p)), None)


def fit_one(name, mags):
    ic = get_ichrone_v2p5_alpha(bands=ALL_BANDS)
    obs = {b: (float(m), ERR) for b, m in zip(OBS_BANDS, mags)}
    model = StarModelV2p5(ic, **obs)
    model.set_bounds(distance=DIST_BOUNDS, age=AGE_BOUNDS, AV=AV_BOUNDS)
    model._priors["distance"] = GaussianPrior(M31_DIST_MU, M31_DIST_SIG, bounds=DIST_BOUNDS)

    t0 = time.time()
    model.fit(n_live_points=N_LIVE, verbose=False, overwrite=True,
              basename=os.path.expanduser(f"~/chains_m31/{name}-"))
    return model, time.time() - t0


def summarize(name, mags, model, dt):
    s, ds = model.samples, model.derived_samples
    row = {"star": name, "fit_time_s": round(dt, 1)}
    for p in ["eep", "age", "feh", "afe", "distance", "AV"]:
        c = col(s, p)
        row[p] = s[c].median()
        row[p + "_lo"] = s[c].quantile(.16)
        row[p + "_hi"] = s[c].quantile(.84)
    for q in ["mass", "Teff", "logg"]:
        if q in ds.columns:
            row[q] = ds[q].median()
    for b, lab in zip(PRED_BANDS, ["F390W", "F115W", "F250M"]):
        row["pred_" + lab] = ds[b + "_mag"].median()
    # best-fit photometric residual (observed - model) per observed band
    for b, lab, m in zip(OBS_BANDS, OBS_LABELS, mags):
        row["resid_" + lab] = float(m) - ds[b + "_mag"].median()
    return row


def corner_page(pdf, name, model):
    s, ds = model.samples, model.derived_samples
    data = {}
    for cname in CORNER_COLS:
        data[cname] = (ds[cname].values if cname.endswith("_mag")
                       else s[col(s, cname)].values)
    cdf = pd.DataFrame(data)
    fig = corner.corner(cdf.values, labels=CORNER_LABELS,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".2f",
                        title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 8})
    fig.suptitle(f"{name}  (MIST v2.5, [a/Fe] free)", fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def systematics_pages(pdf, df):
    # page 1: residuals per band across the sample
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    rng = np.random.default_rng(0)
    for i, lab in enumerate(OBS_LABELS):
        y = df["resid_" + lab].values
        x = np.full_like(y, i) + rng.uniform(-0.12, 0.12, size=len(y))
        ax.scatter(x, y, s=22, alpha=0.7, color="C0")
        ax.plot([i - 0.25, i + 0.25], [np.median(y)] * 2, color="k", lw=2)
    ax.axhline(0.0, color="0.5", ls="--", lw=1)
    ax.set_xticks(range(len(OBS_LABELS)))
    ax.set_xticklabels(OBS_LABELS)
    ax.set_ylabel("residual  (observed - model)  [mag]")
    ax.set_title("Best-fit photometric residuals per band (black bar = median)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # page 2: alpha trends
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    aerr = np.vstack([df["afe"] - df["afe_lo"], df["afe_hi"] - df["afe"]])
    ferr = np.vstack([df["feh"] - df["feh_lo"], df["feh_hi"] - df["feh"]])
    axes[0].errorbar(df["feh"], df["afe"], xerr=ferr, yerr=aerr,
                     fmt="o", ms=5, capsize=2, lw=1)
    axes[0].axhline(0.0, color="0.6", ls="--", lw=1)
    axes[0].set_xlabel("[Fe/H]"); axes[0].set_ylabel("[a/Fe]")
    axes[0].set_title("alpha vs metallicity")
    axes[1].errorbar(df["Teff"], df["afe"], yerr=aerr, fmt="o", ms=5, capsize=2, lw=1)
    axes[1].axhline(0.0, color="0.6", ls="--", lw=1)
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Teff [K]"); axes[1].set_ylabel("[a/Fe]")
    axes[1].set_title("alpha vs Teff (look for spurious trend)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    rows = []
    pdf = PdfPages(PDF_PATH)
    for i, mags in enumerate(STARS, start=1):
        name = f"star_{i:02d}"
        print(f"[{name}] fitting ({len(STARS)} total, n_live={N_LIVE}) ...", flush=True)
        model, dt = fit_one(name, mags)
        row = summarize(name, mags, model, dt)
        rows.append(row)
        print(f"  eep={row['eep']:.1f}  age={row['age']:.2f}  feh={row['feh']:+.2f}  "
              f"afe={row['afe']:+.2f}  AV={row['AV']:.2f}  Teff={row['Teff']:.0f}  "
              f"logg={row['logg']:.2f}  mass={row['mass']:.2f}  ({dt:.1f}s)", flush=True)
        corner_page(pdf, name, model)

    df = pd.DataFrame(rows).set_index("star")
    df.to_csv(CSV_PATH)
    systematics_pages(pdf, df.reset_index())
    pdf.close()

    cols = ["eep", "age", "feh", "afe", "AV", "Teff", "logg", "mass", "fit_time_s"]
    print("\n================ sample summary (medians) ================")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df[cols].to_string(float_format=lambda x: f"{x:8.3f}"))
    print("\nmedian per-band residual (obs - model), mag:")
    print("  " + "  ".join(f"{lab}:{df['resid_'+lab].median():+.3f}" for lab in OBS_LABELS))
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
