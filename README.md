# isochrones-v2p5

MIST **v2.5** (α-enhanced) stellar models for Tim Morton's
[`isochrones`](https://github.com/timothydmorton/isochrones) package.

This is an **additive extension**, not a rewrite. It adds parallel `*_v2p5`
modules that subclass the existing `isochrones` classes and override only what
MIST v2.5 requires. The original source files are left completely untouched, and
the v1.2 grids remain the default — nothing in your existing workflow changes
unless you explicitly ask for the v2.5 path.

The headline new capability is that **[α/Fe] can be a free parameter in a fit**,
interpolated across the five MIST α-nodes ({−0.2, 0.0, +0.2, +0.4, +0.6}),
alongside the usual EEP, age, [Fe/H], distance, and Aᵥ.

---

## What MIST v2.5 changes (and how this handles it)

| v2.5 change | Where it's handled |
|---|---|
| New `[α/Fe]` abundance axis; GS98 mixture (Z⊙/X⊙ = 0.0231); 17-point `[Fe/H]` grid | `models_v2p5`, `isochrone_v2p5` |
| `log_surf_z` → `log_surf_cell_z` (tracks and isochrones) | `compute_additional_columns` overrides |
| BC tables use a `# lgTef` (log₁₀ Teff) header + `Fe_H`/`a_Fe`/`Rv` axes | `bc_v2p5.parse_table` |
| JWST BC columns prefixed `NIRCAM_`; F470N/F480M mislabeled as F470W/F480W | `bc_v2p5._V25_PREFIX`, `_V25_BAND_FIX` |
| Qualified HST band names (`ACS_WFC_*`, `WFC3_*`) the stock resolver can't parse | `bc_v2p5.resolve_band_v25` short-circuits |
| EEP grid extends to 1721; non-uniform per-`[Fe/H]` mass sampling | widened bounds + tolerant array-grid builders |
| On-disk layout differs from v1.2 | `download_mist_v25.py` migrates to the v1.2 decimal convention |

Two design choices keep the α axis cheap: the **fixed-α** path collapses
`[α/Fe]` (and Rv = 3.1) to recover the exact v1.2 interpolator layout, and the
**variable-α** path (`AlphaInterpIsochrone`) wraps five fixed-α grids and
linearly interpolates their outputs in α. The second adds only ~1.4× wall-clock
over a v1.2 fit, because the per-evaluation cost is dominated by fixed overhead,
not the extra interpolation.

---

## Requirements

- A working install of `isochrones` (this code subclasses it). Any version
  works — the v2.5 modules resolve all of their own bands (NIRCam, HST/ACS,
  HST/WFC3) and never depend on the parent package's band tables.
- `pymultinest` + MultiNest for the fitting examples; `corner` for the example
  plots. The grids themselves only need `numpy`/`pandas`/`pytables`.
- The MIST v2.5 data (tracks, bolometric corrections, isochrones) — see below.

> Tested against `pandas < 2.0` and Python 3.10, matching a typical `isochrones`
> conda environment.

---

## Installation

The `_v2p5` modules import from inside the `isochrones` package, so they need to
sit alongside it. With `isochrones` already installed, copy the four core modules
into its `mist/` directory:

```bash
# find your installed package
python -c "import isochrones, os; print(os.path.dirname(isochrones.__file__))"

# copy the four modules into that package's mist/ subdirectory
cp mist/*_v2p5.py  <that-path>/mist/
```

The downloader and examples can live anywhere. Nothing else in `isochrones` is
touched, and the v1.2 behavior is unchanged.

---

## Getting the data

`download_mist_v25.py` is stdlib-only and fetches each component on request:

```bash
python download_mist_v25.py --what tracks       # EEP evolutionary tracks
python download_mist_v25.py --what bc           # bolometric corrections (per system)
python download_mist_v25.py --what full_isos    # precomputed isochrones
python download_mist_v25.py --what bc --bc-systems JWST HST_ACS_WFC HST_WFC3
python download_mist_v25.py --reorganize-existing   # migrate an existing layout
```

Data lands under `$ISOCHRONES` (default `~/.isochrones`), with v2.5 kept separate
from v1.2 (`BC/mist/v2/`, `mist/MIST_v2.5_vvcrit0.4_full_isos/`, etc.).

---

## Quickstart

### Fixed [α/Fe] — drop-in replacement for a v1.2 fit

```python
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_iso
from isochrones import SingleStarModel

ic = get_ichrone_v2p5_iso(bands=["F090W", "F162M", "F200W"], afe=0.4)
model = SingleStarModel(ic, F090W=(21.03, 0.02), F162M=(19.20, 0.02),
                        F200W=(18.98, 0.02), parallax=(0.5, 0.1))
model.fit()
print(model.derived_samples[["mass", "Teff", "logg"]].median())
```

### Variable [α/Fe] — α as a fit parameter

```python
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_alpha
from isochrones.mist.starmodel_v2p5 import StarModelV2p5

ic = get_ichrone_v2p5_alpha(bands=["ACS_WFC_F475W", "F090W", "F162M",
                                   "F200W", "F300M", "F356W", "F460M"])
model = StarModelV2p5(ic, **{b: (mag, 0.02) for b, mag in obs.items()})
model.fit(n_live_points=500)
print(model.samples["afe"].median())   # parameter vector: eep, age, feh, afe, distance, AV
```

See `examples/fit_m31_giants_v25.py` (two stars, predicted-band corner plots) and
`examples/fit_m31_sample_v25.py` (a sample, with cross-star systematics
diagnostics) for complete, runnable workflows.

---

## Band naming

- **NIRCam** filters use bare names (`F090W`, `F162M`, `F300M`, …) — they resolve
  to the JWST system automatically.
- **HST** filters must be system-qualified: `ACS_WFC_F475W`, `WFC3_UVIS_F390W`.
  Bare `F475W`/`F390W` are ambiguous (both ACS and WFC3 have them) and will not
  resolve.

---

## Factories and classes

| Object | Purpose |
|---|---|
| `get_ichrone_v2p5_iso(bands, afe)` | isochrone interpolator at fixed α |
| `get_ichrone_v2p5(bands, afe)` | evolution-track interpolator at fixed α |
| `get_ichrone_v2p5_alpha(bands)` | α-interpolating isochrone (α as a parameter) |
| `MISTBolometricCorrectionGridV2p5` | v2.5 BC grid |
| `MISTIsochroneGridV2p5`, `MISTEvolutionTrackGridV2p5` | v2.5 model grids |
| `AlphaInterpIsochrone` | wraps five fixed-α grids; linear in α |
| `StarModelV2p5` | single-star model with an `[α/Fe]` parameter |

---

## Scientific caveats

A few things this code makes *possible* but does not make *correct* — read before
interpreting results:

- **α is only as good as its photometric leverage.** Inferring [α/Fe] from broad-
  band photometry (e.g. a water-sensitive band such as F300M) is indirect; the
  absolute scale carries a systematic floor set by model BCs and zero-points, and
  α is partially degenerate with [Fe/H] along the total-metallicity ridge.
  Relative, star-to-star α is more robust than the absolute value.
- **The TP-AGB is not safe for surface-abundance work.** Stars above the RGB tip
  (EEP ≳ 605) undergo dredge-up that alters surface C/O, so molecular-band
  abundance proxies no longer track natal [α/Fe]. Restrict abundance
  interpretation to the RGB.
- **Ages of giants are prior-dominated** without a main-sequence turnoff.
- **The extinction law matters.** v2.5 carries an Rv axis; the fixed-α collapse
  freezes Rv = 3.1. A wrong reddening curve produces wavelength-monotonic
  photometric residuals that the fit will absorb into Aᵥ.

---

## Attribution and license

This extension builds directly on the `isochrones` package by Tim Morton and on
the MIST stellar models (Choi et al. 2016; Dotter 2016). If you use it, please
cite `isochrones`, MIST, and MultiNest as appropriate, in addition to this
repository.

Distributed under the same MIT license as `isochrones` (see `LICENSE`). The
original `isochrones` source is unmodified; this repository adds sibling modules
only.
