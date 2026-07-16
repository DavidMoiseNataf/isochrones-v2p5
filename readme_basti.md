# BaSTI-IAC support for `isochrones` (additive extension)

## 1. Overview and model physics

Adds the [BaSTI-IAC](http://basti-iac.oa-abruzzo.inaf.it/) stellar models
(Hidalgo et al. 2018; Pietrinferni et al. 2021) to `isochrones`, with
[α/Fe] as a fit parameter. Physics case "O1D1E1": convective-core
overshooting + atomic diffusion + Reimers mass loss η = 0.3, Y_BBN = 0.247,
at [α/Fe] = −0.2, 0.0, +0.4.

**Purely additive:** installation adds one new subpackage directory,
`<site-packages>/isochrones/basti/` (five modules, no suffixes — the
directory itself is the namespace). Nothing stock is modified.

## 2. Installing the code

```bash
python -c "import isochrones, os; print(os.path.dirname(isochrones.__file__))"
cp -r basti  <that-path>/
```

## 3. Downloading the data

One command fetches everything (3 alphas × 15 photometric systems, ~885
tarballs → **206,655 isochrone files**; tens of GB):

```bash
cd ~/.isochrones
python download_basti.py --what isos --scrape
```

- `--scrape` falls back to the built-in probe-verified seed manifest (the
  server 403-blocks listings); the fallback message is expected.
- Resumable via `~/.isochrones/manifest_basti_O1D1E1.json`; subsets via
  `--afe` and `--iso-systems`; `--probe` / `--dry-run` / `--guess-ext`
  available.
- Systems: JWST WFC3 ACS GAIA 2MASS DECAM EUCLID GALEX HAWKI TESS VISTA
  WISE JC PANSTARRS SKYMAPPER. **Roman is deliberately excluded** (three
  server product generations with inconsistent per-alpha tarball coverage;
  see the comment in `download_basti.py`).
- Files land flat in `~/.isochrones/basti/BaSTI_O1D1E1_isos/`; grid caches
  are written beside them on first use and are never downloaded.

## 4. Verifying the data

```bash
python diagnose_basti_grid.py --systems jwst-nircam_zp_vega-sirius acs
```

Hard-fails on: point count ≠ 2100, non-monotonic initial mass, TRGB not at
row 1289 for ages ≥ 2 Gyr, or theory columns differing between photometric
systems at the same node. Expected on a complete download: `0 hard-check
failure(s)` and `missing (age, composition) nodes below 15 Gyr: 0` for every
alpha (all coverage raggedness is super-Hubble). Use ≥ 2 `--systems` to
exercise the cross-system check.

## 5. Extinction

Applied at fit time as `m_X = M_X + mu + R_X(A_V) · A_V`, with
`R_X(band, A_V)` from a precomputed table — the A_V dependence captures the
nonlinearity of broadband extinction. Build once (needs the
[`extinction`](https://extinction.readthedocs.io/) package; SVO filter
curves are fetched and cached under `~/.isochrones/filters/svo/`):

```bash
python build_extinction_table.py                 # Fitzpatrick & Massa 2007
python build_extinction_table.py --curve f99 --rv 3.1   # alternative law
```

Defaults: photon-counting bandpass integral, 4500 K blackbody source SED
(`--sed`, `--teff` configurable). Each curve writes a provenance-stamped
`~/.isochrones/basti/extinction_table_<curve>.npz`; select the active one
with `isochrones.basti.extinction.set_extinction_curve()`. Bands missing
from the table fall back to approximate scalar coefficients with a logged
warning. NIR proxies: Johnson-Cousins J/H/K use the 2MASS curves; JC L uses
WISE W1.

## 6. Using the models

Fixed [α/Fe]:

```python
from isochrones.basti import get_ichrone_basti
ic = get_ichrone_basti(bands=["F090W", "F162M", "F200W"], afe=0.0,
                       age_range=(0.02, 14.5))
# parameter vector: [eep(row), log10(age), feh, distance_pc, AV]
Teff, logg, feh, mags = ic.interp_mag([1289, 10.0, -1.0, 776e3, 0.3], ic.bands)
```

`age_range` (Gyr) restricts which files are parsed and is part of the cache
tag — pass the SAME value everywhere to reuse caches. Pre-building with
diagnostics: `python build_basti_grids.py` (optional; grids also build
lazily on first use, exactly like stock MIST).

Variable [α/Fe] (α as a sixth fit parameter, reusing the MIST v2.5 fitter):

```python
from isochrones.basti.starmodel import get_ichrone_basti_alpha, BastiStarModel
from isochrones.mist.starmodel_v2p5 import StarModelV2p5
ic = get_ichrone_basti_alpha(bands=["F090W", "F162M"], age_range=(0.02, 14.5))
model = StarModelV2p5(ic, F090W=(21.3, 0.02), F162M=(19.9, 0.02))
model.fit()          # pymultinest; vector: eep, age, feh, afe, distance, AV
```

`BastiStarModel` fits at fixed α with the 5-parameter vector. The α scheme
has two regimes, split at the +0.4 grid's iron ceiling
(`ic.feh_quad_max`, ≈ +0.09, auto-read from the grid): below it, quadratic
Lagrange interpolation over (−0.2, 0, +0.4) with linear extrapolation to
+0.6; at or above it, linear through (−0.2, 0) only, NaN above α = +0.4.

### Worked example: synthesize and fit a star

Galactic-astronomy bandpasses — Gaia G, HST/ACS F814W, 2MASS J/H/Ks — plus a
parallax. Band tokens are exact (verified column maps): `G`,
`ACS_WFC_F814W`, `2MASS_J`, `2MASS_H`, `2MASS_Ks`. These touch three BaSTI
systems (gaia-dr3, acs, 2mass), so the first run builds a three-system grid
per α (a few minutes each; cached thereafter).

```python
import numpy as np
from isochrones.basti.starmodel import get_ichrone_basti_alpha
from isochrones.mist.starmodel_v2p5 import StarModelV2p5

BANDS = ["G", "ACS_WFC_F814W", "2MASS_J", "2MASS_H", "2MASS_Ks"]
ic = get_ichrone_basti_alpha(bands=BANDS, age_range=(0.02, 14.5))

# --- 1. truth: an upper-RGB halo giant at 8 kpc -------------------------
#     vector: [eep(row), log10(age/yr), [Fe/H], [a/Fe], distance_pc, AV]
truth = [1150, 10.0, -1.0, 0.30, 8000.0, 0.5]
_, _, _, mags = ic.interp_mag(truth, BANDS)

# --- 2. synthetic measurements: 0.02 mag errors; parallax 0.125+/-0.020 mas
rng = np.random.default_rng(42)
obs = {b: (float(m) + rng.normal(0, 0.02), 0.02) for b, m in zip(BANDS, mags)}
plx = (1000.0 / truth[4] + rng.normal(0, 0.020), 0.020)   # mas

# --- 3. fit ---------------------------------------------------------------
model = StarModelV2p5(ic, parallax=plx, **obs)
model.fit(n_live_points=400)
for p, t in zip(["eep", "age", "feh", "afe", "distance", "AV"], truth):
    s = model.samples[p]
    print(f"{p:>9}: truth {t:>8.3f}   fit {s.median():>8.3f} "
          f"+/- {s.std():.3f}")
```

The posterior medians should recover the truth within the quoted
uncertainties; `afe` = +0.30 sits inside the quadratic-interpolation regime
at [Fe/H] = −1.0 (well below `feh_quad_max`).

## 7. EEP conventions

Every isochrone has exactly 2100 rows whose 0-based row index equals the
Hidalgo et al. (2018) Table 4 normalized-track line − 1 — row number IS the
pseudo-EEP; no construction step exists. Anchors (importable as
`isochrones.basti.models.BASTI_EEP_ANCHORS`): ZAMS = 99, track turn-off =
359, RGB base = 489, RGB bump = 859/889, **TRGB = 1289**, quiescent core-He
burning from 1299, core-He exhaustion = 1949, early-AGB end = 2099.
Validated against ~55,000 files (20 Myr – 29.5 Gyr, all alphas). Typical
fitter bounds for evolved stars: `eep = (300, 2100)` (just below the
turn-off through the early AGB) — the analogue of MIST's (400, 1425). Note
the anchors are track-defined; along an isochrone the morphological feature
can sit a few rows later (measured isochrone MSTO: rows 359–403).

## 8. Differences from the MIST v2.5 extension

| Aspect | BaSTI (this readme) | MIST v2.5 (readme_mist_v25.md) |
|---|---|---|
| [α/Fe] nodes | three: −0.2, 0, +0.4; quadratic/linear scheme, 0.4–0.6 extrapolated | five: −0.2 … +0.6 step 0.2, native interpolation |
| [Fe/H] range | α-dependent ceilings: +0.44 (α=−0.2), +0.30 (α=0), **+0.09 (α=+0.4)** | −4.0 … +0.5, uniform for all α |
| [Fe/H] floor | ≈ −3.1 … −3.2 per α | −4.0 |
| EEP coordinate | row number (Hidalgo+18 Table 4), always 2100 rows | constructed EEPs (Dotter 2016), to 1721 |
| TP-AGB | **absent** — isochrones end at the early AGB | included (EEP 808–1409) |
| Extinction | precomputed R_X(band, A_V) table (fm07 default; swappable) | native BC-table axis |
| Data products | isochrone files only (206,655 files, 15 systems) | tracks + isochrones + per-system BC tables |
| Free-α support | ends at [Fe/H] = +0.30 (linear regime blends α=−0.2 and 0) | full range |

## 9. Scientific caveats

- **The (feh, α) support notch**: the +0.4 grid stops at [Fe/H] ≈ +0.09, so
  free-α fits switch to the linear (−0.2, 0) regime above that (flagged per
  star by the validation fitter), and free-α support ends entirely at
  [Fe/H] = +0.30 (the α=0 ceiling — its +0.45 node was never computed on the
  server). Only fixed α = −0.2 reaches [Fe/H] = +0.44.
- **α = 0.4–0.6 is linear extrapolation**, not interpolation. Flagged per
  star; treat α posteriors there with care.
- **No TP-AGB**: the brightest AGB stars cannot be represented; fits of
  such stars will differ from MIST by construction.
- **NIR filter proxies** in the extinction table: JC J/H/K ← 2MASS,
  JC L ← WISE W1 (few-mmag effect); `JC_BX`, `JC_Lprime`,
  `SkyMapper_u_leak` have no SVO curve and fall back to scalar coefficients.
- Nominal vs exact Z: tarball names carry nominal compositions; all
  metadata is read from file headers (exact Z), so [Fe/H] node values are
  exact.
