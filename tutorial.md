# Tutorial: using `isochrones-v2p5`

A hands-on walkthrough of the MIST **v2.5** (α-enhanced) extension to Tim Morton's
[`isochrones`](https://github.com/timothydmorton/isochrones) package. It is meant
to be read top to bottom with a Python session open, and it loosely follows the
upstream [ReadTheDocs](https://isochrones.readthedocs.io/en/latest/) progression —
install, access the grids, interpolate, predict magnitudes, then fit a star —
adapted to the v2.5 modules. The two upstream chapters on *multiple star systems*
and *simulating stellar populations* are intentionally left out.

Throughout, the one capability v2.5 adds over stock `isochrones` is that
**`[α/Fe]` can be a free parameter**, interpolated across the five MIST α-nodes
`{-0.2, 0.0, +0.2, +0.4, +0.6}` alongside the usual EEP, age, `[Fe/H]`, distance,
and Aᵥ.

> **Conventions used everywhere below**
> - **Age** is `log10(age / yr)`. So `9.6` means ~4 Gyr, `10.1` means ~12.6 Gyr.
> - **Distance** is in parsecs; **Aᵥ** is V-band extinction in magnitudes.
> - **Gaia** and **2MASS** bands use their standard short names (`G`, `BP`, `RP`;
>   `J`, `H`, `Ks`); **NIRCam** bands are bare (`F090W`, `F162M`, …); **HST** bands
>   must be system-qualified (`ACS_WFC_F475W`, `WFC3_UVIS_F606W`). Bare HST names
>   like `F606W` are ambiguous (both ACS and WFC3 have them) and will not resolve.

> **The running example.** Every code block below fits the same star: a **G-type
> dwarf about 6 kpc away**, with a **Gaia parallax good to 10%** and apparent
> magnitudes in **Gaia G**, **HST/WFC3 F606W and F814W**, and **2MASS J, H, Ks**.
> Its true line-of-sight extinction is **Aᵥ = 2.5**, but we will fit Aᵥ under a
> **flat prior from 0 to 6 mag** and let the data recover it. The illustrative
> magnitudes are:
>
> | G | F606W | F814W | J | H | Ks | parallax |
> | --- | --- | --- | --- | --- | --- | --- |
> | 20.63 | 20.84 | 19.49 | 18.28 | 17.67 | 17.46 | 0.167 ± 0.017 mas |

---

## 1. Install

The `_v2p5` modules `import` from inside the `isochrones` package, so they must
sit beside it. Install `isochrones` first, then drop the four core modules into
its `mist/` directory:

```bash
# locate your installed isochrones package
python -c "import isochrones, os; print(os.path.dirname(isochrones.__file__))"

# copy the four v2.5 modules into that package's mist/ subdirectory
cp bc_v2p5.py isochrone_v2p5.py models_v2p5.py starmodel_v2p5.py  <that-path>/mist/
```

Nothing in `isochrones` is modified, and the v1.2 grids stay the default — your
existing scripts behave exactly as before unless they explicitly ask for a
`_v2p5` object.

**Dependencies.** The grids themselves only need `numpy` / `pandas` / `pytables`.
Fitting needs **MultiNest** and **`pymultinest`**; the example corner plots need
`corner`. Tested against Python 3.10 and `pandas < 2.0`, matching a typical
`isochrones` conda environment.

### Getting the v2.5 data

`download_mist_v25.py` is stdlib-only and pulls each component on request:

```bash
python download_mist_v25.py --what tracks                       # EEP evolutionary tracks
python download_mist_v25.py --what bc                           # bolometric corrections
python download_mist_v25.py --what full_isos                    # precomputed isochrones
python download_mist_v25.py --what bc --bc-systems JWST HST_ACS_WFC HST_WFC3
python download_mist_v25.py --reorganize-existing               # migrate an existing layout
```

Data lands under `$ISOCHRONES` (default `~/.isochrones`), with v2.5 kept separate
from v1.2.

### Installing MultiNest (needed only for fitting)

MultiNest is a small Fortran library that `pymultinest` wraps. The usual route is
to build `MultiNest` from source (CMake), put `libmultinest.*` on your library
path (`LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS), and
`pip install pymultinest`. You can confirm it imports with:

```bash
python -c "import pymultinest; print('pymultinest OK')"
```

---

## 2. Quick start

The shortest end-to-end fit, with `[α/Fe]` held fixed (a drop-in replacement for a
v1.2 fit):

```python
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_iso
from isochrones import SingleStarModel

ic = get_ichrone_v2p5_iso(bands=["G", "J", "H", "Ks"], afe=0.0)
model = SingleStarModel(ic,
                        G=(20.63, 0.02), J=(18.28, 0.02),
                        H=(17.67, 0.02), Ks=(17.46, 0.02),
                        parallax=(0.167, 0.017))     # parallax in mas (~6 kpc, 10%)
model.set_bounds(AV=(0, 6))                          # flat extinction prior, 0-6 mag
model.fit()
print(model.derived_samples[["mass", "Teff", "logg"]].median())
```

The rest of the tutorial unpacks every piece of this and then shows the
**variable-α** path, which is the reason the package exists.

---

## 3. Accessing the model grids

There are three entry-point factories. Two give you a **fixed-α** interpolator
(identical layout and speed to v1.2); the third gives you the **α-interpolating**
object used when `[α/Fe]` is a fit parameter.

| Factory | Returns | Parameter axis |
| --- | --- | --- |
| `get_ichrone_v2p5_iso(bands, afe)` | isochrone interpolator at fixed α | `[eep, age, feh]` |
| `get_ichrone_v2p5(bands, afe)` | evolution-track interpolator at fixed α | `[eep, age, feh]` |
| `get_ichrone_v2p5_alpha(bands)` | α-interpolating isochrone (α free) | `[eep, age, feh, afe]` |

```python
from isochrones.mist.isochrone_v2p5 import (
    get_ichrone_v2p5_iso,     # fixed-alpha isochrones
    get_ichrone_v2p5,         # fixed-alpha evolution tracks
    get_ichrone_v2p5_alpha,   # alpha as a free axis
)

# fixed solar alpha, the Gaia + 2MASS subset
ic_fixed = get_ichrone_v2p5_iso(bands=["G", "J", "H", "Ks"], afe=0.0)

# alpha-interpolating grid spanning the full Gaia + HST + 2MASS band set
ic_alpha = get_ichrone_v2p5_alpha(
    bands=["G", "WFC3_UVIS_F606W", "WFC3_UVIS_F814W", "J", "H", "Ks"]
)
```

The first call builds (or loads a cached) interpolator and can take a little time;
subsequent calls in the same session are fast. The α-interpolating object wraps
five fixed-α grids internally and interpolates their outputs linearly in α, which
adds only ~1.4× wall-clock over a single fixed-α evaluation.

### 3b. Peeking at the grid itself

Behind every interpolator sits a big pandas DataFrame indexed by
`(log10_age, [Fe/H], EEP)` — the actual model grid. You can inspect it
directly, which is the quickest way to see what columns exist and what the
node spacing looks like:

```python
df = ic_fixed.model_grid.df
print(df.shape)                    # (rows, columns)
print(df.index.names)              # ['log10_isochrone_age_yr', 'feh', 'EEP']
print(list(df.columns))            # every physical + magnitude column

# pull ONE on-node isochrone out of the grid with a cross-section:
iso_node = df.xs((9.6, 0.0), level=(0, 1))   # log-age 9.6, [Fe/H] = 0.0
print(iso_node[["mass", "Teff", "logg"]].head())
```

`.xs(...)` only works at exact grid nodes; for arbitrary ages and
metallicities, interpolate instead (§4b).

---

## 4. Interpolating stellar properties

Given a point in the grid's parameter space, `interp_value` returns whatever
physical columns you ask for. For a **fixed-α** interpolator the point is
`[eep, age, feh]`:

```python
# EEP 330 (a main-sequence G dwarf), log-age 9.6 (~4 Gyr), [Fe/H] = 0.0
mass, radius, Teff, logg, logL = ic_fixed.interp_value(
    [330, 9.6, 0.0], ["mass", "radius", "Teff", "logg", "logL"]
)
print(mass, Teff, logg)   # ~1 Msun, ~5800 K, ~4.4
```

Any column the grid stores can be requested (`mass`, `radius`, `Teff`, `logg`,
`logL`, `feh`, `age`, …). For the **α-interpolating** grid the point carries the
extra α slot, `[eep, age, feh, afe]`.

A note on EEP (Equivalent Evolutionary Phase): it is a monotonic stage index that
replaces "mass" as the primary track coordinate so that the same EEP means the
same evolutionary stage across different masses and metallicities. Rough
landmarks: ZAMS ≈ 202, mid main sequence ≈ 330 (our G dwarf), main-sequence
turnoff ≈ 454, RGB tip ≈ 605, with the v2.5 grid extending to 1721.

### 4b. Extracting a whole interpolated isochrone

`.isochrone(age, feh)` returns a complete isochrone — every EEP row — as a
DataFrame, interpolated to ANY age and metallicity (not just grid nodes),
with predicted magnitudes included as `<band>_mag` columns:

```python
iso = ic_fixed.isochrone(9.53, 0.12)      # log-age 9.53, [Fe/H] = +0.12
print(iso[["mass", "Teff", "logg", "G_mag", "Ks_mag"]].head())
```

A classic sanity check (adapted from the upstream docs' visualization demo):
an interpolated isochrone should fall neatly between its bracketing grid
nodes. In matplotlib:

```python
import matplotlib.pyplot as plt

df = ic_fixed.model_grid.df
iso1 = df.xs((9.5, 0.00), level=(0, 1))    # on-node
iso2 = df.xs((9.5, 0.25), level=(0, 1))    # on-node
iso3 = ic_fixed.isochrone(9.5, 0.12)       # interpolated between them

for iso, lab in [(iso1, "[Fe/H]=0.00"), (iso2, "[Fe/H]=0.25"),
                 (iso3, "[Fe/H]=0.12 (interpolated)")]:
    plt.plot(iso["logTeff"], iso["logL"], label=lab)
plt.gca().invert_xaxis(); plt.xlabel("logTeff"); plt.ylabel("logL")
plt.legend(); plt.show()
```

---

## 5. Predicting magnitudes (synthetic photometry)

`interp_mag` is the photometric workhorse: it places a model star at a distance,
reddens it, and returns apparent magnitudes in your bands. It also returns the
star's `(Teff, logg, feh)` as a convenience. This is how you generate synthetic
observed properties for a star of known physical parameters.

For a **fixed-α** interpolator the vector is `[eep, age, feh, distance, AV]`:

```python
# our G dwarf: 6 kpc, AV = 2.5
Teff, logg, feh, mags = ic_fixed.interp_mag(
    [330, 9.6, 0.0, 6000.0, 2.5], ["G", "J", "H", "Ks"]
)
print(dict(zip(["G", "J", "H", "Ks"], mags)))
```

For the **α-interpolating** grid, insert α right after `[Fe/H]`, giving
`[eep, age, feh, afe, distance, AV]`:

```python
Teff, logg, feh, mags = ic_alpha.interp_mag(
    [330, 9.6, 0.0, 0.0, 6000.0, 2.5],
    ["G", "WFC3_UVIS_F606W", "J"]
)
```

Sweeping one axis while holding the others fixed is the easy way to build
intuition — e.g. how the optical–NIR colors redden as you push Aᵥ from 0 to 6, or
how weakly the broadband magnitudes respond to α (broadband optical/NIR carries
little α leverage, which is exactly why α is hard to pin from a star like this one;
see the caveats in §9).

### 5b. Vectorized calls: the interpolator is callable

Every interpolator is itself callable, taking arrays and returning a
DataFrame of all physical columns plus `<band>_mag` magnitudes — the
one-line way to generate a synthetic CMD or a whole sequence of stars:

```python
import numpy as np

# a synthetic isochrone ribbon: EEPs 250..600 at 4 Gyr, solar Z, 6 kpc, AV=2.5
eeps = np.arange(250, 600, 5)
stars = ic_fixed(eeps, 9.6, 0.0, 6000.0, 2.5)     # -> DataFrame, one row per EEP
plt.scatter(stars["G_mag"] - stars["Ks_mag"], stars["G_mag"], s=4)
plt.gca().invert_yaxis(); plt.xlabel("G - Ks"); plt.ylabel("G"); plt.show()
```

For the α-interpolating object the call carries the extra α slot:
`ic_alpha(eeps, 9.6, 0.0, 0.3, 6000.0, 2.5)`.

### 5c. From mass to EEP: `generate()` and `get_eep()` (evolution tracks)

If you know masses rather than EEPs — the natural situation when simulating
a population — the **evolution-track** interpolator can invert
(mass, age, feh) → EEP for you. `generate()` wraps the inversion and the
property lookup in one call (adapted from the upstream docs):

```python
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5
track = get_ichrone_v2p5(bands=["G", "J", "Ks"], afe=0.0)

track.generate([0.81, 0.91, 1.01], 9.51, 0.01)        # 3 stars by mass
print(track.get_eep(1.01, 9.51, 0.01))                # the EEP it solved for
```

The default inversion is a fast interpolation, accurate on the main sequence
but sloppier for evolved stars (the fundamental reason fitting always uses
EEP as the sampled parameter). For precise EEPs, pass `accurate=True` to
either method — a real function minimization, ~1000× slower per star but
still fast in absolute terms:

```python
track.get_eep(1.01, 9.51, 0.01, accurate=True)
track.generate([0.81, 0.91, 1.01], 9.51, 0.01, accurate=True)
```

Because the default mode is vectorized and fast (~10⁵ stars/second), drawing
masses from an IMF and calling `generate()` once is a perfectly good way to
paint a synthetic population. Note these two methods live on the **track**
interpolator (`get_ichrone_v2p5`) only — the isochrone-grid objects raise
`NotImplementedError` for them.

---

## 6. Bolometric corrections

The magnitudes above come from a v2.5 bolometric-correction grid,
`MISTBolometricCorrectionGridV2p5`, which is band-aware and self-contained: it
resolves NIRCam, HST/ACS, and HST/WFC3 filters on its own and never depends on the
parent package's band tables. You normally never touch it directly — the factories
wire it in — but you can instantiate one to inspect or extend the band set:

```python
from isochrones.mist.bc_v2p5 import MISTBolometricCorrectionGridV2p5
bc = MISTBolometricCorrectionGridV2p5(["G", "WFC3_UVIS_F606W", "J"])
```

The v2.5 BC tables are indexed by `(log10 Teff, logg, [Fe/H], [α/Fe], Aᵥ, Rv)`.
The fixed-α paths collapse the `[α/Fe]` and `Rv` axes (freezing `Rv = 3.1`) to
recover the exact v1.2 interpolator layout; the α-interpolating path keeps α live.

---

## 7. Fitting a single star

This is the core workflow. You build an observation model, optionally constrain
its parameters with bounds and priors, sample with MultiNest, and read the
posterior.

### 7a. Choose fixed-α or variable-α

- **Fixed α** — use `SingleStarModel` (from stock `isochrones`) with a fixed-α
  interpolator. Parameter vector: `[eep, age, feh, distance, AV]`.
- **Variable α** — use `StarModelV2p5` (from this package) with an
  α-interpolating interpolator. Parameter vector:
  `[eep, age, feh, afe, distance, AV]`.

Only fit α freely when your bands actually constrain it (e.g. a molecular,
α-sensitive band is present); otherwise α is prior-dominated and you are better
off fixing it.

### 7b. Define the model

Observations are passed as `band=(value, uncertainty)`, plus a
`parallax=(mas, err)` term for the distance. (Alternatively, distance can come
from a prior — see the next step.)

```python
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_alpha
from isochrones.mist.starmodel_v2p5 import StarModelV2p5

bands = ["G", "WFC3_UVIS_F606W", "WFC3_UVIS_F814W", "J", "H", "Ks"]
ic = get_ichrone_v2p5_alpha(bands=bands)

obs = {  # the G dwarf's apparent magnitudes
    "G": 20.63, "WFC3_UVIS_F606W": 20.84, "WFC3_UVIS_F814W": 19.49,
    "J": 18.28, "H": 17.67, "Ks": 17.46,
}
model = StarModelV2p5(ic,
                      parallax=(0.167, 0.017),                    # ~6 kpc, 10%
                      **{b: (m, 0.02) for b, m in obs.items()})
```

(With no α-sensitive band in this set, α here is essentially prior-dominated; the
fit is shown to exercise the variable-α machinery, but for a star like this you
would normally use the fixed-α path from §2 instead.)

### 7c. Bounds and priors

`set_bounds` clips the prior support of any parameter to a range. For our star the
key one is extinction: we want a **flat prior on Aᵥ from 0 to 6 mag**. Aᵥ's
default prior is already uniform, so bounding it to `(0, 6)` realizes exactly that
flat prior — the fit then has to pull the true value (2.5) out of the photometry on
its own:

```python
model.set_bounds(
    AV=(0.0, 6.0),       # flat extinction prior, 0-6 mag
    age=(8.0, 10.13),    # optional: restrict log-age to 0.1-13.5 Gyr
    # afe omitted -> alpha free over the grid's full range
)
```

Here distance is constrained by the `parallax` observation we passed in §7b, so
there is nothing to pin. If instead you had an independently known distance (a
star in a cluster, say) and no parallax, you could fix it with a narrow box plus a
Gaussian prior, keyed by name in `model._priors`:

```python
from isochrones.priors import GaussianPrior
d = 6000.0                                   # known distance in pc
model.set_bounds(distance=(d * 0.97, d * 1.03))
model._priors["distance"] = GaussianPrior(d, 0.1 * d, bounds=(d * 0.97, d * 1.03))
```

Other prior classes from `isochrones.priors` include `FlatPrior`, `FlatLogPrior`,
and `PowerLawPrior` (the default IMF-like mass/EEP prior). To **fix α** instead of
fitting it, pin it the same way: `model.set_bounds(afe=(-0.01, 0.01))`.

### 7d. The `model.fit(...)` call — every argument

Here is the call in the form used in practice, with every argument explained:

```python
import os, tempfile
basename = os.path.join(tempfile.mkdtemp(), "c-")   # unique output prefix per fit

model.fit(
    n_live_points=500,     # MultiNest live points (see below)
    basename=basename,     # where MultiNest writes its output files
    verbose=False,         # suppress MultiNest's progress printing
    overwrite=True,        # start fresh; do not resume an old chain at basename
)
```

- **`n_live_points`** — the number of "live points" MultiNest maintains while
  nested-sampling the posterior. This is the single most important accuracy/cost
  knob: more live points sample fine structure and multimodality better and give a
  more reliable Bayesian evidence, at a roughly proportional increase in run time.
  A few hundred (200–650) is typical for these low-dimensional stellar fits; raise
  it if the posterior looks ragged or the evidence is noisy.

- **`basename`** — the path **prefix** MultiNest uses for all of its output files
  (live points, posterior samples, the `stats.dat` evidence summary, …). Two fits
  that share a basename will clobber each other's files, so when you run many fits
  — especially in parallel — give every fit a **unique** basename. Putting it in a
  fresh temporary directory per star, as above, is the simplest safe pattern.

- **`verbose`** — whether MultiNest prints its per-iteration progress to stdout.
  `False` keeps logs clean, which matters when fitting in bulk.

- **`overwrite`** — if `True`, ignore and overwrite any existing chain found at
  `basename` and fit from scratch. If `False`, MultiNest may *resume* from a
  partial chain it finds there, which is convenient for one long fit but dangerous
  in batch runs where a stale file could be silently reused.

A few other arguments you may reach for:

- **`refit=True`** — re-run even if this model already holds cached samples in
  memory (otherwise a second `.fit()` call may be a no-op).
- **Extra keyword arguments are forwarded to `pymultinest.run`.** The useful ones
  are `sampling_efficiency` (lower → more thorough, slower; ~0.3 is a good default
  for evidence-quality runs, ~0.8 for parameter estimation) and
  `evidence_tolerance` (the stopping threshold on the evidence; smaller → longer,
  more precise). For example:
  `model.fit(n_live_points=500, sampling_efficiency=0.3, evidence_tolerance=0.3)`.

### 7e. Reading the results

After fitting, two DataFrames hold the posterior, plus the evidence:

```python
s  = model.samples            # raw fit parameters, incl. 'afe' for StarModelV2p5
ds = model.derived_samples    # physical/photometric quantities, incl. predicted *_mag

print(s["afe"].median())                                  # inferred [alpha/Fe]
print(s["AV"].median())                                   # recovered extinction (truth: 2.5)
print(ds[["mass", "Teff", "logg"]].median())              # physical params
# predicted-minus-observed residual in a band:
print(obs["G"] - ds["G_mag"].median())
```

The Bayesian log-evidence (`lnZ`), handy for model comparison or for flagging
poorly-fit stars, comes from MultiNest's stats file:

```python
import pymultinest
stats = pymultinest.Analyzer(
    n_params=model.n_params, outputfiles_basename=basename
).get_stats()
lnZ = stats["global evidence"]
```

(Read `lnZ` **before** deleting the temporary directory that holds `basename`.)

A corner plot of the posterior is one import away (`pip install corner`):

```python
import corner
params = ["eep", "age", "feh", "afe", "distance", "AV"]
fig = corner.corner(model.samples[params], labels=params,
                    quantiles=[0.16, 0.5, 0.84], show_titles=True)
fig.savefig("star_corner.png", dpi=150)
```

The Aᵥ panel is the one to enjoy for our running example: a clean posterior
bump near the true 2.5 mag, recovered from photometry alone under the flat
0–6 prior.

---

## 8. Using multiple CPUs

When you have many stars to fit, the fits are independent, so the natural way to
use more cores is to run **one star per worker** with a `multiprocessing` pool.
Two things keep it fast and correct: each worker builds the (expensive)
interpolator **once** in an initializer and reuses it, and each fit gets its own
`basename` so the workers never clash over output files.

```python
import os
# Limit each worker to one math-library thread (set before importing numpy),
# so the workers don't oversubscribe the cores.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import tempfile
import multiprocessing as mp

BANDS = ["G", "WFC3_UVIS_F606W", "WFC3_UVIS_F814W", "J", "H", "Ks"]

interpolator = None   # each worker fills this once, in init_worker()

def init_worker():
    global interpolator
    from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_alpha
    interpolator = get_ichrone_v2p5_alpha(bands=BANDS)

def fit_one_star(obs):                    # obs = {band: (mag, err), ...}
    from isochrones.mist.starmodel_v2p5 import StarModelV2p5
    basename = os.path.join(tempfile.mkdtemp(), "fit-")   # unique per star
    model = StarModelV2p5(interpolator, **obs)
    model.fit(n_live_points=500, basename=basename, verbose=False, overwrite=True)
    return float(model.samples["afe"].median())

if __name__ == "__main__":
    stars = [...]                         # a list of per-star observation dicts
    pool = mp.get_context("spawn").Pool(6, initializer=init_worker)   # 6 cores
    afe_values = pool.map(fit_one_star, stars)
    pool.close()
```

Throughput scales nearly linearly with the number of workers. The two knobs to
balance are the pool size (how many stars run at once) and `n_live_points` (how
hard each individual fit works).

---

## 9. Band naming and scientific caveats

**Band naming.** Gaia and 2MASS filters use their standard short names (`G`, `BP`,
`RP`; `J`, `H`, `Ks`). NIRCam filters are bare (`F090W`, `F162M`, …) and resolve to
JWST automatically. HST filters must be system-qualified — `ACS_WFC_F475W`,
`WFC3_UVIS_F606W` — because bare names like `F606W` exist in more than one HST
system and are ambiguous. Whichever system you use, its bolometric-correction
tables must be present in your downloaded BC set; if a name doesn't resolve, check
the column names your v2.5 BC grid actually exposes for that system.

**Caveats worth reading before trusting a number** (expanded in the repository
README):

- **α is only as good as its photometric leverage.** Broadband `[α/Fe]` is
  indirect; the absolute scale has a systematic floor from model BCs and
  zero-points, and α is partly degenerate with `[Fe/H]` along the total-metallicity
  ridge. With only broadband optical/NIR (as in this tutorial's example), α has
  little leverage and is best fixed; it becomes informative when an α-sensitive
  band (e.g. a water- or CN-sensitive filter) is included. Relative, star-to-star α
  is more robust than the absolute value.
- **Avoid the TP-AGB for abundance work.** Above the RGB tip (EEP ≳ 605), dredge-up
  changes surface C/O, so molecular-band α proxies no longer track natal `[α/Fe]`.
  Restrict abundance interpretation to the RGB.
- **Ages of unevolved or evolved stars can be prior-dominated.** A clear
  main-sequence turnoff carries most of the age information; far from it (lower main
  sequence or giant branch) the age posterior leans on the prior.
- **The extinction law matters.** The fixed-α collapse freezes `Rv = 3.1`; a wrong
  reddening curve produces wavelength-monotonic residuals the fit will absorb into
  Aᵥ.

---

## 10. The same workflow with BaSTI

Everything above has a BaSTI-flavored twin: this repository also provides the
BaSTI-IAC O1D1E1 models as `isochrones.basti` (installation, data download,
and full reference in [readme_basti.md](readme_basti.md)). The workflow is
identical; the conventions differ:

| Convention | MIST v2.5 (this tutorial) | BaSTI |
| --- | --- | --- |
| Gaia bands | `G`, `BP`, `RP` | `G`, `BP`, `RP` (same) |
| 2MASS bands | `J`, `H`, `Ks` | `2MASS_J`, `2MASS_H`, `2MASS_Ks` |
| HST bands | `WFC3_UVIS_F606W`, … | same tokens |
| EEP coordinate | constructed EEPs; G dwarf ≈ 330, RGB tip = 605 | file row number; G dwarf ≈ 300, TRGB = 1289 |
| α nodes | five, −0.2 … +0.6, native interpolation | three (−0.2, 0, +0.4); quadratic/linear regimes, 0.4–0.6 extrapolated |
| Extinction | native BC-table axis | precomputed R\_X(band, Aᵥ) table (build once) |
| `generate()` / `get_eep()` | available on the track interpolator | not available (isochrone grid only — work in EEP) |

The running example, refit with BaSTI (note the 2MASS token change and the
`age_range` kwarg, which selects and caches the grid age window):

```python
from isochrones.basti.starmodel import get_ichrone_basti_alpha
from isochrones.mist.starmodel_v2p5 import StarModelV2p5   # same fitter class

bands = ["G", "WFC3_UVIS_F606W", "WFC3_UVIS_F814W",
         "2MASS_J", "2MASS_H", "2MASS_Ks"]
ic_b = get_ichrone_basti_alpha(bands=bands, age_range=(0.02, 14.5))

obs = {"G": 20.63, "WFC3_UVIS_F606W": 20.84, "WFC3_UVIS_F814W": 19.49,
       "2MASS_J": 18.28, "2MASS_H": 17.67, "2MASS_Ks": 17.46}
model = StarModelV2p5(ic_b,
                      parallax=(0.167, 0.017),
                      **{b: (m, 0.02) for b, m in obs.items()})
model.set_bounds(AV=(0, 6), eep=(100, 2100))
model.fit(n_live_points=500)
print(model.samples[["feh", "afe", "AV"]].median())
```

`.isochrone(age, feh)`, the vectorized callable, and `model_grid.df` all work
exactly as in §§3b–5b. Fitting the same star with both libraries and
differencing the posteriors is the cleanest way to measure model systematics
— see readme_basti.md §8 for the known differences.

---

## See also

- `README.md` — design, the full factory/class table, and the v2.5-vs-v1.2 change
  list.
- `examples/fit_m31_giants_v25.py` — two stars, predicted-band corner plots.
- `examples/fit_m31_sample_v25.py` — a sample, with cross-star systematics
  diagnostics.
- `readme_mist_v25.md` / `readme_basti.md` — the two libraries documented in
  parallel (install, data, extinction, usage, EEP conventions, differences,
  caveats).
- Upstream `isochrones` docs: <https://isochrones.readthedocs.io/en/latest/>
