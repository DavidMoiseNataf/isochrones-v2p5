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
> - **Age** is `log10(age / yr)`. So `9.7` means ~5 Gyr, `10.1` means ~12.6 Gyr.
> - **Distance** is in parsecs; **Aᵥ** is V-band extinction in magnitudes.
> - **NIRCam** bands are bare (`F090W`, `F162M`, …); **HST** bands must be
>   system-qualified (`ACS_WFC_F475W`, `WFC3_UVIS_F390W`). Bare `F475W`/`F390W`
>   are ambiguous and will not resolve.

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

ic = get_ichrone_v2p5_iso(bands=["F090W", "F162M", "F200W"], afe=0.4)
model = SingleStarModel(ic,
                        F090W=(21.03, 0.02), F162M=(19.20, 0.02),
                        F200W=(18.98, 0.02), parallax=(0.5, 0.1))   # parallax in mas
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

# fixed alpha = +0.4, three NIRCam bands
ic_fixed = get_ichrone_v2p5_iso(bands=["F090W", "F162M", "F200W"], afe=0.4)

# alpha-interpolating grid spanning a JWST + HST band set
ic_alpha = get_ichrone_v2p5_alpha(
    bands=["ACS_WFC_F475W", "F090W", "F162M", "F200W", "F300M", "F356W", "F460M"]
)
```

The first call builds (or loads a cached) interpolator and can take a little time;
subsequent calls in the same session are fast. The α-interpolating object wraps
five fixed-α grids internally and interpolates their outputs linearly in α, which
adds only ~1.4× wall-clock over a single fixed-α evaluation.

---

## 4. Interpolating stellar properties

Given a point in the grid's parameter space, `interp_value` returns whatever
physical columns you ask for. For a **fixed-α** interpolator the point is
`[eep, age, feh]`:

```python
# EEP 350 (lower RGB), log-age 9.7 (~5 Gyr), [Fe/H] = -0.5
mass, radius, Teff, logg, logL = ic_fixed.interp_value(
    [350, 9.7, -0.5], ["mass", "radius", "Teff", "logg", "logL"]
)
print(mass, Teff, logg)
```

Any column the grid stores can be requested (`mass`, `radius`, `Teff`, `logg`,
`logL`, `feh`, `age`, …). For the **α-interpolating** grid the point carries the
extra α slot, `[eep, age, feh, afe]`.

A note on EEP (Equivalent Evolutionary Phase): it is a monotonic stage index that
replaces "mass" as the primary track coordinate so that the same EEP means the
same evolutionary stage across different masses and metallicities. Rough
landmarks: ZAMS ≈ 202, main-sequence turnoff ≈ 454, RGB tip ≈ 605, with the v2.5
grid extending to 1721.

---

## 5. Predicting magnitudes (synthetic photometry)

`interp_mag` is the photometric workhorse: it places a model star at a distance,
reddens it, and returns apparent magnitudes in your bands. It also returns the
star's `(Teff, logg, feh)` as a convenience. This is how you generate synthetic
observed properties for a star of known physical parameters.

For a **fixed-α** interpolator the vector is `[eep, age, feh, distance, AV]`:

```python
# a star at 10 kpc with AV = 0.3
Teff, logg, feh, mags = ic_fixed.interp_mag(
    [350, 9.7, -0.5, 10_000.0, 0.3], ["F090W", "F162M", "F200W"]
)
print(dict(zip(["F090W", "F162M", "F200W"], mags)))
```

For the **α-interpolating** grid, insert α right after `[Fe/H]`, giving
`[eep, age, feh, afe, distance, AV]`:

```python
Teff, logg, feh, mags = ic_alpha.interp_mag(
    [350, 9.7, -0.5, 0.4, 10_000.0, 0.3],
    ["ACS_WFC_F475W", "F090W", "F300M"]
)
```

Sweeping one axis while holding the others fixed is the easy way to see, e.g., how
a water-sensitive band like `F300M` responds to α at fixed Teff — the leverage
the variable-α fit relies on.

---

## 6. Bolometric corrections

The magnitudes above come from a v2.5 bolometric-correction grid,
`MISTBolometricCorrectionGridV2p5`, which is band-aware and self-contained: it
resolves NIRCam, HST/ACS, and HST/WFC3 filters on its own and never depends on the
parent package's band tables. You normally never touch it directly — the factories
wire it in — but you can instantiate one to inspect or extend the band set:

```python
from isochrones.mist.bc_v2p5 import MISTBolometricCorrectionGridV2p5
bc = MISTBolometricCorrectionGridV2p5(["F090W", "ACS_WFC_F475W"])
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

Observations are passed as `band=(value, uncertainty)`. Distance information can
come from a `parallax=(mas, err)` observation or from a distance prior (next
step).

```python
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_alpha
from isochrones.mist.starmodel_v2p5 import StarModelV2p5

bands = ["ACS_WFC_F475W", "F090W", "F162M", "F200W", "F300M", "F356W", "F460M"]
ic = get_ichrone_v2p5_alpha(bands=bands)

obs = {  # one star's apparent magnitudes
    "ACS_WFC_F475W": 23.81, "F090W": 21.40, "F162M": 19.95, "F200W": 19.61,
    "F300M": 19.83, "F356W": 19.55, "F460M": 19.40,
}
model = StarModelV2p5(ic, **{b: (m, 0.02) for b, m in obs.items()})
```

### 7c. Bounds and priors

`set_bounds` clips the prior support of any parameter to a range — useful for
pinning distance to a known value (a star in M31, say) or for restricting age to
old populations. Pinning a parameter to a near-zero-width window effectively fixes
it:

```python
mu = 24.38                       # M31 distance modulus
dist_pc = 10 ** (mu / 5 + 1)     # -> ~7.5e5 pc
model.set_bounds(
    distance=(dist_pc * 0.99, dist_pc * 1.01),   # narrow box around the known distance
    age=(8.0, 10.13),                            # log-age: 0.1-13.5 Gyr
    AV=(0.0, 1.5),                               # extinction ceiling
    # afe=(...) omitted -> alpha free over the grid's full alpha range
)
```

Priors live in `model._priors`, keyed by parameter name; assign a prior object to
replace the default. To impose a Gaussian distance prior on top of the box above:

```python
from isochrones.priors import GaussianPrior
model._priors["distance"] = GaussianPrior(
    dist_pc, dist_pc * 0.003, bounds=(dist_pc * 0.99, dist_pc * 1.01)
)
```

Other prior classes from `isochrones.priors` include `FlatPrior`,
`FlatLogPrior`, and `PowerLawPrior` (the default IMF-like mass/EEP prior). To
**fix α** instead of fitting it, pin it the same way bounds pin distance:
`model.set_bounds(afe=(-0.01, 0.01))`.

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
print(ds[["mass", "Teff", "logg"]].median())              # physical params
# predicted-minus-observed residual in a band:
print(obs["F300M"] - ds["F300M_mag"].median())
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

---

## 8. Using multiple CPUs

There are two distinct ways to spread work across cores. Pick based on whether you
have **one expensive fit** or **many independent fits**.

### 8a. One fit, many cores — MultiNest via MPI

MultiNest is natively MPI-parallel: it distributes the live-point likelihood
evaluations across ranks. If `pymultinest`/MultiNest were built with MPI support
and `mpi4py` is installed, you simply launch your *unmodified* fitting script
under `mpiexec`:

```bash
mpiexec -n 4 python my_single_fit.py     # one fit, 4 cores
```

No code change is required — `pymultinest` detects the MPI communicator. This is
the right tool for a single high-`n_live_points` fit. It does **not** help much
for many small fits, where the per-fit fixed overhead dominates.

### 8b. Many fits, many cores — one process per star

For a catalog of stars, the fits are independent, so the efficient pattern is
"embarrassingly parallel": a `multiprocessing` pool with **one star per worker**.
Two rules keep it fast and correct: build the (expensive) interpolator **once per
worker** in an initializer, and pin the math libraries to a single thread each so
N workers don't oversubscribe the CPU.

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"          # set BEFORE importing numpy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import tempfile
import multiprocessing as mp

BANDS = ["ACS_WFC_F475W", "F090W", "F162M", "F200W", "F300M", "F356W", "F460M"]
_IC = None                                    # per-worker grid, filled by the initializer

def _init():
    global _IC
    from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_alpha
    _IC = get_ichrone_v2p5_alpha(bands=BANDS)   # built once, reused for every star this worker handles

def _fit_one(star):                           # star: {band: (mag, err), ...}
    from isochrones.mist.starmodel_v2p5 import StarModelV2p5
    base = os.path.join(tempfile.mkdtemp(), "c-")   # unique basename per fit -> no file races
    m = StarModelV2p5(_IC, **star)
    m.fit(n_live_points=500, basename=base, verbose=False, overwrite=True)
    return float(m.samples["afe"].median())

if __name__ == "__main__":
    stars = [...]                             # list of per-star obs dicts
    ctx = mp.get_context("spawn")             # 'spawn' is the safe context with MultiNest
    with ctx.Pool(6, initializer=_init) as pool:     # up to 6 cores
        afe_values = list(pool.imap_unordered(_fit_one, stars, chunksize=1))
```

This scales nearly linearly with core count until you saturate memory or I/O.
Two combinable knobs trade throughput against per-star quality: the pool size (how
many stars run at once) and `n_live_points` (how hard each fit works).

> Do **not** combine 8a and 8b — running an MPI build under a `multiprocessing`
> pool will oversubscribe and can deadlock. Use the pool with a non-MPI build, or
> use `mpiexec` on a single fit, not both.

---

## 9. Band naming and scientific caveats

**Band naming.** NIRCam filters are bare (`F090W`, `F162M`, `F300M`, …) and resolve
to JWST automatically. HST filters must be system-qualified — `ACS_WFC_F475W`,
`WFC3_UVIS_F390W` — because bare `F475W`/`F390W` exist in more than one HST system
and are ambiguous.

**Caveats worth reading before trusting a number** (expanded in the repository
README):

- **α is only as good as its photometric leverage.** Broadband `[α/Fe]` is
  indirect; the absolute scale has a systematic floor from model BCs and
  zero-points, and α is partly degenerate with `[Fe/H]` along the total-metallicity
  ridge. Relative, star-to-star α is more robust than the absolute value.
- **Avoid the TP-AGB for abundance work.** Above the RGB tip (EEP ≳ 605), dredge-up
  changes surface C/O, so molecular-band α proxies no longer track natal `[α/Fe]`.
  Restrict abundance interpretation to the RGB.
- **Ages of giants are prior-dominated** without a main-sequence turnoff.
- **The extinction law matters.** The fixed-α collapse freezes `Rv = 3.1`; a wrong
  reddening curve produces wavelength-monotonic residuals the fit will absorb into
  Aᵥ.

---

## See also

- `README.md` — design, the full factory/class table, and the v2.5-vs-v1.2 change
  list.
- `examples/fit_m31_giants_v25.py` — two stars, predicted-band corner plots.
- `examples/fit_m31_sample_v25.py` — a sample, with cross-star systematics
  diagnostics.
- Upstream `isochrones` docs: <https://isochrones.readthedocs.io/en/latest/>
