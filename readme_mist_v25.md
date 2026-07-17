# MIST v2.5 support for `isochrones` (additive extension)

## 1. Overview and model physics

Adds MIST v2.5 (vvcrit0.4) to `isochrones`, with [α/Fe] as a fit parameter.
Grids span [Fe/H] = −4.0 … +0.5 and [α/Fe] = −0.2 … +0.6 in steps of 0.2
(five alpha nodes, natively interpolated). Products: EEP evolutionary
tracks, precomputed isochrones, and bolometric-correction (BC) tables.

**Purely additive:** four modules (`bc_v2p5.py`, `models_v2p5.py`,
`isochrone_v2p5.py`, `starmodel_v2p5.py`) are copied into the existing
`isochrones/mist/` subpackage, BESIDE the stock v1.2 modules — hence the
`_v2p5` suffixes. Nothing stock is modified.

## 2. Installing the code

```bash
python -c "import isochrones, os; print(os.path.dirname(isochrones.__file__))"
cp mist/*_v2p5.py  <that-path>/mist/
```

## 3. Downloading the MIST2 models

`download_mist_v25.py` (stdlib-only) fetches each component on request and
migrates the on-disk layout to the v1.2 decimal convention:

```bash
cd ~/.isochrones
python download_mist_v25.py --what tracks       # EEP evolutionary tracks
python download_mist_v25.py --what full_isos    # precomputed isochrones
python download_mist_v25.py --what bc --bc-systems JWST HST_ACS_WFC HST_WFC3
python download_mist_v25.py --reorganize-existing   # migrate an old layout
```

Resulting layout: `~/.isochrones/mist/MIST_v2.5_vvcrit0.4_full_isos/`,
`~/.isochrones/mist/tracks/MIST_v2.5_*`, `~/.isochrones/BC/mist/v2/`.

## 4. Verifying the download of the MIST2 models

There is no dedicated diagnostic script: the first grid build parses every
file and fails loudly on malformed input, and the widened-bounds grid
builders tolerate MIST v2.5's non-uniform per-[Fe/H] mass sampling by
design. A quick functional check:

```bash
python -c "
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_iso
ic = get_ichrone_v2p5_iso(bands=['F090W'], afe=0.0)
print(ic.interp_mag([600, 10.0, -1.0, 1e4, 0.1], ic.bands))"
```

## 5. Extinction

Native to the BC tables: MIST v2.5 BC grids tabulate magnitudes as a
function of (Teff, logg, [Fe/H], [α/Fe], A_V), so extinction enters through
the same interpolation as the photometry itself — no separate coefficient
table exists or is needed. A_V is simply the fifth axis of the BC lookup.

## 6. Using the models

Fixed [α/Fe] (drop-in for a v1.2 fit):

```python
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_iso
ic = get_ichrone_v2p5_iso(bands=["F090W", "F162M", "F200W"], afe=0.4)
# parameter vector: [eep, log10(age), feh, distance_pc, AV]
Teff, logg, feh, mags = ic.interp_mag([600, 10.0, -1.0, 776e3, 0.3], ic.bands)
```

Variable [α/Fe] (α as a sixth fit parameter):

```python
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_alpha
from isochrones.mist.starmodel_v2p5 import StarModelV2p5
ic = get_ichrone_v2p5_alpha(bands=["F090W", "F162M", "F200W"])
model = StarModelV2p5(ic, F090W=(21.3, 0.02), F162M=(19.9, 0.02))
model.fit()          # pymultinest; vector: eep, age, feh, afe, distance, AV
print(model.samples["afe"].median())
```

Factories: `get_ichrone_v2p5_iso(bands, afe)` (isochrone grid, fixed α),
`get_ichrone_v2p5(bands, afe)` (evolution-track grid, fixed α),
`get_ichrone_v2p5_alpha(bands)` (α-interpolating).

### Worked example: synthesize and fit a star

The same exercise as readme_basti.md §6, with MIST: Gaia G, HST/ACS F814W,
2MASS J/H/Ks, plus a parallax. One convention difference to mind: **MIST
band tokens follow the v2.5 BC-table column headers** of the systems you
downloaded (`--what bc --bc-systems ...`), so before running, confirm the
five tokens against your install — e.g. HST/ACS F814W is
`ACS_WFC_F814W` under the conventions used throughout this repository,
while the Gaia and 2MASS tokens live in the BC system that provides them;
list what your interpolator actually has with
`print(sorted(ic.bc_grid.bands))` and substitute below if yours differ.

```python
import numpy as np
from isochrones.mist.isochrone_v2p5 import get_ichrone_v2p5_alpha
from isochrones.mist.starmodel_v2p5 import StarModelV2p5

BANDS = ["G", "ACS_WFC_F814W", "2MASS_J", "2MASS_H", "2MASS_Ks"]  # verify!
ic = get_ichrone_v2p5_alpha(bands=BANDS)

# --- 1. truth: an upper-RGB halo giant at 8 kpc -------------------------
#     vector: [eep, log10(age/yr), [Fe/H], [a/Fe], distance_pc, AV]
truth = [550, 10.0, -1.0, 0.30, 8000.0, 0.5]      # MIST EEP 550 ~ upper RGB
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

Here α = +0.30 is a native interpolation between the +0.2 and +0.4 grids
(no regime switching — see §8 for the contrast with BaSTI).

## 7. EEP conventions

MIST EEPs are constructed (Dotter 2016): ZAMS = 202, IAMS = 353, TAMS = 454,
RGB tip = 605, ZACHeB = 631, TACHeB = 707, TP-AGB begins = 808, post-AGB =
1409; the v2.5 grid extends to EEP 1721. Typical fitter bounds for evolved
stars: `eep = (400, 1425)` — mid-MS through the end of the TP-AGB.

## 8. Differences from the BaSTI extension

| Aspect | MIST v2.5 (this readme) | BaSTI (readme_basti.md) |
|---|---|---|
| [α/Fe] nodes | five: −0.2 … +0.6 step 0.2, native interpolation | three: −0.2, 0, +0.4; quadratic/linear scheme, 0.4–0.6 extrapolated |
| [Fe/H] range | −4.0 … +0.5, uniform for all α | α-dependent ceilings: +0.44 (α=−0.2), +0.30 (α=0), **+0.09 (α=+0.4)** |
| EEP coordinate | constructed EEPs (Dotter 2016), to 1721 | file row number = Hidalgo+18 Table 4 line − 1, always 2100 rows |
| TP-AGB | included (EEP 808–1409) | **absent** — isochrones end at the early AGB |
| Extinction | native BC-table axis (A_V inside the BC grids) | precomputed R_X(band, A_V) table (fm07 default; curve swappable) |
| Data products | tracks + isochrones + per-system BC tables | isochrone files only (206,655 files, 15 systems) |
| Age coverage on disk | isochrone grid ages | 20 Myr – 29.5 Gyr; complete (age, feh) coverage below 15 Gyr |

## 9. Scientific caveats

- **TP-AGB surface abundances**: stars above the RGB tip (EEP ≳ 605) undergo
  dredge-up that alters surface C/O; molecular-band photometry there should
  not be over-interpreted.
- v2.5's non-uniform mass sampling per [Fe/H] is handled by tolerant grid
  builders, but extreme corner queries ([Fe/H] ≲ −3.5 with α = +0.6) sample
  sparse track coverage.
- The α-interpolator treats [α/Fe] as continuous between the five nodes;
  behavior beyond ±(node range) is clamped, not extrapolated.
