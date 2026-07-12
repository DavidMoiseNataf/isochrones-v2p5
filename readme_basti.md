# BaSTI-IAC support for `isochrones` (additive extension)

This extension adds the [BaSTI-IAC](http://basti-iac.oa-abruzzo.inaf.it/)
stellar models (Hidalgo et al. 2018; Pietrinferni et al. 2021) to Tim
Morton's [`isochrones`](https://isochrones.readthedocs.io/) package, as an
alternative to the MIST v2.5 extension in this repository.

**Physics case:** overshooting + atomic diffusion + mass loss η = 0.3,
Y_BBN = 0.247 (BaSTI tag "O1D1E1"), at [α/Fe] = −0.2, 0.0, +0.4.

**Purely additive:** installation adds a single new subpackage directory,
`<site-packages>/isochrones/basti/`. No file in `isochrones/` or
`isochrones/mist/` belonging to the original package is modified, created,
or shadowed. Uninstalling means deleting that one directory.

---

## 1. Installing the code

Copy the `basti/` directory of this repository into your installed
`isochrones` package:

```bash
# find the package location
python -c "import isochrones, os; print(os.path.dirname(isochrones.__file__))"

# copy the subpackage (adjust the destination to the path printed above)
cp -r basti <site-packages>/isochrones/
```

The two scripts (`download_basti.py`, `diagnose_basti_grid.py`) are
standalone (Python 3 standard library only for the downloader) and can be
run from anywhere; a convenient place is `$ISOCHRONES`
(default `~/.isochrones`), alongside `download_mist_v25.py`.

Requirements: the stock `isochrones` package. The MIST v2.5 extension in
this repository is required only for two optional features: deriving
extinction coefficients from the MIST v2.5 bolometric-correction tables, and
reusing `StarModelV2p5` for free-[α/Fe] fits.

## 2. Downloading the BaSTI data

One command fetches everything (all 3 alphas × 15 photometric systems,
~885 tarballs → **206,655 isochrone files**; record the disk footprint with
`du -sh ~/.isochrones/basti` after the run):

```bash
cd ~/.isochrones
python download_basti.py --what isos --scrape
```

Notes:

* `--scrape` attempts to read the live server listing; the BaSTI server
  403-blocks directory listings, so the script falls back to its built-in
  seed manifest — which has been probe-verified complete (236/236 core
  URLs). The fallback message is expected and harmless.
* Downloads are **resumable**: re-running skips everything recorded in
  `~/.isochrones/manifest_basti_O1D1E1.json` (reported as `[have]`).
* Subsets: `--afe 0.0 0.4`, `--iso-systems JWST WFC3 ACS GAIA 2MASS DECAM
  EUCLID GALEX HAWKI TESS VISTA WISE JC PANSTARRS SKYMAPPER`.
* `--probe` HEAD-checks every URL without downloading; `--dry-run` previews;
  `--guess-ext` probes candidate extension names (used when BaSTI's tarball
  naming is inconsistent).
* **Roman is deliberately excluded**: the server carries three Roman product
  generations with inconsistent tarball coverage per alpha directory (see
  the comment in `download_basti.py`). Revisit if BaSTI posts uniform
  `roman_vega` tarballs.

Resulting layout, parallel to the MIST v2.5 products:

```
~/.isochrones/
├── mist/MIST_v2.5_vvcrit0.4_full_isos/       (MIST v2.5 extension)
├── basti/BaSTI_O1D1E1_isos/                  (this extension; flat, all
│                                              alphas/compositions/systems)
├── manifest_mist_v2.5_vvcrit0.4.json
└── manifest_basti_O1D1E1.json
```

Each fixed-alpha grid selects its files by the alpha tag embedded in the
filenames (`...p00o1d1e1.isc_...`) — the analogue of the MIST v2.5 grid's
`_afe_p4_` filename filter. Grid caches (HDF/NPZ/dm_deep, written under
`~/.isochrones/basti/` on first use) carry a code-version tag and are
regenerated automatically when needed; they are never downloaded.

## 3. Verifying the data

```bash
cd ~/.isochrones
python diagnose_basti_grid.py --systems jwst-nircam_zp_vega-sirius acs
```

This parses every selected file and hard-fails on: point count ≠ 2100,
non-monotonic initial mass, TRGB not at row 1289 for ages ≥ 2 Gyr, or
theoretical columns differing between photometric systems at the same
(age, composition). Expected result on a complete download:
`0 hard-check failure(s)` and `missing (age, composition) nodes below
15 Gyr: 0` for every alpha (all coverage raggedness is super-Hubble).
Two or more `--systems` are needed to exercise the cross-system check;
runtime scales with the number of systems selected.

### Why row number is the EEP

Every BaSTI O1D1E1 isochrone has exactly 2100 points whose row indices map
one-to-one onto the normalized-track line numbers of Hidalgo et al. (2018),
Table 4 (0-based row = 1-based line − 1): ZAMS at row 99, track turn-off at
359, RGB base at 489, RGB bump at 859/889, **TRGB at row 1289**, quiescent
core-He burning from row 1299, end of the early AGB at row 2099. These
anchors are importable as `isochrones.basti.models.BASTI_EEP_ANCHORS` and
have been validated against the full downloaded grid (~55,000 files,
20 Myr – 29.5 Gyr, all three alphas). Row number therefore serves directly
as the pseudo-EEP coordinate of the interpolation grid — no EEP construction
step exists or is needed.

## 4. Using the models

*(To be appended: fixed-alpha and free-alpha interpolators, StarModel
fitting, `age_range` grid restriction, extinction-coefficient strategy,
EEP-bound selection from `BASTI_EEP_ANCHORS`.)*
