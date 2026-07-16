# isochrones-v2p5

Additive extensions to Tim Morton's
[`isochrones`](https://isochrones.readthedocs.io/) package, adding two modern
stellar-model libraries with [α/Fe] as a fit parameter:

| Library | Models | Docs |
|---|---|---|
| **MIST v2.5** | MIST v2.5 vvcrit0.4 (tracks, isochrones, bolometric corrections), [α/Fe] = −0.2 … +0.6 | [readme_mist_v25.md](readme_mist_v25.md) |
| **BaSTI-IAC** | BaSTI O1D1E1 isochrones (Hidalgo+2018; Pietrinferni+2021), [α/Fe] = −0.2, 0.0, +0.4 | [readme_basti.md](readme_basti.md) |

Both extensions are **purely additive**: no file belonging to the original
`isochrones` package is modified, created, or shadowed. Installation is
copying files into the installed package; removal is deleting them.

## Repository layout

```
mist/       four modules copied into  <site-packages>/isochrones/mist/
            (suffix _v2p5: they live BESIDE the stock v1.2 modules)
basti/      one subpackage copied to  <site-packages>/isochrones/basti/
            (no suffixes: the directory itself is the namespace)
*.py        infrastructure scripts at root: download_mist_v25.py,
            download_basti.py, diagnose_basti_grid.py,
            build_extinction_table.py, build_basti_grids.py
analysis/   fitting-campaign and model-comparison scripts
```

The naming rule: **a file carries a version suffix if and only if it shares a
directory with stock files it must not overwrite.** The MIST modules do; the
BaSTI subpackage does not.

## Shared environment

Python 3.10, `numpy < 2`, `pandas < 2`, the stock `isochrones` package, and
(for fitting) `pymultinest`. The BaSTI extinction-table builder additionally
uses the [`extinction`](https://extinction.readthedocs.io/) package. Data
downloads to `$ISOCHRONES` (default `~/.isochrones`):

```
~/.isochrones/
├── mist/    + BC/mist/v2/        MIST v2.5 products   (readme_mist_v25.md §3)
├── basti/                        BaSTI products        (readme_basti.md §3)
└── filters/svo/                  cached SVO filter transmission curves
```

## Quickstart pointers

Each library readme follows the same section plan — 1 Overview, 2 Installing
the code, 3 Downloading the data, 4 Verifying the data, 5 Extinction,
6 Using the models, 7 EEP conventions, 8 Differences from the other library,
9 Scientific caveats — so the two can be read side by side. Section 6 of
each contains the same worked example (synthesize Gaia G + HST F814W +
2MASS JHKs photometry and a parallax for a halo giant, then recover its
parameters with a fit), written in each library's conventions.

## Attribution

Built on `isochrones` (Morton 2015). Model data: MIST v2.5
(Dotter/Choi et al.) and BaSTI-IAC (Hidalgo et al. 2018; Pietrinferni et al.
2021). Please cite the model papers and `isochrones` when using either
extension.
