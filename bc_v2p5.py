"""MIST v2.5 (alpha-enhanced) bolometric-correction grid.

This module is purely additive: it subclasses the existing
``MISTBolometricCorrectionGrid`` and changes only what MIST v2.5 requires.
The original ``mist/bc.py`` is left completely untouched, and nothing here
runs unless you explicitly construct ``MISTBolometricCorrectionGridV2p5``
(or, later, request ``version="2.5"`` through the wiring layer).

What v2.5 changes, and how this handles it:
  * Data lives under  $ISOCHRONES/BC/mist/v2/  (separate from v1.2).
  * Tables have a '# lgTef ...' header (log10 Teff) plus extra Fe_H / a_Fe
    columns. We read the header, convert lgTef -> linear Teff, and rename
    Fe_H -> [Fe/H].
  * The grid carries two extra axes (a_Fe, Rv). For "Option A" we fix
    [a/Fe] to a chosen value and Rv to 3.1, collapsing the index back to the
    v1.2 layout (Teff, logg, [Fe/H], Av) so the interpolator is unchanged.
  * Some v2.5 column names differ from the v1.2 short names (e.g. JWST bands
    are prefixed 'NIRCAM_', and the file mislabels F470N/F480M as
    F470W/F480W). Those are remapped here.
"""

import os
import glob

import pandas as pd

from ..config import ISOCHRONES
from .bc import MISTBolometricCorrectionGrid


class MISTBolometricCorrectionGridV2p5(MISTBolometricCorrectionGrid):
    name = "mist"
    version = "2.5"

    # Index of the raw v2.5 tables after lgTef->Teff and Fe_H->[Fe/H].
    _V25_INDEX = ["Teff", "logg", "[Fe/H]", "a_Fe", "Av", "Rv"]

    # v1.2 internal photometric-system name -> v2.5 on-disk file extension.
    # (List only the ones that differ; everything else maps to itself.)
    _V25_SYSTEM = {
        "HST_ACSWF": "HST_ACS_WFC",
        "HST_ACSHR": "HST_ACS_HRC",
        "HST_WFC3": "HST_WFC3",
    }

    # Column-name prefix used inside the v2.5 tables, per system.
    _V25_PREFIX = {
        "JWST": "NIRCAM_",
    }

    # Per-system exact-column overrides where v2.5 column names don't follow
    # the simple prefix rule. The MIST v2.5 JWST table mislabels the F470N and
    # F480M columns as F470W/F480W; users still request the real names.
    _V25_BAND_FIX = {
        "JWST": {
            "F470N": "NIRCAM_F470W",
            "F480M": "NIRCAM_F480W",
        },
    }

    # Bare NIRCam filter names. Resolved to JWST here (rather than via the
    # parent's phot_bands) so the v2.5 layer is self-contained against any
    # isochrones version, including ones predating JWST support.
    _V25_JWST_BANDS = frozenset({
        "F070W", "F090W", "F115W", "F140M", "F150W2", "F150W", "F162M",
        "F164N", "F182M", "F187N", "F200W", "F210M", "F212N", "F250M",
        "F277W", "F300M", "F322W2", "F323N", "F335M", "F356W", "F360M",
        "F405N", "F410M", "F430M", "F444W", "F460M", "F466N", "F470N", "F480M",
    })

    def __init__(self, bands=None, afe=0.0, **kwargs):
        self.afe = afe
        super().__init__(bands)
        self.version = "2.5"

    @property
    def datadir(self):
        return os.path.join(ISOCHRONES, "BC", self.name, "v2")

    # ---- band resolution ----------------------------------------------------

    def resolve_band_v25(self, b):
        """Map a user band token -> (v2.5 file-extension system, v2.5 column)."""
        # Allow passing a fully-qualified v2.5 column directly.
        if b.startswith("NIRCAM_"):
            return "JWST", b
        if b.startswith("NIRISS_"):
            return "NIRISS", b

        # Bare NIRCam filter names -> JWST, resolved without the parent's
        # phot_bands. Applies the same column prefix / F470N-F480M fix the
        # parent path would, so downstream behavior is identical.
        if b in self._V25_JWST_BANDS:
            fix = self._V25_BAND_FIX.get("JWST", {})
            if b in fix:
                return "JWST", fix[b]
            return "JWST", self._V25_PREFIX.get("JWST", "") + b

        # Qualified HST names: resolve straight to the v2.5 system file. The
        # parent get_band can't (its regex needs a letters-only system token, so
        # the digit in 'WFC3' breaks it, and some installs don't list WFC3 in
        # phot_bands at all). v2.5 HST columns are plain (e.g. 'ACS_WFC_F475W',
        # 'WFC3_UVIS_F390W' -- no prefix), confirmed from the table headers.
        if b.startswith("WFC3_"):
            return "HST_WFC3", b
        if b.startswith("ACS_WFC_"):
            return "HST_ACS_WFC", b
        if b.startswith("ACS_HRC_"):
            return "HST_ACS_HRC", b

        # Reuse the parent's (v1.2) resolver, then translate the names.
        phot, band = self.get_band(b)  # parent classmethod; no version kwarg
        system = self._V25_SYSTEM.get(phot, phot)

        fix = self._V25_BAND_FIX.get(phot, {})
        if b in fix:
            return system, fix[b]
        if band in fix:
            return system, fix[band]

        column = self._V25_PREFIX.get(phot, "") + band
        return system, column

    def _make_band_map(self):
        phot_systems = set()
        band_map = {}
        for b in self.bands:
            phot, column = self.resolve_band_v25(b)
            phot_systems.add(phot)
            band_map[b] = column
        self._band_map = band_map
        self._phot_systems = phot_systems

    # ---- table parsing ------------------------------------------------------

    def parse_table(self, filename):
        names = None
        with open(filename) as fin:
            for line in fin:
                s = line.lstrip("#").strip()
                if s.startswith("lgTef"):
                    names = s.split()
                    break
        if names is None:
            raise ValueError("No '# lgTef ...' header found in {}".format(filename))

        df = pd.read_csv(filename, names=names, comment="#", sep=r"\s+")
        df["Teff"] = 10 ** df["lgTef"]
        df = df.rename(columns={"Fe_H": "[Fe/H]"})
        df = df.drop(columns=["lgTef"]).set_index(self._V25_INDEX)
        return df

    # ---- assembling the grid ------------------------------------------------

    def get_hdf_filename(self, phot):
        sgn = "p" if self.afe >= 0 else "m"
        return os.path.join(self.datadir, "{}_afe{}{:.1f}.h5".format(phot, sgn, abs(self.afe)))

    def get_df(self, *args, **kwargs):
        df_all = pd.DataFrame()
        for phot in self.phot_systems:
            hdf_filename = self.get_hdf_filename(phot=phot)
            if not os.path.exists(hdf_filename):
                filenames = glob.glob(os.path.join(self.datadir, "*.{}".format(phot)))
                if not filenames:
                    raise FileNotFoundError(
                        "No MIST v2.5 BC files for system '{0}' in {1}.\n"
                        "Download them with:\n"
                        "    python download_mist_v25.py --what bc --bc-systems {0}".format(
                            phot, self.datadir
                        )
                    )
                df = pd.concat([self.parse_table(f) for f in filenames]).sort_index()
                # Collapse the two extra v2.5 axes to recover the v1.2 layout:
                # fix [a/Fe] to the requested value, and Rv to 3.1.
                df = df.xs(round(self.afe, 3), level="a_Fe").xs(3.1, level="Rv")
                df.to_hdf(hdf_filename, key="df")
            df = pd.read_hdf(hdf_filename)
            df_all = pd.concat([df_all, df], axis=1)

        df_all = df_all.rename(columns={v: k for k, v in self.band_map.items()})
        for col in list(df_all.columns):
            if col not in self.bands:
                del df_all[col]

        return df_all
