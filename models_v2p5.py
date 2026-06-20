"""MIST v2.5 (alpha-enhanced) evolution-track grid -- Option A.

Purely additive: subclasses ``MISTEvolutionTrackGrid`` and changes only what
MIST v2.5 requires. The original ``mist/models.py`` is left untouched, and
nothing here runs unless ``MISTEvolutionTrackGridV2p5`` is constructed.

Option A treats [a/Fe] as a fixed discrete choice (default 0.0), so the grid
stays 3-D ``(initial_feh, initial_mass, EEP)`` exactly like v1.2. Because the
data was laid out with v1.2-style decimal directory names
(``MIST_v2.5_feh_<f>_afe_<a>_vvcrit<v>_EEPS``), the inherited
``get_file_basename`` / ``get_directory_path`` resolve correctly with no change.

What v2.5 changes, and how this handles it:
  * default_kwargs -> version "2.5" (so paths/tarball names are v2.5).
  * 17-point [Fe/H] grid (adds -2.75 and -2.25).
  * GS98 (Z/X)_sun for the *derived surface* [Fe/H] column (v1.2 used
    Asplund+2009). Interpolation uses initial_feh, so this affects only an
    output column.
  * A tolerant tail-filling step so v2.5's slightly different track coverage
    cannot crash the build (short tracks that can't be bracketed are left
    ragged; with is_full=False the interpolator pads holes with NaN).

The raw ``.track.eep`` files parse with the inherited ``to_df`` unchanged
(the '# EEPs:' and '# star_age' header lines match v1.2).
"""

import os
import re
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..models import StellarModelGrid
from ..logger import getLogger
from .models import MISTEvolutionTrackGrid, MISTIsochroneGrid
from .eep import default_max_eep


# MIST v2 adopts the Grevesse & Sauval (1998) solar abundance scale; v1.2 used
# Asplund+2009 with (Z/X)_sun = 0.0181. This normalization is used ONLY for the
# derived surface-[Fe/H] output column; the interpolation axis is initial_feh
# (set from the grid), so CMD generation is unaffected by this value.
# >>> VERIFY against the MIST v2 paper/docs and adjust if needed. <<<
SOLAR_ZX_V25 = 0.0231  # GS98 photospheric (Z/X)_sun


# Full MIST v2.5 [Fe/H] grid (17 points; adds -2.75, -2.25 vs v1.2's 15).
_FEHS_V25 = np.array(
    (
        -4.00, -3.50, -3.00, -2.75, -2.50, -2.25, -2.00, -1.75, -1.50,
        -1.25, -1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50,
    )
)


def max_eep_v2p5(mass, feh):
    """Target maximum EEP per (mass, feh) for v2.5 tail-filling.

    Starts from the v1.2 mass-based default. Because the filling step below is
    tolerant, an over- or under-estimate here cannot crash the build; it only
    changes how aggressively short tracks are extended. Refine with measured
    v2.5 coverage later if the ragged-track warnings flag anything important.
    """
    return default_max_eep(mass)


class MISTEvolutionTrackGridV2p5(MISTEvolutionTrackGrid):
    default_kwargs = {"version": "2.5", "vvcrit": 0.4, "afe": 0.0}

    fehs = _FEHS_V25
    n_fehs = len(_FEHS_V25)

    # v2.5 tracks extend a few EEPs past v1.2's final EEP of 1710. n_eep sizes
    # the age/dt_deep arrays in get_array_grids, so it MUST cover the longest
    # track or the build raises (and drops into a pdb in the parent).
    n_eep = 1721
    bounds = (("age", (5, 10.13)), ("feh", (-4, 0.5)), ("eep", (0, 1721)), ("mass", (0.1, 300)))

    def max_eep(self, mass, feh):
        return max_eep_v2p5(mass, feh)

    def compute_additional_columns(self, df):
        """As the v1.2 track grid, but with the GS98 (Z/X)_sun constant.

        Calls the StellarModelGrid base directly (skipping MISTModelGrid's
        Asplund-normalized feh line) so the original models.py is untouched.
        Note v2.5 renamed the surface-metallicity column log_surf_z ->
        log_surf_cell_z; we accept either.
        """
        df = StellarModelGrid.compute_additional_columns(self, df)
        zcol = "log_surf_cell_z" if "log_surf_cell_z" in df.columns else "log_surf_z"
        with np.errstate(divide="ignore", invalid="ignore"):
            df["feh"] = df[zcol] - np.log10(df["surface_h1"]) - np.log10(SOLAR_ZX_V25)
            df["age"] = np.log10(df["star_age"])
        return df

    def get_dt_deep(self, compute=False):
        """Correct, vectorized d(log10 age)/d(EEP) per track.

        The parent's compute path uses ``df.loc[f, m]`` (read as row,column on a
        3-level MultiIndex by modern pandas) and writes to a copy -- it never runs
        for v1.2 because the dt_deep HDF ships prebuilt, but v2.5 must compute it.
        This replacement groups by (initial_feh, initial_mass) and differentiates
        within each track. dt_deep is not used in magnitude interpolation, so the
        infinities at age=0 PMS points are mapped to NaN.
        """
        filename = os.path.join(self.datadir, "dt_deep{}.h5".format(self.kwarg_tag))
        if os.path.exists(filename):
            try:
                return pd.read_hdf(filename, "dt_deep")
            except Exception:
                pass

        df = self.get_df()

        def _track_deriv(g):
            with np.errstate(divide="ignore", invalid="ignore"):
                log_age = np.log10(g["star_age"].to_numpy())
            return pd.Series(np.gradient(log_age, g["eep"].to_numpy()), index=g.index)

        dt_deep = (
            df.groupby(level=[0, 1], group_keys=False)
            .apply(_track_deriv)
            .reindex(df.index)
            .replace([np.inf, -np.inf], np.nan)
        )
        dt_deep.name = "dt_deep"
        dt_deep.to_hdf(filename, key="dt_deep")
        return dt_deep

    def get_array_grids(self, recalc=False):
        """Build the (feh x mass) age / dt_deep arrays used by the fast numba
        age->EEP lookup (get_eep / interp_eep).

        Two v2.5 fixes vs the parent's StellarModelGrid.get_array_grids:
          * arrays are sized to self.n_eep (1721); v2.5 tracks run past 1710,
            which would otherwise overflow the array and drop into a pdb;
          * (feh, mass) pairs absent from the grid are tolerated as length-0
            rows instead of raising KeyError -- v2.5 has non-uniform mass
            sampling across [Fe/H] (e.g. fewer masses at very low metallicity).
        """
        if not (recalc or not os.path.exists(self.array_grid_filename)):
            d = np.load(self.array_grid_filename)
            return d["age"], d["dt_deep"], d["lengths"]

        fehs = self.fehs
        masses = self.masses
        n = len(fehs) * len(masses)
        age_arrays = np.full((n, self.n_eep), np.nan)
        dt_deep_arrays = np.full((n, self.n_eep), np.nan)
        lengths = np.zeros(n)
        n_missing = 0
        for i, (f, m) in tqdm(
            enumerate(itertools.product(fehs, masses)),
            total=n,
            desc="building irregular age grid (v2.5)",
        ):
            try:
                subdf = self.df.xs((f, m), level=(0, 1))
            except KeyError:
                lengths[i] = 0
                n_missing += 1
                continue
            xs = subdf[self.eep_replaces].values
            L = min(len(xs), self.n_eep)
            lengths[i] = L
            age_arrays[i, :L] = xs[:L]
            dt_deep_arrays[i, :L] = subdf.dt_deep.values[:L]

        if n_missing:
            getLogger().info(
                "v2.5 age grid: {}/{} (feh, mass) pairs absent (non-uniform "
                "mass sampling); marked length 0.".format(n_missing, n)
            )
        np.savez(
            self.array_grid_filename,
            age=age_arrays,
            dt_deep=dt_deep_arrays,
            lengths=lengths.astype(int),
        )
        d = np.load(self.array_grid_filename)
        return d["age"], d["dt_deep"], d["lengths"]

    def df_all_feh_interpolated(self, feh):
        """Tolerant version of the parent's missing-tail interpolation.

        Same algorithm, but a short track whose tail cannot be bracketed by
        complete-enough neighbors is skipped (logged) instead of raising. Index
        level names are preserved through the concat.
        """
        hdf_filename = self.get_feh_interpolated_hdf_filename(feh)
        if os.path.exists(hdf_filename):
            return pd.read_hdf(hdf_filename, "df")

        getLogger().info("Interpolating incomplete tracks for feh = {}".format(feh))
        df = self.df_all_feh(feh)
        df_interp = df.copy()
        df_interp["interpolated"] = False
        masses = df.index.levels[1]
        names = df.index.names

        for i, m in tqdm(
            enumerate(masses),
            total=len(masses),
            desc="interpolating tracks (feh={})".format(feh),
        ):
            n_eep = len(df.xs(m, level="initial_mass"))
            eep_max = self.max_eep(m, feh)
            if not eep_max or n_eep >= eep_max:
                continue

            # Find complete-enough bracketing masses; skip this one if we can't.
            try:
                ilo = i
                while True:
                    ilo -= 1
                    if ilo < 0:
                        raise LookupError
                    mlo = masses[ilo]
                    if len(df.xs(mlo, level="initial_mass")) >= eep_max:
                        break
                ihi = i
                while True:
                    ihi += 1
                    if ihi >= len(masses):
                        raise LookupError
                    mhi = masses[ihi]
                    if len(df.xs(mhi, level="initial_mass")) >= eep_max:
                        break
            except LookupError:
                getLogger().info(
                    "feh={}: cannot bracket mass={} up to EEP {}; leaving ragged.".format(
                        feh, m, eep_max
                    )
                )
                continue

            new_eeps = np.arange(n_eep + 1, eep_max + 1)
            new_index = pd.MultiIndex.from_product([[feh], [m], new_eeps], names=names)
            norm_distance = (m - mlo) / (mhi - mlo)
            lo_index = pd.MultiIndex.from_product([[feh], [mlo], new_eeps])
            hi_index = pd.MultiIndex.from_product([[feh], [mhi], new_eeps])

            new_data = pd.DataFrame(
                df.loc[lo_index, :].values * (1 - norm_distance)
                + df.loc[hi_index, :].values * norm_distance,
                index=new_index,
                columns=df.columns,
            )
            new_data["interpolated"] = True
            df_interp = pd.concat([df_interp, new_data])

        df_interp.sort_index(inplace=True)
        df_interp.to_hdf(hdf_filename, key="df")
        return pd.read_hdf(hdf_filename, "df")


class MISTIsochroneGridV2p5(MISTIsochroneGrid):
    """MIST v2.5 theoretical-isochrone grid (full_isos), fixed [a/Fe].

    The isochrone counterpart to MISTEvolutionTrackGridV2p5: this is the grid
    behind the *isochrone* interpolator (eep_replaces='mass'), which is what
    StarModel.fit() samples. v2.5 splits full_isos by BOTH [Fe/H] and [a/Fe],
    one file per pair (feh_p000_afe_p0_vvcrit0.4_full.iso, etc.), so this grid
    selects a single [a/Fe] and interpolates across the 17 [Fe/H].

    v2.5-specific handling vs the v1.2 parent, all additive:
      * filenames encode [Fe/H] as a signed 3-digit integer (m025 = -0.25),
        not the v1.2 decimal form -> get_feh override;
      * only the files matching the requested [a/Fe] are read -> get_filenames;
      * caches (HDF, dm_deep) are keyed by [a/Fe] so different alpha don't
        collide, while the shared data directory name omits alpha;
      * the surface-Z column renamed log_surf_z -> log_surf_cell_z, and the GS98
        (Z/X)_sun is used -> compute_additional_columns;
      * get_dm_deep is reimplemented (the parent's df.loc[f,a] reads a column on
        a 3-level index and writes to a copy; it never runs on v1.2 because the
        dm_deep HDF ships prebuilt, but v2.5 must compute it).
    """

    default_kwargs = {"version": "2.5", "vvcrit": 0.4, "kind": "full_isos", "afe": 0.0}

    fehs = _FEHS_V25
    n_fehs = len(_FEHS_V25)
    n_eep = 1721
    bounds = (("age", (5, 10.13)), ("feh", (-4, 0.5)), ("eep", (0, 1721)), ("mass", (0.1, 300)))

    @staticmethod
    def _afe_str(afe):
        """'p0' for 0.0, 'm2' for -0.2, 'p4' for +0.4 -- matches v2.5 filenames."""
        return "{}{:.0f}".format("p" if afe >= 0 else "m", abs(afe) * 10)

    @property
    def kwarg_tag(self):
        # afe-specific cache tag so HDF/dm_deep for different [a/Fe] don't collide
        return "{}_afe{}".format(super().kwarg_tag, self._afe_str(self.kwargs.get("afe", 0.0)))

    def get_directory_path(self, **kwargs):
        # Data directory is shared across [a/Fe]; its name omits alpha (unlike
        # kwarg_tag), so override rather than inherit the kwarg_tag-based path.
        return os.path.join(
            self.datadir,
            "MIST_v{version}_vvcrit{vvcrit}_{kind}".format(**self.kwargs),
        )

    def get_tarball_url(self, **kwargs):
        return (
            "https://mist.science/data/tarballs_v{version}/isos/"
            "MIST_v{version}_vvcrit{vvcrit}_{kind}.txz".format(**self.kwargs)
        )

    def get_filenames(self, **kwargs):
        files = self.get_existing_filenames(**kwargs)
        token = "_afe_{}_".format(self._afe_str(self.kwargs.get("afe", 0.0)))
        sel = [f for f in files if token in os.path.basename(f)]
        if not sel:
            raise ValueError(
                "No v2.5 isochrone files for afe={} in {} (looked for '{}').".format(
                    self.kwargs.get("afe", 0.0), self.get_directory_path(), token
                )
            )
        return sel

    @classmethod
    def get_feh(cls, filename):
        # v2.5 encodes [Fe/H] as a signed 3-digit integer: m025 = -0.25, p050 = +0.50
        m = re.search(r"feh_([mp])(\d{3})_afe", os.path.basename(filename))
        if not m:
            raise ValueError("Cannot parse [Fe/H] from {}".format(filename))
        sign = 1 if m.group(1) == "p" else -1
        return sign * int(m.group(2)) / 100.0

    def compute_additional_columns(self, df):
        df = StellarModelGrid.compute_additional_columns(self, df)
        with np.errstate(divide="ignore", invalid="ignore"):
            zcol = "log_surf_cell_z" if "log_surf_cell_z" in df.columns else "log_surf_z"
            df["feh"] = df[zcol] - np.log10(df["surface_h1"]) - np.log10(SOLAR_ZX_V25)
        return df

    def get_dm_deep(self, compute=False):
        """Correct, vectorized d(initial_mass)/d(EEP) per (age, feh) isochrone."""
        filename = os.path.join(self.datadir, "dm_deep{}.h5".format(self.kwarg_tag))
        if os.path.exists(filename):
            try:
                return pd.read_hdf(filename, "dm_deep")
            except Exception:
                pass

        df = self.get_df()

        def _iso_deriv(g):
            if len(g) < 2:
                return pd.Series(np.nan, index=g.index)
            return pd.Series(
                np.gradient(g["initial_mass"].to_numpy(), g["eep"].to_numpy()),
                index=g.index,
            )

        dm_deep = (
            df.groupby(level=[0, 1], group_keys=False)
            .apply(_iso_deriv)
            .reindex(df.index)
            .replace([np.inf, -np.inf], np.nan)
        )
        dm_deep.name = "dm_deep"
        dm_deep.to_hdf(filename, key="dm_deep")
        return dm_deep
