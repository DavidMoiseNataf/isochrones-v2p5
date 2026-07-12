"""BaSTI-IAC isochrone grid for Tim Morton's ``isochrones`` package.

Purely additive, following the pattern of the MIST v2.5 extension
(``mist/models_v2p5.py``): this subclasses ``StellarModelGrid`` and nothing in
the stock package is touched. Nothing here runs unless ``BastiIsochroneGrid``
is constructed.

Physics case (fixed for now, carried in cache tags for future-proofing):
    overshooting + atomic diffusion + mass loss eta = 0.3, Y_BBN = 0.247
    -> BaSTI file tag "O1D1E1" (case-insensitive on disk).

===========================================================================
THE BaSTI PSEUDO-EEP  (verified empirically on O1D1E1 sample files)
===========================================================================
Every BaSTI-IAC .isc isochrone in this case has EXACTLY Np = 2100 points,
with a fixed two-segment structure:

    rows    0 - 1289 : 0.1 Msun lower MS -> MS -> SGB -> RGB -> **TRGB at
                       row 1289 exactly** (verified across ages 1-12 Gyr,
                       [a/Fe] = -0.2, 0.0, +0.4, and [M/H] from -0.6 to +0.06)
    rows 1290 - 2099 : start of quiescent He burning (ZAHB / clump) ->
                       core He exhaustion -> early AGB

Because the point count per segment is fixed and the segment boundary is
pinned, the RAW ROW NUMBER is a valid pseudo-EEP: interpolating between
adjacent grid nodes at fixed row number never mixes pre- and post-He-flash
phases. This is what makes the whole DFInterpolator machinery work unchanged.
Secondary anchors (e.g. the MSTO) drift by ~5-20 rows between adjacent [Fe/H]
nodes, which mildly smears interpolation near the turnoff; acceptable for
RGB-focused work and re-anchorable later if needed.

The parser VALIDATES Np == 2100 on every block it reads and refuses to build
a grid from a nonconforming file (a changed BaSTI release would silently
corrupt the EEP mapping otherwise).

===========================================================================
CACHE INVALIDATION -- same three layers as MIST v2.5
===========================================================================
Values computed here are baked into three on-disk caches, each loaded in
preference to recomputing. Editing this module has NO EFFECT until ALL are
deleted:

  1. grid dataframe HDF    ~/.isochrones/basti/basti_<tag>.h5
  2. DFInterpolator NPZ    ~/.isochrones/basti/full_grid_<tag>.npz
  3. dm_deep HDF           ~/.isochrones/basti/dm_deep_<tag>.h5

where <tag> encodes case, [a/Fe], and BASTI_GRID_VERSION below. Bump
BASTI_GRID_VERSION whenever the parser or column definitions change --
that renames all three caches at once and makes stale-cache collisions
impossible.
===========================================================================

On-disk layout expected (produced by ``download_basti.py``), parallel to
MIST's mist/MIST_v2.5_vvcrit0.4_full_isos/ single flat product directory:

    ~/.isochrones/basti/BaSTI_O1D1E1_isos/
        1000z0172100y269P00O1D1E1.isc_jwst-nircam_zp_vega-sirius
        12000z0130900y264M02O1D1E1.isc_wfc3
        ...

i.e. ONE directory for all ages, [Fe/H], [a/Fe], and photometric systems --
BaSTI filenames encode the alpha tag, so this grid selects its [a/Fe] by
filename filtering, exactly as MISTIsochroneGridV2p5.get_filenames selects
its '_afe_p4_' files. Metadata ([M/H], Z, Y, age) is parsed from the FILE
HEADER, not the filename (filenames truncate Y).

[Fe/H] convention: the grid's interpolation axis is IRON abundance [Fe/H],
converted from the header's TOTAL metallicity [M/H] via the Salaris-like
relation with BaSTI's own alpha offsets (see MH_TO_FEH_OFFSET). The header
[M/H] is preserved in the 'mh' column.
"""

import os
import re
import glob
import itertools

import numpy as np
import pandas as pd

from ..config import ISOCHRONES
from ..models import StellarModelGrid
from ..logger import getLogger

# ---------------------------------------------------------------------------
# Configuration block (per package convention: explicit and adjustable)
# ---------------------------------------------------------------------------

# Bump this whenever the parser, band maps, or computed columns change.
# It is part of every cache filename (HDF, NPZ, dm_deep), so old caches can
# never be silently reused after a code change.
BASTI_GRID_VERSION = "1.0"

# Physics case: overshooting, diffusion, eta = 0.3 (BaSTI tag), Y_BBN=0.247.
DEFAULT_CASE = "O1D1E1"

# Expected points per isochrone and segment boundary (validated per file).
BASTI_NP = 2100
BASTI_TRGB_ROW = 1289   # = Hidalgo+18 Table 4 line 1290, 0-based

# Official BaSTI key points (Hidalgo et al. 2018, Table 4), converted from
# their 1-based normalized-track line numbers to 0-based row indices. These
# are CONSTRUCTION anchors -- row k of every isochrone corresponds to
# normalized line k+1 -- so they hold by definition, independent of whether
# a morphological feature (e.g. a luminosity maximum) exists at that age.
# NB: anchors are defined on the TRACKS; along an ISOCHRONE the morphological
# feature can sit a few rows later (measured isochrone MSTO: rows 359-403
# across the full grid, with the track anchor 359 as the lower envelope).
BASTI_EEP_ANCHORS = {
    "phase_start": 0,        # line 1:    age = 1000 yr
    "zams": 99,              # line 100:  zero-age main sequence
    "msto": 359,             # line 360:  max Teff along the MS (turn-off)
    "rgb_base": 489,         # line 490:  base of the RGB (low-mass)
    "rgb_bump_max": 859,     # line 860:  max L of the RGB bump
    "rgb_bump_min": 889,     # line 890:  min L of the RGB bump
    "trgb": 1289,            # line 1290: tip of the red giant branch
    "zaheb": 1299,           # line 1300: start of quiescent core He burning
    "core_he_exhausted": 1949,  # line 1950: central Y = 0.00
    "eagb_end": 2099,        # line 2100: CNO energy > He-burning energy
}

# [M/H] -> [Fe/H]: [Fe/H] = [M/H] - offset([a/Fe]).
# For the BaSTI heavy-element mixture, [M/H] ~= [Fe/H] + log10(0.694*10^[a/Fe] + 0.306)
# (Salaris, Chieffi & Straniero 1993 form). Values below evaluate that formula;
# they reproduce the BaSTI-IAC website's quoted [Fe/H]/[M/H] pairs to ~0.01 dex.
# >>> VERIFY against Pietrinferni et al. (2021) Table conventions if used for
#     anything more precise than grid labeling. <<<
MH_TO_FEH_OFFSET = {
    -0.2: -0.1080,   # log10(0.694*10^-0.2 + 0.306)
    0.0: 0.0,
    0.4: 0.2717,     # log10(0.694*10^+0.4 + 0.306)
}

# BaSTI alpha tag <-> numeric [a/Fe]
AFE_TAGS = {-0.2: "M02", 0.0: "P00", 0.4: "P04"}
AFE_FROM_TAG = {v: k for k, v in AFE_TAGS.items()}

# ---------------------------------------------------------------------------
# Photometric-system configuration.
# Keys are the .isc_<system> filename extensions as served by BaSTI-IAC.
# 'rename' maps the raw BaSTI column header token -> the canonical band token
# users request (matching the isochrones-v2p5 conventions: bare NIRCam names;
# system-qualified HST names; Gaia short names). None => keep BaSTI's name.
# WFC3 needs an explicit map because its single file mixes UVIS and IR
# channels and its F410M would otherwise collide with NIRCam's F410M.
# ---------------------------------------------------------------------------

_WFC3_UVIS = (
    "F218W F225W F275W F336W F390W F438W F475W F555W F606W F625W F775W F814W "
    "F200LP F300X F350LP F475X F600LP F850LP F390M F410M F467M F547M F621M "
    "F689M F763M F845M F395N"
).split()
_WFC3_IR = "F105W F110W F125W F140W F160W F098M F127M F139M F153M".split()

PHOT_SYSTEMS = {
    "jwst-nircam_zp_vega-sirius": {
        # bare NIRCam names are already the user-facing tokens
        "rename": None,
    },
    "wfc3": {
        "rename": {b: "WFC3_UVIS_" + b for b in _WFC3_UVIS}
        | {b: "WFC3_IR_" + b for b in _WFC3_IR},
    },
    "acs": {
        # BaSTI 'acs' is ACS/WFC; prefix every magnitude column
        "rename": "ACS_WFC_",   # string => prefix rule applied to all mag cols
    },
    "gaia-dr3": {
        "rename": {"G": "G", "G_BP": "BP", "G_RP": "RP", "G_RVS": "G_RVS"},
    },
    # extracted members of the gaia-dr3 tarballs carry this extension
    "gaia-dr3-new": {
        "rename": {"G": "G", "G_BP": "BP", "G_RP": "RP", "G_RVS": "G_RVS"},
    },
    # ---- column maps read from real BaSTI files (verified 2026-07-11) ----
    "john": {   # Johnson-Cousins; JC_Lprime avoids a quote char in band tokens
        "rename": {"U": "JC_U", "BX": "JC_BX", "B": "JC_B", "V": "JC_V",
                   "R": "JC_R", "I": "JC_I", "J": "JC_J", "H": "JC_H",
                   "K": "JC_K", "L'": "JC_Lprime", "L": "JC_L", "M": "JC_M"},
    },
    "panstrss1": {   # BaSTI's own spelling of PanSTARRS1
        "rename": {"g_p1": "PS1_g", "r_p1": "PS1_r", "i_p1": "PS1_i",
                   "z_p1": "PS1_z", "y_p1": "PS1_y", "w_p1": "PS1_w"},
    },
    "skym": {
        "rename": {"Mu": "SkyMapper_u", "Mv": "SkyMapper_v",
                   "Mg": "SkyMapper_g", "Mr": "SkyMapper_r",
                   "Mi": "SkyMapper_i", "Mz": "SkyMapper_z",
                   "Mu_leak": "SkyMapper_u_leak"},
    },
    # Roman intentionally unsupported: inconsistent tarball coverage across
    # alpha directories on the BaSTI server (see download_basti.py).
    # ---- column maps verified against real file headers (2026-07-11) ----
    # Prefixes prevent cross-system collisions (Y/J/H/Ks in HAWK-I vs VISTA;
    # ugrizY in DECam vs PS1 vs SkyMapper).
    "2mass": {
        "rename": {"Mj": "2MASS_J", "Mh": "2MASS_H", "Mk": "2MASS_Ks"},
    },
    "decam": {"rename": "DECam_"},    # u g r i z Y -> DECam_u ... DECam_Y
    "euclid": {
        "rename": {"VIS": "Euclid_VIS", "NISP_Y": "Euclid_Y",
                   "NISP_J": "Euclid_J", "NISP_H": "Euclid_H"},
    },
    "galex": {"rename": "GALEX_"},    # FUV NUV
    "hawki": {"rename": "HAWKI_"},    # Y J H Ks CH4
    "tess": {"rename": {"Tess": "TESS_T"}},
    "vista": {"rename": "VISTA_"},    # Z Y J H Ks
    "wise": {"rename": "WISE_"},      # W1 W2 only (no W3/W4 in BaSTI)
}

# tarball-name -> extracted-member-extension aliases (requesting the left
# name transparently reads files with the right extension)
SYSTEM_ALIASES = {
    "gaia-dr3": "gaia-dr3-new",
}

# Default systems merged into the grid if the user doesn't specify.
DEFAULT_SYSTEMS = ("jwst-nircam_zp_vega-sirius",)

# Theoretical (non-magnitude) columns as named in the BaSTI header.
_THEORY_COLS = ["M/Mo(ini)", "M/Mo(fin)", "log(L/Lo)", "logTe"]

# Solar constants for derived quantities
_LOGG_SUN = 4.4374
_TEFF_SUN = 5772.0


# ---------------------------------------------------------------------------
# Raw-file parsing
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r"#\s*Np\s*=\s*(\d+)\s+\[M/H\]\s*=\s*([\-\+\d\.]+)\s+Z\s*=\s*([\d\.]+)"
    r"\s+Y\s*=\s*([\d\.]+)\s+Age\s*\(Myr\)\s*=\s*([\d\.]+)"
)


def parse_isc_file(filename):
    """Parse one BaSTI .isc_<system> file into a list of per-age blocks.

    Handles one or many age blocks per file. Returns a list of dicts with
    keys: np, mh, z, y, age_myr, columns (list of str), data (2-D ndarray).
    Raises ValueError if any block's row count differs from its header Np,
    or if Np != BASTI_NP (which would corrupt the row-number pseudo-EEP).
    """
    blocks = []
    meta = None
    columns = None
    rows = []

    def _close_block():
        if meta is None:
            return
        data = np.array(rows, dtype=float)
        if len(data) != meta["np"]:
            raise ValueError(
                "{}: block at age {} Myr has {} rows, header says Np={}".format(
                    filename, meta["age_myr"], len(data), meta["np"]
                )
            )
        if meta["np"] != BASTI_NP:
            raise ValueError(
                "{}: Np={} != expected {}. The row-number pseudo-EEP relies on "
                "a fixed point count; refusing to ingest. If BaSTI changed "
                "their isochrone structure, re-verify the segment boundaries "
                "and update BASTI_NP / BASTI_TRGB_ROW (and bump "
                "BASTI_GRID_VERSION).".format(filename, meta["np"], BASTI_NP)
            )
        blocks.append(dict(meta, columns=list(columns), data=data))

    with open(filename, encoding="latin-1") as fin:
        for line in fin:
            if line.startswith("#"):
                m = _HEADER_RE.search(line)
                if m:
                    _close_block()
                    rows = []
                    meta = {
                        "np": int(m.group(1)),
                        "mh": float(m.group(2)),
                        "z": float(m.group(3)),
                        "y": float(m.group(4)),
                        "age_myr": float(m.group(5)),
                    }
                    continue
                s = line.lstrip("#").strip()
                if s.startswith("M/Mo(ini)"):
                    columns = s.split()
                continue
            s = line.strip()
            if s:
                rows.append(s.split())
    _close_block()

    if not blocks:
        raise ValueError("{}: no isochrone blocks found".format(filename))
    return blocks


def _rename_mag_columns(columns, system):
    """Map raw BaSTI magnitude column names -> canonical band tokens."""
    rule = PHOT_SYSTEMS[system]["rename"]
    out = []
    for c in columns:
        if c in _THEORY_COLS:
            out.append(c)
        elif rule is None:
            out.append(c)
        elif isinstance(rule, str):
            out.append(rule + c)
        else:
            out.append(rule.get(c, c))
    return out


def blocks_to_df(blocks, system, feh):
    """Stack the blocks of one file into a long DataFrame with metadata cols."""
    dfs = []
    for b in blocks:
        cols = _rename_mag_columns(b["columns"], system)
        df = pd.DataFrame(b["data"], columns=cols)
        df["EEP"] = np.arange(len(df), dtype=float)
        df["log10_isochrone_age_yr"] = np.log10(b["age_myr"] * 1e6)
        df["feh"] = feh
        df["mh"] = b["mh"]
        df["Z"] = b["z"]
        df["Y"] = b["y"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

class BastiIsochroneGrid(StellarModelGrid):
    """BaSTI-IAC isochrone grid at fixed [a/Fe], physics case O1D1E1.

    Index: (log10_isochrone_age_yr, feh, EEP) -- identical layout to
    MISTIsochroneGrid, with EEP = raw row number (0..2099). All requested
    photometric systems are merged column-wise, with theoretical columns
    cross-validated between systems at ingestion.
    """

    name = "basti"
    eep_col = "EEP"
    age_col = "log10_isochrone_age_yr"
    feh_col = "feh"
    mass_col = "M/Mo(fin)"
    initial_mass_col = "M/Mo(ini)"
    logTeff_col = "logTe"
    logL_col = "log(L/Lo)"
    logg_col = "logg"          # computed, see compute_additional_columns

    index_cols = ("log10_isochrone_age_yr", "feh", "EEP")
    filename_pattern = r"\.isc_"
    eep_replaces = "mass"
    is_full = False

    default_kwargs = {"afe": 0.0, "case": DEFAULT_CASE, "systems": DEFAULT_SYSTEMS,
                      "age_range": None}

    n_eep = BASTI_NP
    bounds = (("eep", (0, BASTI_NP)),)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        afe = self.kwargs["afe"]
        if afe not in AFE_TAGS:
            raise ValueError(
                "BaSTI [a/Fe] must be one of {} (got {})".format(sorted(AFE_TAGS), afe)
            )
        self.afe = afe
        self.case = self.kwargs["case"]
        self.systems = tuple(SYSTEM_ALIASES.get(s, s) for s in self.kwargs["systems"])
        # (min_gyr, max_gyr) or None. The BaSTI grids carry ~200-320 ages per
        # composition; restricting to the science range at build time keeps
        # the assembled grid and the DFInterpolator array tractable.
        self.age_range = self.kwargs.get("age_range", None)
        for s in self.systems:
            if s not in PHOT_SYSTEMS:
                raise ValueError(
                    "Unknown BaSTI photometric system '{}'. Known: {}. Add it to "
                    "PHOT_SYSTEMS in basti/models.py.".format(s, sorted(PHOT_SYSTEMS))
                )

    # -- paths / tags -------------------------------------------------------

    @property
    def afe_tag(self):
        return AFE_TAGS[self.afe]

    @property
    def kwarg_tag(self):
        # systems are part of the tag so grids with different band sets
        # cannot share (and thus corrupt) each other's caches
        sys_tag = "-".join(sorted(s.split("_")[0][:8] for s in self.systems))
        tag = "_v{}_{}{}_{}".format(
            BASTI_GRID_VERSION, self.afe_tag, self.case, sys_tag
        )
        if self.age_range is not None:
            tag += "_age{:g}-{:g}".format(*self.age_range)
        return tag

    @property
    def datadir(self):
        return os.path.join(ISOCHRONES, self.name)

    def get_directory_path(self, **kwargs):
        # Flat, shared across [a/Fe] and systems -- parallel to
        # mist/MIST_v2.5_vvcrit0.4_full_isos/. Alpha is selected by filename.
        return os.path.join(self.datadir, "BaSTI_{}_isos".format(self.case))

    def get_existing_filenames(self, **kwargs):
        d = self.get_directory_path(**kwargs)
        if not os.path.exists(d):
            raise FileNotFoundError(
                "No BaSTI data at {}.\nDownload it with:\n"
                "    python download_basti.py --what isos --scrape".format(d)
            )
        return sorted(
            f for f in glob.glob(os.path.join(d, "*"))
            if re.search(self.filename_pattern, os.path.basename(f))
        )

    def get_filenames(self, system):
        # select this grid's [a/Fe] by the alpha tag embedded in the filename
        # (e.g. '...y269P00O1D1E1.isc_...'), then this system by extension --
        # the BaSTI analogue of the v2.5 grid's '_afe_p4_' filename filter.
        # CASE-INSENSITIVE: tarball members use lowercase tags (m02o1d1e1)
        # while web-interface downloads use uppercase (M02O1D1E1).
        token = "{}{}.".format(self.afe_tag, self.case).lower()
        suffix = ".isc_{}".format(system).lower()
        files = [f for f in self.get_existing_filenames()
                 if f.lower().endswith(suffix)
                 and token in os.path.basename(f).lower()]
        if self.age_range is not None:
            lo, hi = self.age_range
            def _in_range(f):
                m = re.match(r"(\d+)z", os.path.basename(f))
                if not m:
                    return True   # unparseable age -> keep; header decides
                age_gyr = int(m.group(1)) / 1000.0
                return lo <= age_gyr <= hi
            files = [f for f in files if _in_range(f)]
        if not files:
            raise FileNotFoundError(
                "No '{}' files for [a/Fe]={} (tag {}) in {}. Download with:\n"
                "    python download_basti.py --what isos --scrape "
                "--afe {} --iso-systems {}".format(
                    suffix, self.afe, self.afe_tag, self.get_directory_path(),
                    self.afe, system
                )
            )
        return files

    # -- assembly -----------------------------------------------------------

    def mh_to_feh(self, mh):
        return mh - MH_TO_FEH_OFFSET[self.afe]

    def df_one_system(self, system):
        """All (age, feh) isochrones of one photometric system, stacked."""
        dfs = []
        for f in self.get_filenames(system):
            for b_group, blocks in [(f, parse_isc_file(f))]:
                mh = blocks[0]["mh"]
                feh = round(self.mh_to_feh(mh), 3)
                dfs.append(blocks_to_df(blocks, system, feh))
        df = pd.concat(dfs, ignore_index=True)
        # Guard against duplicate isochrones (e.g. a hand-downloaded file with
        # long-form Z naming coexisting with the tarball-extracted short-form
        # file for the same composition/age): keep the first occurrence.
        keys = ["log10_isochrone_age_yr", "feh", "EEP"]
        n0 = len(df)
        df = df.drop_duplicates(subset=keys, keep="first")
        if len(df) < n0:
            getLogger().warning(
                "System '{}': dropped {} duplicate (age, feh, EEP) rows -- "
                "the isos directory likely contains the same isochrone under "
                "two filenames (e.g. manual + tarball-extracted copies). "
                "Consider deleting the redundant files.".format(
                    system, n0 - len(df)))
        return df

    def df_all(self):
        """Merge all requested systems into one wide DataFrame.

        Theoretical columns are taken from the first system and cross-checked
        against every other system (they must be the same isochrones row for
        row); magnitude columns are appended.
        """
        getLogger().info(
            "Building BaSTI grid: afe={} case={} systems={}".format(
                self.afe, self.case, self.systems
            )
        )
        base = None
        keys = ["log10_isochrone_age_yr", "feh", "EEP"]
        for system in self.systems:
            df = self.df_one_system(system).sort_values(keys).reset_index(drop=True)
            if base is None:
                base = df
                continue
            merged = base[keys].merge(df[keys], on=keys, how="outer", indicator=True)
            if (merged["_merge"] != "both").any():
                n_l = (merged["_merge"] == "left_only").sum()
                n_r = (merged["_merge"] == "right_only").sum()
                getLogger().warning(
                    "System '{}' (age, feh) coverage differs from '{}': "
                    "{} rows only in base, {} only in '{}'. Missing magnitudes "
                    "will be NaN.".format(system, self.systems[0], n_l, n_r, system)
                )
            # cross-validate theory columns where both exist
            chk = base.merge(
                df[keys + _THEORY_COLS], on=keys, suffixes=("", "_x"), how="inner"
            )
            for c in _THEORY_COLS:
                bad = ~np.isclose(chk[c], chk[c + "_x"], rtol=0, atol=5e-5)
                if bad.any():
                    raise ValueError(
                        "Theoretical column '{}' differs between systems '{}' and "
                        "'{}' in {} rows -- files are not from the same physics "
                        "case / release. Aborting merge.".format(
                            c, self.systems[0], system, int(bad.sum())
                        )
                    )
            mag_cols = [
                c for c in df.columns
                if c not in keys + _THEORY_COLS + ["mh", "Z", "Y"]
            ]
            base = base.merge(df[keys + mag_cols], on=keys, how="outer")

        base = base.sort_values(by=list(self.index_cols))
        base.index = [base[c] for c in self.index_cols]
        return base

    # -- standardized columns ------------------------------------------------

    @property
    def band_columns(self):
        """Canonical magnitude column names present for the chosen systems."""
        cols = []
        for system in self.systems:
            rule = PHOT_SYSTEMS[system]["rename"]
            if rule is None or isinstance(rule, str):
                # discovered at parse time; cache from df
                cols.extend(
                    c for c in self.df.columns
                    if c not in self.default_columns and c not in ("mh", "Z", "Y", "dm_deep")
                )
                return list(dict.fromkeys(cols))
            cols.extend(rule.values())
        return list(dict.fromkeys(cols))

    @property
    def default_columns(self):
        return (
            "eep", "age", "feh", "mass", "initial_mass", "radius", "density",
            "logTeff", "Teff", "logg", "logL", "Mbol", "mh", "Z", "Y",
        )

    @property
    def prop_map(self):
        return dict(
            eep=self.eep_col,
            age=self.age_col,
            feh=self.feh_col,
            mass=self.mass_col,
            initial_mass=self.initial_mass_col,
            logTeff=self.logTeff_col,
            logL=self.logL_col,
        )

    def compute_additional_columns(self, df):
        """Derive Teff, logg, radius, density, Mbol from (M, logL, logTe).

        Skips the base implementation, which expects MIST's log_R / log_g
        columns; BaSTI provides only mass, logL, logTe, so everything is
        derived from those plus solar constants.
        """
        from ..models import MSUN, RSUN  # local import: match package values

        df["Teff"] = 10 ** df["logTeff"]
        df["Mbol"] = 4.74 - 2.5 * df["logL"]
        # R/Rsun = (L/Lsun)^0.5 (Teff/Teff_sun)^-2
        df["radius"] = 10 ** (0.5 * df["logL"]) * (_TEFF_SUN / df["Teff"]) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            df["logg"] = (
                _LOGG_SUN
                + np.log10(df["mass"])
                + 4.0 * (df["logTeff"] - np.log10(_TEFF_SUN))
                - df["logL"]
            )
        df["density"] = df["mass"] * MSUN / (4.0 / 3 * np.pi * (df["radius"] * RSUN) ** 3)
        return df

    def get_df(self, orig=False):
        df = self.df_all()
        if orig:
            return df
        df = df.rename(columns=self.column_map)
        df = self.compute_additional_columns(df)
        keep = list(self.default_columns) + [
            c for c in df.columns
            if c not in self.default_columns
            and c not in self.prop_map.values()
            and c not in _THEORY_COLS
        ]
        # preserve column uniqueness and order: standard cols first, then bands
        keep = list(dict.fromkeys(keep))
        return df[keep]

    # -- caches ---------------------------------------------------------------

    @property
    def hdf_filename(self):
        return os.path.join(self.datadir, "basti{}.h5".format(self.kwarg_tag))

    @property
    def interp_grid_npz_filename(self):
        return os.path.join(self.datadir, "full_grid{}.npz".format(self.kwarg_tag))

    def write_hdf(self, orig=False):
        # Same as Grid.write_hdf but passes `key` by keyword: the stock
        # positional call breaks on pandas >= 2.0. Harmless on pandas < 2.
        df = self.get_df(orig=orig)
        path = "orig" if orig else "df"
        df.to_hdf(self.hdf_filename, key=path)
        getLogger().info("{} written to {}.".format(path, self.hdf_filename))
        return df

    @property
    def df(self):
        if self._df is None:
            self._df = self.read_hdf()
            self._df["dm_deep"] = self.get_dm_deep()
        return self._df

    def get_dm_deep(self, compute=False):
        """Vectorized d(initial_mass)/d(EEP) per (age, feh) isochrone.

        (Same replacement as in models_v2p5: the base-class loop misreads a
        3-level MultiIndex with df.loc[f, a] and writes to a copy.)
        """
        filename = os.path.join(self.datadir, "dm_deep{}.h5".format(self.kwarg_tag))
        if os.path.exists(filename):
            try:
                return pd.read_hdf(filename, "dm_deep")
            except Exception:
                pass

        df = self.read_hdf()

        # positional per-isochrone gradient: immune to the pandas quirk where
        # groupby.apply on Series-returning functions yields a DataFrame when
        # there is a single group (e.g. narrow age_range selections).
        mini = df["initial_mass"].to_numpy(dtype=float)
        eep = df["eep"].to_numpy(dtype=float)
        vals = np.full(len(df), np.nan)
        for _, pos in df.groupby(level=[0, 1]).indices.items():
            if len(pos) >= 2:
                vals[pos] = np.gradient(mini[pos], eep[pos])
        dm_deep = pd.Series(vals, index=df.index, name="dm_deep")
        dm_deep = dm_deep.replace([np.inf, -np.inf], np.nan)
        dm_deep.to_hdf(filename, key="dm_deep")
        return dm_deep

    # -- convenience ----------------------------------------------------------

    @property
    def fehs(self):
        return np.array(sorted(self.df.index.levels[1]))

    @property
    def ages(self):
        return np.array(sorted(self.df.index.levels[0]))
      
