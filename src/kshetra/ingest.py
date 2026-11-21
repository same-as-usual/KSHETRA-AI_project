"""Loaders that turn the three raw district datasets into tidy frames.

Every loader returns a frame keyed by (state_key, district_key), where the keys are
normalised with kshetra.text so the datasets can be joined (see crosswalk.py).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CENSUS_SCST_FILE, INDICATORS, MPI_FILE, NFHS_FILE
from .text import norm_name, norm_state, parse_number


def _find_column(columns: list[str], prefix: str) -> str:
    """Return the single column whose stripped name starts with ``prefix``."""
    hits = [c for c in columns if c.strip().startswith(prefix)]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one NFHS column starting with {prefix!r}, found {len(hits)}")
    return hits[0]


def load_nfhs(path: Path = NFHS_FILE) -> pd.DataFrame:
    """NFHS-5 district fact sheet (2019-21): deprivation indicators per district.

    Columns returned: state_key, district_key, state_raw, district_raw, survey_households,
    plus one float column per indicator in ``INDICATORS``.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]

    out = pd.DataFrame(
        {
            "state_raw": raw["State/UT"].str.strip(),
            "district_raw": raw["District Names"].str.strip(),
        }
    )
    out["state_key"] = out["state_raw"].map(norm_state)
    out["district_key"] = out["district_raw"].map(norm_name)
    out["survey_households"] = raw["Number of Households surveyed"].map(parse_number)

    for short, (_domain, _direction, prefix) in INDICATORS.items():
        col = _find_column(list(raw.columns), prefix)
        out[short] = raw[col].map(parse_number)

    dupes = out.duplicated(["state_key", "district_key"]).sum()
    if dupes:
        raise ValueError(f"NFHS file has {dupes} duplicate district keys")
    return out


def load_mpi(path: Path = MPI_FILE) -> pd.DataFrame:
    """Census 2011 district table with a multidimensional poverty headcount ratio (MPI HCR).

    Notes on the raw file (see docs/DATA_QUALITY.md):
    * the header has a trailing empty column and some rows are exact duplicates; both are dropped;
    * the ``Households`` column does not equal rural + urban households, so we use
      ``Households.1`` (= rural + urban) as the household count;
    * some district keys appear with conflicting values; both rows are excluded (with a warning).
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw = raw.loc[:, [c for c in raw.columns if c.strip() and not c.startswith("Unnamed")]]
    raw = raw.drop_duplicates()

    out = pd.DataFrame(
        {
            "state_raw": raw["State"].str.strip(),
            "district_raw": raw["District"].str.strip(),
        }
    )
    out["state_key"] = out["state_raw"].map(norm_state)
    out["district_key"] = out["district_raw"].map(norm_name)
    out["mpi_hcr_pct"] = raw["MPI HCR"].map(parse_number)
    pop = raw["Total Population"].map(parse_number)
    out["census_population"] = pop
    out["census_households"] = raw["Households.1"].map(parse_number)
    out["census_sc_pct"] = 100 * raw["SC"].map(parse_number) / pop
    out["census_st_pct"] = 100 * raw["ST"].map(parse_number) / pop
    out["census_literacy_pct"] = 100 * raw["Literate Population"].map(parse_number) / pop
    working = raw["Total Working Population"].map(parse_number)
    out["census_unemployed_pct"] = 100 * raw["Unemployed Population"].map(parse_number) / working.replace(0, np.nan)

    # Keys that still occur more than once carry conflicting values (e.g. two different MPI HCRs
    # for the same district). We cannot tell which is right, so both rows are excluded.
    conflict = out.duplicated(["state_key", "district_key"], keep=False)
    if conflict.any():
        warnings.warn(
            "Excluding conflicting MPI rows: "
            + ", ".join(f"{s}/{d}" for s, d in out.loc[conflict, ["state_key", "district_key"]].drop_duplicates().values)
        )
    out = out.loc[~conflict].reset_index(drop=True)
    return out


def load_census_scst(path: Path = CENSUS_SCST_FILE) -> pd.DataFrame:
    """Census 2011 SC/ST shares, keyed by NFHS-5 district names.

    Only rows whose ``Source`` is ``PUBLISHED`` contain district-level Census values.
    Rows marked ``STATE_MEAN_ESTIMATED`` carry the state average, not the district value,
    so their SC/ST shares are set to NaN here to avoid treating estimates as observations.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = pd.DataFrame(
        {
            "state_key": raw["State_NFHS5"].str.strip().map(norm_state),
            "district_key": raw["District_NFHS5"].str.strip().map(norm_name),
        }
    )
    published = raw["Source"].str.strip().eq("PUBLISHED")
    out["scst_published"] = published
    out["census_scst_pct"] = raw["SCST_Combined_Pct"].map(parse_number).where(published)
    return out
