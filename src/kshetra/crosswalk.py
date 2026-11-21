"""Join NFHS-5, Census 2011 (MPI table) and Census SC/ST tables into one district frame.

Matching order for each MPI (Census 2011) district:
1. exact match on normalised state + district;
2. fuzzy match within the same state (difflib ratio >= FUZZY_CUTOFF, unique best match);
   Andhra Pradesh also searches Telangana because Telangana districts sat under AP in 2011.
Unmatched districts are kept in the output with ``mpi_match`` = "none" so the gap is visible.
"""

from __future__ import annotations

import difflib

import pandas as pd

from .config import AUX_INDICATORS, INDICATORS
from .ingest import load_census_scst, load_mpi, load_nfhs
from .text import norm_state

FUZZY_CUTOFF = 0.9

# Known state-name differences between the sources, applied after norm_state.
_STATE_FIX = {"MAHARASTRA": "MAHARASHTRA", "LADAKH": "JAMMU AND KASHMIR"}


def _fix_state(key: str) -> str:
    return _STATE_FIX.get(norm_state(key), norm_state(key))


def match_mpi_to_nfhs(nfhs: pd.DataFrame, mpi: pd.DataFrame) -> pd.DataFrame:
    """Link each MPI (Census 2011) row to an NFHS-5 district.

    Returns a frame aligned to ``mpi`` with the matched NFHS state and district keys
    (NaN when unmatched) and ``mpi_match`` in {"exact", "fuzzy", "none"}.
    """
    nfhs = nfhs.assign(state_key=nfhs["state_key"].map(_fix_state))
    mpi = mpi.assign(state_key=mpi["state_key"].map(_fix_state))

    exact = set(zip(nfhs["state_key"], nfhs["district_key"]))
    rows = []
    for _, row in mpi.iterrows():
        s, d = row["state_key"], row["district_key"]
        if (s, d) in exact:
            rows.append((s, d, "exact"))
            continue
        # Telangana districts were listed under Andhra Pradesh in Census 2011.
        pool_states = {s, "TELANGANA"} if s == "ANDHRA PRADESH" else {s}
        pool = nfhs[nfhs["state_key"].isin(pool_states)]
        scored = sorted(
            ((difflib.SequenceMatcher(None, d, c).ratio(), st, c)
             for st, c in zip(pool["state_key"], pool["district_key"])),
            reverse=True,
        )
        # Accept only a clear, unique best match above the cutoff.
        if scored and scored[0][0] >= FUZZY_CUTOFF and (len(scored) == 1 or scored[1][0] < scored[0][0]):
            rows.append((scored[0][1], scored[0][2], "fuzzy"))
        else:
            rows.append((None, None, "none"))
    return pd.DataFrame(rows, columns=["nfhs_state_key", "nfhs_district_key", "mpi_match"], index=mpi.index)


def build_district_frame() -> pd.DataFrame:
    """One row per NFHS-5 district with indicators, MPI HCR (where matched) and SC/ST shares."""
    nfhs = load_nfhs()
    nfhs["state_key"] = nfhs["state_key"].map(_fix_state)
    mpi = load_mpi()
    census = load_census_scst()
    census["state_key"] = census["state_key"].map(_fix_state)

    # Link each Census 2011 (MPI) district to an NFHS-5 district, then keep one MPI row per NFHS district.
    link = match_mpi_to_nfhs(nfhs, mpi)
    linked = pd.concat([mpi, link], axis=1)
    linked = linked[linked["mpi_match"] != "none"]
    linked = linked.drop_duplicates(subset=["nfhs_state_key", "nfhs_district_key"], keep="first")

    mpi_cols = ["mpi_hcr_pct", "census_population", "census_households", "census_sc_pct",
                "census_st_pct", "census_literacy_pct", "census_unemployed_pct"]
    mpi_for_join = pd.DataFrame(
        {
            "state_key": linked["nfhs_state_key"].values,
            "district_key": linked["nfhs_district_key"].values,
            "mpi_match": linked["mpi_match"].values,
            **{c: linked[c].values for c in mpi_cols},
        }
    ).set_index(["state_key", "district_key"])

    df = nfhs.join(mpi_for_join, on=["state_key", "district_key"])
    df["mpi_match"] = df["mpi_match"].fillna("none")
    df = df.merge(census, on=["state_key", "district_key"], how="left")
    df["scst_published"] = df["scst_published"].fillna(False).astype(bool)

    ordered = ["state_raw", "district_raw", "state_key", "district_key", "survey_households"]
    ordered += list(INDICATORS)
    ordered += list(AUX_INDICATORS)
    ordered += ["mpi_hcr_pct", "mpi_match"] + mpi_cols[1:]
    ordered += ["census_scst_pct", "scst_published"]
    return df[ordered].reset_index(drop=True)
