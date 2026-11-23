"""Administrative MIS beneficiary tables compared against NFHS-5 survey proxies.

Expected input: one row per district, scheme and period with the official enrolment count
and the denominator used by the programme. Provenance columns are mandatory. The loader
does not download MIS data; supply CSV exports from the official dashboard you cite.

Comparisons are diagnostics. MIS enrolment and survey household coverage use different
definitions, periods and denominators, so a gap is a question for field verification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemes import SCHEME_PROXIES
from .text import norm_name, norm_state

MIS_REQUIRED = {"state", "district", "scheme", "period", "enrolled", "denominator", "denominator_type", "source"}
_SERVICE_SCHEMES = {s.key: s for s in SCHEME_PROXIES if s.planner_eligible}


def load_mis_table(data: pd.DataFrame) -> pd.DataFrame:
    """Validate an MIS export. Rows with blank or zero denominators are rejected, not imputed."""
    missing = MIS_REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"MIS table missing columns: {', '.join(sorted(missing))}")
    out = data.copy()
    for col in ("state", "district", "scheme", "denominator_type", "source"):
        out[col] = out[col].astype(str).str.strip()
        if out[col].eq("").any():
            raise ValueError(f"{col} must not be blank")
    unknown = sorted(set(out["scheme"]) - set(_SERVICE_SCHEMES))
    if unknown:
        raise ValueError(f"scheme keys not comparable to a service-access proxy: {unknown}; expected {sorted(_SERVICE_SCHEMES)}")
    out["period"] = pd.to_datetime(out["period"], errors="coerce")
    if out["period"].isna().any():
        raise ValueError("period must be a valid date or YYYY-MM")
    out["enrolled"] = pd.to_numeric(out["enrolled"], errors="coerce")
    out["denominator"] = pd.to_numeric(out["denominator"], errors="coerce")
    if out["enrolled"].isna().any() or (out["enrolled"] < 0).any():
        raise ValueError("enrolled must be a non-negative number on every row")
    out["state_key"] = out["state"].map(norm_state)
    out["district_key"] = out["district"].map(norm_name)
    if out.duplicated(["state_key", "district_key", "scheme", "period"]).any():
        raise ValueError("duplicate district/scheme/period rows in MIS table")
    out["comparable"] = out["denominator"].gt(0)
    return out


def mis_vs_survey(mis: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Compare MIS coverage (enrolled / denominator) with the NFHS-5 service-access proxy per district."""
    valid = load_mis_table(mis) if "state_key" not in mis.columns else mis
    p = profiles.copy()
    if "state_key" not in p:
        p["state_key"] = p["state_raw"].map(norm_state)
    if "district_key" not in p:
        p["district_key"] = p["district_raw"].map(norm_name)
    v = valid.copy()
    v["scheme_key"] = v["scheme"]
    v["_survey_col"] = v["scheme"].map(lambda k: _SERVICE_SCHEMES[k].nfhs_indicator)
    keys = p[["state_key", "district_key"]].copy()
    keys["_n"] = 1
    counts = keys.groupby(["state_key", "district_key"])["_n"].sum().rename("profile_rows").reset_index()
    # Wide lookup: one value per (state, district, scheme) using each scheme's own NFHS column.
    long = []
    for key in sorted(set(v["scheme"])):
        scheme = _SERVICE_SCHEMES[key]
        if scheme.nfhs_indicator not in p.columns:
            raise ValueError(f"profile column missing for scheme {key}: {scheme.nfhs_indicator}")
        part = p[["state_key", "district_key", scheme.nfhs_indicator]].rename(
            columns={scheme.nfhs_indicator: "nfhs_proxy_pct"})
        part["scheme"] = key
        long.append(part)
    lookup = pd.concat(long, ignore_index=True)
    merged = v.merge(counts, on=["state_key", "district_key"], how="left")
    merged = merged.merge(lookup, on=["state_key", "district_key", "scheme"], how="left")
    merged["profile_rows"] = merged["profile_rows"].fillna(0).astype(int)
    merged["profile_match"] = merged["profile_rows"].eq(1)
    merged.loc[~merged["profile_match"], "nfhs_proxy_pct"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        mis_cov = np.where(merged["comparable"], 100.0 * merged["enrolled"] / merged["denominator"], np.nan)
    merged["mis_coverage_pct"] = mis_cov
    survey = merged["nfhs_proxy_pct"].to_numpy(dtype=float)
    merged["proxy_minus_mis_points"] = np.where(np.isfinite(mis_cov) & np.isfinite(survey), survey - mis_cov, np.nan)
    merged["mis_coverage_over_100"] = np.where(np.isfinite(mis_cov), mis_cov > 100, False).astype(bool)
    out = pd.DataFrame({
        "state": merged["state"], "district": merged["district"], "scheme_key": merged["scheme"],
        "period": merged["period"].dt.date.map(lambda d: d.isoformat()), "mis_enrolled": merged["enrolled"],
        "mis_denominator": merged["denominator"], "denominator_type": merged["denominator_type"],
        "mis_coverage_pct": merged["mis_coverage_pct"], "nfhs_proxy_pct": merged["nfhs_proxy_pct"],
        "proxy_minus_mis_points": merged["proxy_minus_mis_points"], "profile_match": merged["profile_match"],
        "mis_coverage_over_100": merged["mis_coverage_over_100"], "source": merged["source"],
    })
    return out.reset_index(drop=True)
