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
    rows = []
    for r in valid.itertuples(index=False):
        scheme = _SERVICE_SCHEMES[r.scheme]
        match = p[(p["state_key"] == r.state_key) & (p["district_key"] == r.district_key)]
        survey = float(match[scheme.nfhs_indicator].iloc[0]) if len(match) == 1 else float("nan")
        mis_cov = 100.0 * r.enrolled / r.denominator if r.comparable else float("nan")
        rows.append({
            "state": r.state, "district": r.district, "scheme_key": r.scheme,
            "period": r.period.date().isoformat(), "mis_enrolled": r.enrolled,
            "mis_denominator": r.denominator, "denominator_type": r.denominator_type,
            "mis_coverage_pct": mis_cov, "nfhs_proxy_pct": survey,
            "proxy_minus_mis_points": survey - mis_cov if np.isfinite(mis_cov) and np.isfinite(survey) else float("nan"),
            "profile_match": len(match) == 1,
            "mis_coverage_over_100": bool(mis_cov > 100) if np.isfinite(mis_cov) else False,
            "source": r.source,
        })
    return pd.DataFrame(rows)
