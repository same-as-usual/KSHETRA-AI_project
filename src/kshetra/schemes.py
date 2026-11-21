"""SchemeGap and action-planning helpers.

These are service-coverage proxies from NFHS-5, not scheme enrolment or eligibility data.
No beneficiary count, amount owed, or eligibility decision is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SchemeProxy:
    key: str
    name: str
    ministry: str
    nfhs_indicator: str
    gap_label: str
    interpretation: str
    url: str


SCHEME_PROXIES = [
    SchemeProxy(
        key="pmuy",
        name="Pradhan Mantri Ujjwala Yojana (PMUY)",
        ministry="Ministry of Petroleum and Natural Gas",
        nfhs_indicator="clean_cooking_fuel_pct",
        gap_label="Households not using clean cooking fuel",
        interpretation="NFHS clean-fuel use is a service-access proxy; it does not identify PMUY beneficiaries or LPG connection ownership.",
        url="https://www.pmuy.gov.in/",
    ),
    SchemeProxy(
        key="sbm_g",
        name="Swachh Bharat Mission (Gramin) (SBM-G)",
        ministry="Department of Drinking Water and Sanitation",
        nfhs_indicator="improved_sanitation_pct",
        gap_label="Population without an improved sanitation facility",
        interpretation="NFHS sanitation coverage is an outcome proxy; it is not SBM-G coverage or eligibility. This indicator is not rural-only.",
        url="https://swachhbharatmission.ddws.gov.in/",
    ),
    SchemeProxy(
        key="pmjay_proxy",
        name="Ayushman Bharat PM-JAY (coverage proxy only)",
        ministry="National Health Authority",
        nfhs_indicator="health_insurance_pct",
        gap_label="Households without any reported health-insurance / financing cover",
        interpretation="NFHS reports any household member with health insurance/financing; it does not isolate PM-JAY beneficiaries or verify eligibility.",
        url="https://pmjay.gov.in/",
    ),
]


def scheme_gap_frame(profiles: pd.DataFrame) -> pd.DataFrame:
    """Long-form service-gap proxy table, preserving NaN as unavailable rather than zero."""
    required = {"state_raw", "district_raw"}
    if not required.issubset(profiles.columns):
        raise ValueError(f"profiles missing required fields: {sorted(required - set(profiles.columns))}")
    rows = []
    for _, row in profiles.iterrows():
        for scheme in SCHEME_PROXIES:
            covered = row.get(scheme.nfhs_indicator, float("nan"))
            covered = pd.to_numeric(pd.Series([covered]), errors="coerce").iloc[0]
            rows.append({
                "state_raw": row["state_raw"],
                "district_raw": row["district_raw"],
                "state_key": row.get("state_key", ""),
                "district_key": row.get("district_key", ""),
                "scheme_key": scheme.key,
                "scheme": scheme.name,
                "ministry": scheme.ministry,
                "coverage_proxy_pct": covered,
                "gap_proxy_pct": (100.0 - covered) if pd.notna(covered) else float("nan"),
                "gap_label": scheme.gap_label,
                "interpretation": scheme.interpretation,
                "source_url": scheme.url,
                "survey_households": row.get("survey_households", float("nan")),
                "survey_period": "NFHS-5, 2019-21",
            })
    return pd.DataFrame(rows)


def plan_budget(gaps: pd.DataFrame, budget_inr: float) -> pd.DataFrame:
    """Allocate a user-entered scenario budget in proportion to each available gap proxy.

    This is a transparent budget-share scenario, not a cost optimizer, beneficiary forecast,
    or impact estimate. Uses the latest row per scheme for the selected district.
    """
    if budget_inr < 0:
        raise ValueError("budget_inr must be non-negative")
    if gaps.empty:
        return gaps.assign(allocation_inr=pd.Series(dtype=float), budget_share_pct=pd.Series(dtype=float))
    out = gaps.copy()
    out["gap_proxy_pct"] = pd.to_numeric(out["gap_proxy_pct"], errors="coerce")
    out = out.dropna(subset=["gap_proxy_pct"]).copy()
    total = out["gap_proxy_pct"].clip(lower=0).sum()
    if total <= 0:
        out["budget_share_pct"] = 0.0
        out["allocation_inr"] = 0.0
    else:
        out["budget_share_pct"] = 100 * out["gap_proxy_pct"].clip(lower=0) / total
        out["allocation_inr"] = budget_inr * out["budget_share_pct"] / 100
    return out.sort_values("allocation_inr", ascending=False).reset_index(drop=True)
