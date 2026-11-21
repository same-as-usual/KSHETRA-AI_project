"""SchemeGap and action-planning helpers.

The NFHS-5 fields below are broad service/outcome proxies, not scheme enrollment,
beneficiary counts, or eligibility. Only the four service-access proxies are used by
the budget-share scenario; nutrition and education outcome indicators are shown for
context but are not mixed into its allocations.
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
    proxy_kind: str
    proxy_measure_label: str
    gap_label: str
    direction: str
    planner_eligible: bool
    interpretation: str
    url: str


SCHEME_PROXIES = [
    SchemeProxy(
        key="pmuy", name="Pradhan Mantri Ujjwala Yojana (PMUY)",
        ministry="Ministry of Petroleum and Natural Gas", nfhs_indicator="clean_cooking_fuel_pct",
        proxy_kind="Service-access proxy", proxy_measure_label="Households using clean cooking fuel",
        gap_label="Households not using clean cooking fuel", direction="coverage",
        planner_eligible=True,
        interpretation="NFHS clean-fuel use is a service-access proxy; it does not identify PMUY beneficiaries or LPG connection ownership.",
        url="https://www.pmuy.gov.in/",
    ),
    SchemeProxy(
        key="sbm_g", name="Swachh Bharat Mission (Gramin) (SBM-G)",
        ministry="Department of Drinking Water and Sanitation", nfhs_indicator="improved_sanitation_pct",
        proxy_kind="Service-access proxy", proxy_measure_label="Population using an improved sanitation facility",
        gap_label="Population without an improved sanitation facility", direction="coverage",
        planner_eligible=True,
        interpretation="NFHS sanitation coverage is an outcome proxy; it is not SBM-G coverage or eligibility. This indicator is not rural-only.",
        url="https://swachhbharatmission.ddws.gov.in/",
    ),
    SchemeProxy(
        key="pmjay_proxy", name="Ayushman Bharat PM-JAY (coverage proxy only)",
        ministry="National Health Authority", nfhs_indicator="health_insurance_pct",
        proxy_kind="Service-access proxy", proxy_measure_label="Households with any health-insurance / financing cover",
        gap_label="Households without any reported health-insurance / financing cover", direction="coverage",
        planner_eligible=True,
        interpretation="NFHS reports any household member with health insurance/financing; it does not isolate PM-JAY beneficiaries or verify eligibility.",
        url="https://pmjay.gov.in/",
    ),
    SchemeProxy(
        key="jal_jeevan", name="Jal Jeevan Mission (service proxy only)",
        ministry="Ministry of Jal Shakti", nfhs_indicator="improved_water_pct",
        proxy_kind="Service-access proxy", proxy_measure_label="Population using an improved drinking-water source",
        gap_label="Population without an improved drinking-water source", direction="coverage",
        planner_eligible=True,
        interpretation="NFHS improved-water access is not a functional household tap connection and does not measure Jal Jeevan Mission coverage.",
        url="https://jaljeevanmission.gov.in/",
    ),
    SchemeProxy(
        key="poshan", name="POSHAN Abhiyaan (nutrition outcome proxy)",
        ministry="Ministry of Women and Child Development", nfhs_indicator="stunted_pct",
        proxy_kind="Nutrition-outcome proxy", proxy_measure_label="Children under 5 with stunting",
        gap_label="Child stunting prevalence (nutrition-burden proxy)", direction="burden",
        planner_eligible=False,
        interpretation="NFHS stunting is a nutrition outcome, not POSHAN Abhiyaan enrollment, service receipt, or causal impact.",
        url="https://poshanabhiyaan.gov.in/",
    ),
    SchemeProxy(
        key="samagra_shiksha", name="Samagra Shiksha (education outcome proxy)",
        ministry="Ministry of Education", nfhs_indicator="female_school_pct",
        proxy_kind="Education-outcome proxy", proxy_measure_label="Females age 6+ who ever attended school",
        gap_label="Females age 6+ who never attended school (attainment proxy)", direction="coverage",
        planner_eligible=False,
        interpretation="NFHS ever-attended-school is an education-access/attainment outcome, not Samagra Shiksha enrollment, current attendance, or program impact.",
        url="https://samagra.education.gov.in/about.html",
    ),
]


def scheme_gap_frame(profiles: pd.DataFrame) -> pd.DataFrame:
    """Long-form service/outcome proxy table; unavailable measurements remain NaN."""
    required = {"state_raw", "district_raw"}
    if not required.issubset(profiles.columns):
        raise ValueError(f"profiles missing required fields: {sorted(required - set(profiles.columns))}")
    rows = []
    for _, row in profiles.iterrows():
        for scheme in SCHEME_PROXIES:
            measure = pd.to_numeric(pd.Series([row.get(scheme.nfhs_indicator, float("nan"))]), errors="coerce").iloc[0]
            if pd.isna(measure):
                gap = float("nan")
            elif scheme.direction == "burden":
                gap = float(measure)
            else:
                gap = max(0.0, 100.0 - float(measure))
            rows.append({
                "state_raw": row["state_raw"], "district_raw": row["district_raw"],
                "state_key": row.get("state_key", ""), "district_key": row.get("district_key", ""),
                "scheme_key": scheme.key, "scheme": scheme.name, "ministry": scheme.ministry,
                "proxy_kind": scheme.proxy_kind, "proxy_measure_pct": measure,
                "proxy_measure_label": scheme.proxy_measure_label,
                "coverage_proxy_pct": measure if scheme.proxy_kind == "Service-access proxy" else float("nan"),
                "outcome_proxy_pct": measure if scheme.proxy_kind != "Service-access proxy" else float("nan"),
                "gap_proxy_pct": gap, "gap_label": scheme.gap_label,
                "planner_eligible": scheme.planner_eligible,
                "interpretation": scheme.interpretation, "source_url": scheme.url,
                "survey_households": row.get("survey_households", float("nan")),
                "survey_period": "NFHS-5, 2019-21",
            })
    return pd.DataFrame(rows)


def plan_budget(gaps: pd.DataFrame, budget_inr: float) -> pd.DataFrame:
    """Allocate a scenario budget among service-access proxies in proportion to gap values.

    This is not a cost optimizer, beneficiary forecast, or impact estimate. Nutrition and
    education outcome proxies are excluded because they are not service-coverage denominators.
    """
    if budget_inr < 0:
        raise ValueError("budget_inr must be non-negative")
    out = gaps.copy()
    if "planner_eligible" in out:
        out = out[out["planner_eligible"].fillna(False).astype(bool)]
    if out.empty:
        out["budget_share_pct"] = pd.Series(dtype=float)
        out["allocation_inr"] = pd.Series(dtype=float)
        return out
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
