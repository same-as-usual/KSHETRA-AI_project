"""SchemeGap and action-planning helpers.

The NFHS-5 fields below are broad service/outcome proxies, not scheme enrollment,
beneficiary counts, or eligibility. Only the four service-access proxies are used by
the budget-share scenario; nutrition and education outcome indicators are shown for
context but are not mixed into its allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import DOMAIN_LABELS, INDICATORS
from .index import _tier


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
    illustrative_unit_cost_inr: float | None
    interpretation: str
    url: str


SCHEME_PROXIES = [
    SchemeProxy(
        key="pmuy", name="Pradhan Mantri Ujjwala Yojana (PMUY)",
        ministry="Ministry of Petroleum and Natural Gas", nfhs_indicator="clean_cooking_fuel_pct",
        proxy_kind="Service-access proxy", proxy_measure_label="Households using clean cooking fuel",
        gap_label="Households not using clean cooking fuel", direction="coverage",
        planner_eligible=True, illustrative_unit_cost_inr=2200.0,
        interpretation="NFHS clean-fuel use is a service-access proxy; it does not identify PMUY beneficiaries or LPG connection ownership.",
        url="https://www.pmuy.gov.in/",
    ),
    SchemeProxy(
        key="sbm_g", name="Swachh Bharat Mission (Gramin) (SBM-G)",
        ministry="Department of Drinking Water and Sanitation", nfhs_indicator="improved_sanitation_pct",
        proxy_kind="Service-access proxy", proxy_measure_label="Population using an improved sanitation facility",
        gap_label="Population without an improved sanitation facility", direction="coverage",
        planner_eligible=True, illustrative_unit_cost_inr=12000.0,
        interpretation="NFHS sanitation coverage is an outcome proxy; it is not SBM-G coverage or eligibility. This indicator is not rural-only.",
        url="https://swachhbharatmission.ddws.gov.in/",
    ),
    SchemeProxy(
        key="pmjay_proxy", name="Ayushman Bharat PM-JAY (coverage proxy only)",
        ministry="National Health Authority", nfhs_indicator="health_insurance_pct",
        proxy_kind="Service-access proxy", proxy_measure_label="Households with any health-insurance / financing cover",
        gap_label="Households without any reported health-insurance / financing cover", direction="coverage",
        planner_eligible=True, illustrative_unit_cost_inr=1500.0,
        interpretation="NFHS reports any household member with health insurance/financing; it does not isolate PM-JAY beneficiaries or verify eligibility.",
        url="https://pmjay.gov.in/",
    ),
    SchemeProxy(
        key="jal_jeevan", name="Jal Jeevan Mission (service proxy only)",
        ministry="Ministry of Jal Shakti", nfhs_indicator="improved_water_pct",
        proxy_kind="Service-access proxy", proxy_measure_label="Population using an improved drinking-water source",
        gap_label="Population without an improved drinking-water source", direction="coverage",
        planner_eligible=True, illustrative_unit_cost_inr=18000.0,
        interpretation="NFHS improved-water access is not a functional household tap connection and does not measure Jal Jeevan Mission coverage.",
        url="https://jaljeevanmission.gov.in/",
    ),
    SchemeProxy(
        key="poshan", name="POSHAN Abhiyaan (nutrition outcome proxy)",
        ministry="Ministry of Women and Child Development", nfhs_indicator="stunted_pct",
        proxy_kind="Nutrition-outcome proxy", proxy_measure_label="Children under 5 with stunting",
        gap_label="Child stunting prevalence (nutrition-burden proxy)", direction="burden",
        planner_eligible=False, illustrative_unit_cost_inr=None,
        interpretation="NFHS stunting is a nutrition outcome, not POSHAN Abhiyaan enrollment, service receipt, or causal impact.",
        url="https://poshanabhiyaan.gov.in/",
    ),
    SchemeProxy(
        key="samagra_shiksha", name="Samagra Shiksha (education outcome proxy)",
        ministry="Ministry of Education", nfhs_indicator="female_school_pct",
        proxy_kind="Education-outcome proxy", proxy_measure_label="Females age 6+ who ever attended school",
        gap_label="Females age 6+ who never attended school (attainment proxy)", direction="coverage",
        planner_eligible=False, illustrative_unit_cost_inr=None,
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
                "nfhs_indicator": scheme.nfhs_indicator,
                "proxy_kind": scheme.proxy_kind, "proxy_measure_pct": measure,
                "proxy_measure_label": scheme.proxy_measure_label,
                "coverage_proxy_pct": measure if scheme.proxy_kind == "Service-access proxy" else float("nan"),
                "outcome_proxy_pct": measure if scheme.proxy_kind != "Service-access proxy" else float("nan"),
                "gap_proxy_pct": gap, "gap_label": scheme.gap_label,
                "planner_eligible": scheme.planner_eligible,
                "illustrative_unit_cost_inr": scheme.illustrative_unit_cost_inr,
                "interpretation": scheme.interpretation, "source_url": scheme.url,
                "survey_households": row.get("survey_households", float("nan")),
                "census_households": row.get("census_households", float("nan")),
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

    # Illustrative unit-cost reach calculation when unit cost is present
    if "illustrative_unit_cost_inr" in out.columns:
        unit_cost = pd.to_numeric(out["illustrative_unit_cost_inr"], errors="coerce")
        out["illustrative_units_funded"] = np.where(unit_cost > 0, np.floor(out["allocation_inr"] / unit_cost), np.nan)
    return out.sort_values("allocation_inr", ascending=False).reset_index(drop=True)


def simulate_counterfactual_profile(
    district_row: pd.Series,
    all_profiles: pd.DataFrame,
    gap_closure_pct: float,
) -> dict[str, Any]:
    """Recompute a district's composite deprivation index and percentile under a target service-gap closure.

    Closes ``gap_closure_pct`` (0-100%) of the uncovered gap for the living-standards
    service indicators included in the composite index (`clean_cooking_fuel_pct`,
    `improved_sanitation_pct`, `improved_water_pct`), re-standardizes against the
    national distribution in ``all_profiles``, and reports the simulated composite index,
    percentile rank, and screening tier.
    """
    closure = float(np.clip(gap_closure_pct, 0.0, 100.0)) / 100.0
    domains = sorted({d for d, _, _ in INDICATORS.values()}, key=list(DOMAIN_LABELS).index)
    n_domain_indicators = {d: sum(1 for dd, _, _ in INDICATORS.values() if dd == d) for d in domains}

    sim_indicators = {"clean_cooking_fuel_pct", "improved_sanitation_pct", "improved_water_pct"}
    new_contrib: dict[str, float] = {}
    new_z: dict[str, float] = {}

    for name, (domain, direction, _) in INDICATORS.items():
        col = pd.to_numeric(all_profiles[name], errors="coerce")
        med = float(col.median())
        filled = col.fillna(med)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0)) or 1.0

        raw_val = pd.to_numeric(pd.Series([district_row.get(name, np.nan)]), errors="coerce").iloc[0]
        val = med if pd.isna(raw_val) else float(raw_val)
        if name in sim_indicators:
            val = min(100.0, val + max(0.0, 100.0 - val) * closure)

        z = (val - mean) / std
        new_z[name] = direction * z
        weight = 1.0 / (len(domains) * n_domain_indicators[domain])
        new_contrib[name] = weight * direction * z

    domain_scores = {}
    for domain in domains:
        members = [new_z[n] for n, (d, _, _) in INDICATORS.items() if d == domain]
        domain_scores[domain] = float(np.mean(members))

    new_composite = float(np.mean(list(domain_scores.values())))
    orig_composite = float(district_row.get("composite_index", new_composite))
    orig_percentile = float(district_row.get("composite_percentile", 50.0))

    # Percentile rank against the baseline national distribution
    other_composites = pd.to_numeric(all_profiles["composite_index"], errors="coerce").dropna().to_numpy()
    if len(other_composites) > 0:
        new_percentile = float(np.mean(other_composites <= new_composite) * 100.0)
    else:
        new_percentile = orig_percentile

    return {
        "gap_closure_pct": round(closure * 100.0, 1),
        "baseline_composite": round(orig_composite, 4),
        "simulated_composite": round(new_composite, 4),
        "composite_delta": round(new_composite - orig_composite, 4),
        "baseline_percentile": round(orig_percentile, 1),
        "simulated_percentile": round(new_percentile, 1),
        "percentile_delta": round(new_percentile - orig_percentile, 1),
        "baseline_tier": str(district_row.get("risk_tier", _tier(orig_percentile))),
        "simulated_tier": _tier(new_percentile),
        "simulated_living_standards_score": round(domain_scores.get("living_standards", 0.0), 3),
    }
