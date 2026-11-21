"""Explainable deprivation index, driver decomposition, confidence and district profiles.

Method (deliberately simple so every number can be traced back to a formula):
1. Each NFHS-5 indicator is imputed with the national median when missing and z-scored across districts.
   The z-score is multiplied by the indicator's direction so that a positive value always means
   "more deprived than the average district".
2. Indicators are averaged within each domain (living standards, education, health & nutrition).
3. The composite index is the mean of the domain scores (equal domain weights). Because of this,
   the composite is exactly the sum of per-indicator contributions, which powers the driver bars.
4. The composite is also expressed as a percentile rank (0-100) among districts.
5. Confidence reflects data completeness and sample size, not the model's accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import DOMAIN_LABELS, INDICATOR_LABELS, INDICATORS

TIERS = [(75, "Very high"), (50, "High"), (25, "Moderate"), (0, "Lower")]


def _tier(percentile: float) -> str:
    for cut, label in TIERS:
        if percentile >= cut:
            return label
    return "Lower"


def compute_index(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """Add index columns to a district frame produced by ``crosswalk.build_district_frame``."""
    out = df.copy()
    names = list(INDICATORS)
    domains = sorted({d for d, _, _ in INDICATORS.values()}, key=list(DOMAIN_LABELS).index)

    # Missing-data counts are computed before imputation so they can drive the confidence flag.
    out["indicators_missing"] = out[names].isna().sum(axis=1)

    z_cols = {}
    contrib_cols = {}
    n_domain_indicators = {d: sum(1 for dd, _, _ in INDICATORS.values() if dd == d) for d in domains}
    for name in names:
        domain, direction, _ = INDICATORS[name]
        col = out[name]
        filled = col.fillna(col.median())
        std = filled.std(ddof=0)
        z = (filled - filled.mean()) / (std if std else 1.0)
        z_cols[f"z_{name}"] = direction * z
        weight = 1.0 / (len(domains) * n_domain_indicators[domain])
        contrib_cols[f"contrib_{name}"] = weight * direction * z
    out = pd.concat([out, pd.DataFrame(z_cols, index=out.index), pd.DataFrame(contrib_cols, index=out.index)], axis=1)

    for domain in domains:
        members = [f"z_{n}" for n, (d, _, _) in INDICATORS.items() if d == domain]
        out[f"score_{domain}"] = out[members].mean(axis=1)

    out["composite_index"] = out[[f"score_{d}" for d in domains]].mean(axis=1)
    out["composite_percentile"] = out["composite_index"].rank(pct=True) * 100
    out["risk_tier"] = out["composite_percentile"].map(_tier)

    # Primary driver: domain with the highest score, and the top 3 indicators by contribution.
    domain_scores = out[[f"score_{d}" for d in domains]].copy()
    domain_scores.columns = domains
    out["primary_domain"] = domain_scores.idxmax(axis=1).map(DOMAIN_LABELS)
    contrib = out[[f"contrib_{n}" for n in names]].copy()
    contrib.columns = names
    top = contrib.apply(lambda row: [n for n in row.sort_values(ascending=False).index[:3] if row[n] > 0], axis=1)
    out["top_drivers"] = top.map(lambda ns: "; ".join(INDICATOR_LABELS[n] for n in ns))

    # Confidence: completeness of the NFHS indicators and survey sample size.
    sample = out["survey_households"].fillna(0)
    conf = np.where((out["indicators_missing"] == 0) & (sample >= 500), "High",
                    np.where((out["indicators_missing"] <= 2) & (sample >= 200), "Medium", "Low"))
    out["confidence"] = conf

    # Profile clusters (K-Means on the domain scores) as a descriptive complement to the tiers.
    scaler = StandardScaler()
    X = scaler.fit_transform(domain_scores.values)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(X)
    out["profile_cluster"] = km.labels_
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=domains)
    order = centers.mean(axis=1).rank().astype(int).sub(1).to_dict()  # 0 = least deprived
    out["profile_cluster"] = out["profile_cluster"].map(order)
    return out


def cluster_centers(profiled: pd.DataFrame) -> pd.DataFrame:
    """Mean domain scores per profile cluster, for the dashboard and the README."""
    domains = list(DOMAIN_LABELS)
    cols = [f"score_{d}" for d in domains]
    table = profiled.groupby("profile_cluster")[cols].mean().round(2)
    table.columns = [DOMAIN_LABELS[d] for d in domains]
    table["districts"] = profiled.groupby("profile_cluster").size()
    return table
