"""Risk & Early Warning features for sourced district-level environmental observations.

This module does not manufacture satellite or weather measurements. It accepts a district
aggregate CSV produced from a documented source pipeline, validates provenance fields,
and computes a transparent alert score from available anomalies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .text import norm_name, norm_state

REQUIRED_COLUMNS = {"state", "district", "observed_at", "source"}
SIGNAL_COLUMNS = {
    "rainfall_deficit_pct": "Rainfall below seasonal normal (%)",
    "ndvi_change_pct": "NDVI change versus seasonal baseline (%)",
    "viirs_change_pct": "VIIRS night-light change versus baseline (%)",
    "fire_count": "NASA FIRMS active-fire count",
}


def validate_environmental_signals(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize uploaded observations. Raises ValueError with actionable details."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if not any(c in data.columns for c in SIGNAL_COLUMNS):
        raise ValueError(f"At least one signal column is required: {', '.join(SIGNAL_COLUMNS)}")

    out = data.copy()
    out["state"] = out["state"].astype(str).str.strip()
    out["district"] = out["district"].astype(str).str.strip()
    out["source"] = out["source"].astype(str).str.strip()
    out["observed_at"] = pd.to_datetime(out["observed_at"], errors="coerce", utc=True)
    if out[["state", "district", "source"]].eq("").any().any():
        raise ValueError("state, district, and source must not be blank")
    if out["observed_at"].isna().any():
        raise ValueError("observed_at must contain valid dates/timestamps")
    for col in SIGNAL_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    available = [c for c in SIGNAL_COLUMNS if c in out.columns]
    if out[available].notna().sum(axis=1).eq(0).any():
        raise ValueError("Each row must contain at least one numeric environmental signal")
    if "fire_count" in out.columns and (out["fire_count"].dropna() < 0).any():
        raise ValueError("fire_count must be non-negative")
    out["state_key"] = out["state"].map(norm_state)
    out["district_key"] = out["district"].map(norm_name)
    if out.duplicated(["state_key", "district_key", "observed_at"]).any():
        raise ValueError("Duplicate state/district/observed_at rows; aggregate to one row per district and date")
    return out


def score_environmental_risk(data: pd.DataFrame) -> pd.DataFrame:
    """Add a transparent rule-based signal score and alert level to validated observations.

    Rainfall deficit is scaled against 40%, vegetation and light declines against 30%.
    Equal weighting is used across available numeric anomaly measures. FIRMS count is shown
    as a separate shock signal and does not enter the score. Thresholds are screening rules,
    not trained probabilities or poverty classifications.
    """
    out = validate_environmental_signals(data)
    component_cols = []
    if "rainfall_deficit_pct" in out:
        out["score_rainfall"] = (pd.to_numeric(out["rainfall_deficit_pct"], errors="coerce") / 40 * 100).clip(0, 100)
        component_cols.append("score_rainfall")
    if "ndvi_change_pct" in out:
        out["score_vegetation"] = (-pd.to_numeric(out["ndvi_change_pct"], errors="coerce") / 30 * 100).clip(0, 100)
        component_cols.append("score_vegetation")
    if "viirs_change_pct" in out:
        out["score_night_lights"] = (-pd.to_numeric(out["viirs_change_pct"], errors="coerce") / 30 * 100).clip(0, 100)
        component_cols.append("score_night_lights")
    out["environmental_score"] = out[component_cols].mean(axis=1) if component_cols else np.nan
    out["environmental_signals_used"] = out[component_cols].notna().sum(axis=1) if component_cols else 0
    out["alert_level"] = np.select(
        [out["environmental_score"].isna(), out["environmental_score"] >= 66, out["environmental_score"] >= 33],
        ["No anomaly baseline", "High signal", "Moderate signal"], default="Watch",
    )
    out["fire_shock"] = out.get("fire_count", pd.Series(0, index=out.index)).fillna(0).gt(0)
    out["requires_field_verification"] = out["environmental_score"].ge(33) | out["fire_shock"]
    return out.sort_values(["observed_at", "environmental_score"], ascending=[False, False]).reset_index(drop=True)


def attach_profile_scores(signals: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Attach the deprivation index to dated environment observations by exact normalized name."""
    scored = score_environmental_risk(signals)
    p = profiles.copy()
    p["state_key"] = p["state_raw"].map(norm_state)
    p["district_key"] = p["district_raw"].map(norm_name)
    cols = [c for c in ["state_key", "district_key", "composite_percentile", "risk_tier", "confidence"] if c in p]
    out = scored.merge(p[cols].drop_duplicates(["state_key", "district_key"]), on=["state_key", "district_key"], how="left")
    out["profile_match"] = out["composite_percentile"].notna()
    return out
