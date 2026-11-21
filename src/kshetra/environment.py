"""Risk & Early Warning features for sourced district-level environmental observations.

This module does not manufacture satellite or weather measurements. It accepts a district
aggregate CSV produced from a documented source pipeline, validates provenance fields,
and computes a transparent alert score from available anomalies. It also provides
computer-vision change-detection metrics (RGB vegetation index ΔExG, pixel structural
change, and spatial anomaly map) for uploaded before/after satellite composites.
"""

from __future__ import annotations

from typing import Any

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


def compute_image_change_features(before_rgb: np.ndarray, after_rgb: np.ndarray) -> dict[str, Any]:
    """Extract quantitative change-detection features from aligned RGB image arrays.

    Computes:
    - Excess Green Index (ExG = 2G - R - B) before and after, and delta ExG (%)
    - Normalized Green-Red Difference Index (NGRDI = (G - R) / (G + R + eps))
    - Mean Absolute Pixel Intensity Change (%)
    - Changed Area Fraction (% of pixels exceeding 15% intensity or vegetation shift)
    - Difference heatmap array (H, W, 3) highlighting vegetation loss (red/orange) vs gain (teal)
    """
    def as_rgb(image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.ndim == 3 and image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, :3]  # Match Pillow convert('RGB'); alpha is not a cloud mask.
        return image

    before_rgb, after_rgb = as_rgb(before_rgb), as_rgb(after_rgb)
    if not before_rgb.size or before_rgb.shape != after_rgb.shape or before_rgb.ndim != 3 or before_rgb.shape[2] != 3:
        raise ValueError("before_rgb and after_rgb must be aligned HxWx3 RGB arrays")

    b_norm = before_rgb.astype(np.float32) / 255.0
    a_norm = after_rgb.astype(np.float32) / 255.0

    # Excess Green Index: 2*G - R - B in [-2, 2], normalized to [0, 1]
    exg_before = (2.0 * b_norm[:, :, 1] - b_norm[:, :, 0] - b_norm[:, :, 2] + 2.0) / 4.0
    exg_after = (2.0 * a_norm[:, :, 1] - a_norm[:, :, 0] - a_norm[:, :, 2] + 2.0) / 4.0
    delta_exg = exg_after - exg_before

    # NGRDI: (G - R) / (G + R + 1e-6)
    ngrdi_before = (b_norm[:, :, 1] - b_norm[:, :, 0]) / (b_norm[:, :, 1] + b_norm[:, :, 0] + 1e-6)
    ngrdi_after = (a_norm[:, :, 1] - a_norm[:, :, 0]) / (a_norm[:, :, 1] + a_norm[:, :, 0] + 1e-6)

    # Pixel-wise L1 intensity shift across channels
    pixel_diff = np.mean(np.abs(a_norm - b_norm), axis=2)
    changed_mask = (pixel_diff > 0.12) | (np.abs(delta_exg) > 0.08)

    # Build RGB change-detection overlay: loss in terracotta (#ad3f24), gain in teal (#4d8b73)
    h, w = delta_exg.shape
    heatmap = np.full((h, w, 3), 243, dtype=np.uint8)
    loss = np.clip(-delta_exg * 4.0, 0.0, 1.0)
    gain = np.clip(delta_exg * 4.0, 0.0, 1.0)
    # Blend toward terracotta (173, 63, 36) for vegetation loss and teal (77, 139, 115) for gain
    heatmap[:, :, 0] = np.clip(243 - loss * (243 - 173) - gain * (243 - 77), 0, 255).astype(np.uint8)
    heatmap[:, :, 1] = np.clip(239 - loss * (239 - 63) - gain * (239 - 139), 0, 255).astype(np.uint8)
    heatmap[:, :, 2] = np.clip(228 - loss * (228 - 36) - gain * (228 - 115), 0, 255).astype(np.uint8)

    return {
        "mean_exg_before": round(float(np.mean(exg_before) * 100), 2),
        "mean_exg_after": round(float(np.mean(exg_after) * 100), 2),
        "delta_exg_pct_points": round(float(np.mean(delta_exg) * 100), 2),
        "delta_ngrdi": round(float(np.mean(ngrdi_after - ngrdi_before)), 4),
        "mean_pixel_change_pct": round(float(np.mean(pixel_diff) * 100), 2),
        "changed_area_pct": round(float(np.mean(changed_mask) * 100), 2),
        "change_heatmap_rgb": heatmap,
    }
