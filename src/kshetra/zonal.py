"""Zonal-statistics arithmetic and the STAC-scene -> environmental-signal bridge.

The functions here operate on arrays and tables that have already been clipped to a
district. Clipping rasters to district polygons is an external GIS step (e.g. QGIS,
rasterstats); this module does not read rasters. It computes transparent anomaly values
from zonal means, applies the Sentinel-2 cloud-cover screen, and writes rows in the schema
accepted by ``kshetra.environment``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

ZONAL_REQUIRED = {
    "state", "district", "observed_at", "source", "scene_ids",
    "ndvi_current", "ndvi_baseline", "viirs_current", "viirs_baseline",
    "rain_observed_mm", "rain_normal_mm",
}
OUTPUT_COLUMNS = ["state", "district", "observed_at", "source", "rainfall_deficit_pct",
                  "ndvi_change_pct", "viirs_change_pct", "fire_count"]


def ndvi(nir: Any, red: Any) -> np.ndarray:
    """Normalized difference vegetation index (NIR - Red) / (NIR + Red); NaN where undefined."""
    nir_a = np.asarray(nir, dtype=float)
    red_a = np.asarray(red, dtype=float)
    num = nir_a - red_a
    den = nir_a + red_a
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=np.isfinite(den) & (den != 0))
    return out


def zonal_mean(values: Any, mask: Any, min_valid_fraction: float = 0.5) -> float:
    """Mean of finite pixels inside a boolean district mask.

    Returns NaN when fewer than ``min_valid_fraction`` of masked pixels are finite, so that
    heavily cloudy or missing zones are not summarized from a sliver of data.
    """
    v = np.asarray(values, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if v.shape != m.shape:
        raise ValueError("values and mask must have the same shape")
    sel = v[m]
    if sel.size == 0:
        raise ValueError("district mask selects no pixels")
    valid = np.isfinite(sel)
    if valid.mean() < min_valid_fraction:
        return float("nan")
    return float(sel[valid].mean())


def pct_change(current: float, baseline: float) -> float:
    """Percent change versus baseline; negative means decline. NaN if baseline is zero or missing."""
    if not (np.isfinite(current) and np.isfinite(baseline)) or baseline == 0:
        return float("nan")
    return float(100.0 * (current - baseline) / abs(baseline))


def rainfall_deficit_pct(observed_mm: float, normal_mm: float) -> float:
    """Percent shortfall against a seasonal normal; positive means below normal."""
    if not (np.isfinite(observed_mm) and np.isfinite(normal_mm)) or normal_mm <= 0:
        return float("nan")
    return float(100.0 * (normal_mm - observed_mm) / normal_mm)


def scene_cloud_cover(scene_ids: str, scenes_by_id: Mapping[str, float | None]) -> float | None:
    """Maximum cloud cover across the listed scenes; None if any scene is unknown or lacks a value."""
    ids = [s.strip() for s in str(scene_ids).split(";") if s.strip()]
    if not ids:
        return None
    values = []
    for scene_id in ids:
        if scene_id not in scenes_by_id:
            return None
        cover = scenes_by_id[scene_id]
        if cover is None or not np.isfinite(float(cover)):
            return None
        values.append(float(cover))
    return max(values)


def zonal_table_to_signals(table: pd.DataFrame, scenes_by_id: Mapping[str, float | None],
                           max_cloud_cover: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert per-district zonal means into the environmental-signal schema.

    Returns ``(signals, rejected)``. Sentinel-2 NDVI change is blanked (not imputed) when a
    scene is unknown, lacks cloud cover, or exceeds ``max_cloud_cover``. Rows with no usable
    signal are moved to ``rejected`` with a reason.
    """
    missing = ZONAL_REQUIRED - set(table.columns)
    if missing:
        raise ValueError(f"zonal table missing columns: {', '.join(sorted(missing))}")
    if not 0 <= max_cloud_cover <= 100:
        raise ValueError("max_cloud_cover must be between 0 and 100")

    rows, rejected = [], []
    for r in table.itertuples(index=False):
        ids = [s.strip() for s in str(r.scene_ids).split(";") if s.strip()]
        cloud = scene_cloud_cover(r.scene_ids, scenes_by_id)
        if cloud is None or cloud > max_cloud_cover:
            ndvi_change = float("nan")
            cloud_note = "S2 NDVI blanked: scene unknown, cloud cover missing or above threshold"
        else:
            ndvi_change = pct_change(float(r.ndvi_current), float(r.ndvi_baseline))
            cloud_note = f"max S2 cloud cover {cloud:.0f}%"
        signals = {
            "rainfall_deficit_pct": rainfall_deficit_pct(float(r.rain_observed_mm), float(r.rain_normal_mm)),
            "ndvi_change_pct": ndvi_change,
            "viirs_change_pct": pct_change(float(r.viirs_current), float(r.viirs_baseline)),
            "fire_count": float(getattr(r, "fire_count")) if hasattr(r, "fire_count") and pd.notna(getattr(r, "fire_count")) else float("nan"),
        }
        record = {
            "state": r.state, "district": r.district, "observed_at": r.observed_at,
            "source": f"{r.source} | S2 scenes: {'; '.join(ids) if ids else 'none'} ({cloud_note})",
            **signals,
        }
        if all(np.isnan(signals[k]) for k in ("rainfall_deficit_pct", "ndvi_change_pct", "viirs_change_pct")):
            rejected.append({**record, "reason": "no usable numeric anomaly"})
        else:
            rows.append(record)

    signals_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    rejected_df = pd.DataFrame(rejected)
    return signals_df, rejected_df
