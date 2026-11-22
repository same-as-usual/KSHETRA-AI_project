"""Transparent temporal baseline for drought-like rainfall anomalies (a Markov transition model).

Method, per district:

1. Compute each calendar month's climatology from the district's own history.
2. Express each month as a percent anomaly from that climatology.
3. Mark a month as a deficit when its anomaly is at or below `-deficit_pct`.
4. Estimate P(deficit next month | deficit this month) and P(deficit next | no deficit this month)
 from consecutive months, then forecast from the most recent observed state.

This is a baseline, not an LSTM or Transformer, and it is not validated for operational use. It
requires at least `min_months` consecutive observations and reports sample counts.
"""

from **future** import annotations

import numpy as np
import pandas as pd

def \_to\_month(series: pd.Series) -&gt; pd.PeriodIndex:
    return pd.PeriodIndex(pd.to\_datetime(series).dt.to\_period("M"))

def monthly\_anomalies(data: pd.DataFrame, district\_col: str = "district\_key",
                      month\_col: str = "month", value\_col: str = "value") -&gt; pd.DataFrame:
    """Add a percent anomaly relative to each district's calendar-month climatology."""
    required = {district\_col, month\_col, value\_col}
    if not required.issubset(data.columns):
        raise ValueError(f"data must contain {sorted(required)}")
    out = data.copy()
    out\["\_period"\] = \_to\_month(out\[month\_col\])
    out\[value\_col\] = pd.to\_numeric(out\[value\_col\], errors="coerce")
    out\["\_cal\_month"\] = out\["\_period"\].dt.month
    clim = out.groupby(\[district\_col, "\_cal\_month"\])\[value\_col\].transform("mean")
    obs = out\[value\_col\]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 \* (obs - clim) / clim
    # Zero climatology and zero observation is a normal dry month: 0% anomaly, not a missing value.
    # Mean climatology cannot be zero when any non-negative observation in that group is positive.
    pct = np.where((clim == 0) &amp; (obs == 0), 0.0, pct)
    out\["anomaly\_pct"\] = np.where(clim &gt;= 0, pct, np.nan)
    return out.sort\_values(\[district\_col, "\_period"\]).reset\_index(drop=True)

def drought\_transition\_baseline(data: pd.DataFrame, deficit\_pct: float = 25.0, min\_months: int = 24,
                                district\_col: str = "district\_key", month\_col: str = "month",
                                value\_col: str = "value") -&gt; pd.DataFrame:
    """One row per district: transition probabilities, latest state and forecast for next month.

```
``forecast_deficit_prob`` is NaN when the district lacks history or has no observed
transitions from the latest state. It should be read with ``n_transitions_from_state``.
"""
if deficit_pct <= 0:
    raise ValueError("deficit_pct must be positive")
anomalies = monthly_anomalies(data, district_col, month_col, value_col)
rows = []
for district, group in anomalies.groupby(district_col, sort=True):
    g = group.dropna(subset=["anomaly_pct"]).sort_values("_period")
    months = list(g["_period"])
    deficit = (g["anomaly_pct"].to_numpy() <= -deficit_pct)
    record = {district_col: district, "months_observed": len(g), "sufficient_history": len(g) >= min_months,
              "latest_month": str(months[-1]) if months else None,
              "latest_state_deficit": bool(deficit[-1]) if len(deficit) else None}
    if len(g) < min_months:
        record.update({"p_deficit_given_deficit": np.nan, "p_deficit_given_normal": np.nan,
                       "forecast_deficit_prob": np.nan, "n_transitions_from_state": 0})
        rows.append(record)
        continue
    # Only transitions between adjacent calendar months are counted.
    prev_deficit, next_deficit = [], []
    for i in range(len(months) - 1):
        if (months[i + 1] - months[i]).n == 1:
            prev_deficit.append(deficit[i])
            next_deficit.append(deficit[i + 1])
    prev_arr, next_arr = np.array(prev_deficit, dtype=bool), np.array(next_deficit, dtype=bool)
    n_d, n_n = int(prev_arr.sum()), int((~prev_arr).sum())
    p_dd = float(next_arr[prev_arr].mean()) if n_d else np.nan
    p_dn = float(next_arr[~prev_arr].mean()) if n_n else np.nan
    latest = bool(deficit[-1])
    forecast = p_dd if latest else p_dn
    record.update({
        "p_deficit_given_deficit": p_dd, "p_deficit_given_normal": p_dn,
        "forecast_deficit_prob": forecast,
        "n_transitions_from_state": n_d if latest else n_n,
    })
    rows.append(record)
return pd.DataFrame(rows)
```