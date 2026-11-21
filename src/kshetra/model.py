"""Cross-check model: predict Census-2011 MPI headcount ratio from NFHS-5 indicators.

Why: the explainable index is hand-weighted. This model asks whether NFHS-5 indicators
carry enough signal to reproduce an external multidimensional-poverty measure. Where it
does, it can estimate MPI-like values for districts that lack an MPI record (e.g. districts
created after 2011). Those estimates are model outputs, not observations, and are labelled as such.

Model: supports ``xgboost.XGBRegressor`` (as named on Slide 7 of the presentation) when
``xgboost`` is installed in the runtime environment, with automatic fallback to
scikit-learn's ``HistGradientBoostingRegressor`` (gradient-boosted trees with native NaN
handling) when ``xgboost`` is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from .config import INDICATOR_LABELS, INDICATORS

try:
    from xgboost import XGBRegressor  # type: ignore[import-not-found]
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None  # type: ignore[assignment]
    HAS_XGBOOST = False


def available_backends() -> list[str]:
    """Return available gradient-boosting backends in this Python environment."""
    backends = ["hist_gb"]
    if HAS_XGBOOST:
        backends.insert(0, "xgboost")
    return backends


def resolve_backend(backend: str = "auto") -> str:
    """Resolve 'auto', 'xgboost', or 'hist_gb' to an active backend name."""
    b = backend.strip().lower()
    if b == "auto":
        return "xgboost" if HAS_XGBOOST else "hist_gb"
    if b == "xgboost":
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost is not installed in this environment; use backend='auto' or 'hist_gb'")
        return "xgboost"
    if b in {"hist_gb", "sklearn", "histgradientboosting"}:
        return "hist_gb"
    raise ValueError(f"Unknown model backend: {backend!r}. Expected 'auto', 'xgboost', or 'hist_gb'.")


def build_model(random_state: int = 42, backend: str = "auto") -> Any:
    """Build a gradient-boosted decision tree regressor with native missing-value support."""
    chosen = resolve_backend(backend)
    if chosen == "xgboost":
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=random_state,
    )


def evaluate(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42, backend: str = "auto", *, include_oof: bool = False) -> dict:
    """Cross-validated accuracy, validation against the composite index, and permutation importance."""
    chosen = resolve_backend(backend)
    features = list(INDICATORS)
    train = df[df["mpi_hcr_pct"].notna()].reset_index(drop=True)
    X, y = train[features].to_numpy(), train["mpi_hcr_pct"].to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y, dtype=float)
    importances = []
    for tr, te in kf.split(X):
        model = build_model(random_state=random_state, backend=chosen).fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
        perm = permutation_importance(model, X[te], y[te], n_repeats=5, random_state=random_state)
        importances.append(perm.importances_mean)

    imp = pd.DataFrame(importances, columns=features).mean().sort_values(ascending=False)
    rho, _ = spearmanr(train["composite_index"], y) if "composite_index" in train else (np.nan, None)
    model_label = "XGBoost (XGBRegressor)" if chosen == "xgboost" else "Scikit-Learn HistGradientBoostingRegressor"
    return {
        "model_backend": chosen,
        "model_name": model_label,
        "xgboost_available": HAS_XGBOOST,
        "n_districts_with_mpi": int(len(train)),
        "cv_folds": n_splits,
        "cv_r2": round(float(r2_score(y, preds)), 3),
        "cv_mae_pct_points": round(float(mean_absolute_error(y, preds)), 2),
        "spearman_composite_vs_mpi": round(float(rho), 3) if not np.isnan(rho) else None,
        "permutation_importance": {INDICATOR_LABELS[k]: round(float(v), 4) for k, v in imp.items()},
        **({"_oof_predictions": pd.Series(preds, index=df.index[df["mpi_hcr_pct"].notna()])} if include_oof else {}),
    }


def predict_proxy_mpi(df: pd.DataFrame, random_state: int = 42, backend: str = "auto") -> pd.Series:
    """Fit on all districts with an MPI value and predict for every district.

    Returns a series aligned to ``df``; values for districts that already have an MPI
    record are the fitted values and should not be read as independent predictions.
    """
    chosen = resolve_backend(backend)
    features = list(INDICATORS)
    train = df[df["mpi_hcr_pct"].notna()]
    model = build_model(random_state=random_state, backend=chosen).fit(
        train[features].to_numpy(), train["mpi_hcr_pct"].to_numpy()
    )
    return pd.Series(model.predict(df[features].to_numpy()), index=df.index)
