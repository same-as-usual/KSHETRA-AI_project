"""Command-line entry point: build the district dataset, index, model and outputs.

Usage (from the repository root, after placing the raw CSVs in data/raw/):

    python -m kshetra.pipeline
"""

from __future__ import annotations

import argparse
import json
import warnings

import pandas as pd

from .config import OUTPUT_DIR, PROCESSED_DIR
from .crosswalk import build_district_frame
from .index import cluster_centers, compute_index
from .model import evaluate, predict_proxy_mpi


def run(backend: str = "auto") -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        base = build_district_frame()
    warning_messages = [str(w.message) for w in caught]

    profiled = compute_index(base)
    metrics = evaluate(profiled, backend=backend)
    profiled["proxy_mpi_hcr_pct"] = predict_proxy_mpi(profiled, backend=backend).round(2)

    base.to_csv(PROCESSED_DIR / "district_base.csv", index=False)
    profiled.to_csv(OUTPUT_DIR / "district_profiles.csv", index=False, float_format="%.4f")
    cluster_centers(profiled).to_csv(OUTPUT_DIR / "profile_clusters.csv")

    summary = {
        "districts": int(len(profiled)),
        "mpi_matched": int((profiled["mpi_match"] != "none").sum()),
        "mpi_exact": int((profiled["mpi_match"] == "exact").sum()),
        "mpi_fuzzy": int((profiled["mpi_match"] == "fuzzy").sum()),
        "census_scst_published": int(profiled["scst_published"].sum()),
        "risk_tier_counts": profiled["risk_tier"].value_counts().to_dict(),
        "confidence_counts": profiled["confidence"].value_counts().to_dict(),
        "warnings": warning_messages,
    }
    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "model": metrics}, fh, indent=2, ensure_ascii=False, default=str)
    return {"summary": summary, "model": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "xgboost", "hist_gb"), default="auto")
    result = run(backend=parser.parse_args().backend)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str))
    print("Model:", {k: v for k, v in result["model"].items() if k != "permutation_importance"})
