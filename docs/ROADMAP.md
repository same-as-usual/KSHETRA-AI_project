# Roadmap

Mapped from the KSHETRA-AI panel presentation (next steps, slide 10).

| Stage | Status | Notes |
|---|---|---|
| Collect and clean district poverty data (NFHS-5, Census 2011) | Done (this commit) | See `docs/DATA_QUALITY.md` |
| Explainable district deprivation index with driver breakdown | Done | `src/kshetra/index.py` |
| District atlas and profile dashboard | Prototype | `app/streamlit_app.py`; the map view is not yet built |
| Tree-based baseline model (XGBoost in the plan) | Prototype | scikit-learn `HistGradientBoostingRegressor` used until xgboost is installed |
| K-Means profile clusters | Done | `profile_clusters.csv` |
| CHIRPS rainfall, VIIRS night lights, Sentinel-2 change detection | Not started | Needs internet access and a data-download step. Output should be district-level, with change over time, not a poverty measure. |
| Livelihood-risk early-warning alerts | Not started | Depends on satellite features |
| Intervention planner and budget allocation | Not started | |
| SchemeGap intelligence | Not started | Needs scheme-coverage data |
| Multilingual RAG assistant | Not started | Answers must cite the pipeline's data |
| Field verification workflow | Not started | Required before any intervention decision |
