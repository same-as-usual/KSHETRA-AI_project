# Roadmap and implementation status

Aligned to the six modules in the KSHETRA-AI panel presentation. “Implemented” means a working, tested code path exists in the repository.

| Presentation module / stage | Status | What works / what remains |
|---|---|---|
| 1. Poverty Intelligence Atlas | Implemented (default spatial bubble map + GeoJSON choropleth) | Displays a state-anchored spatial bubble map for all 707 districts out of the box, and upgrades to a polygon choropleth when a GeoJSON FeatureCollection is uploaded or placed at `data/geo/india_districts.geojson`. |
| 2. Risk & Early Warning | Implemented (anomaly engine + Risk Radar + CV image change + STAC script) | Validates sourced dated district aggregates, plots a 3-axis polar Risk Radar, computes before/after satellite image CV metrics (`ΔExG`, `ΔNGRDI`, pixel change, heatmap), and includes `scripts/fetch_satellite_stac.py` for Planetary Computer STAC queries. |
| 3. Explainable AI Insights | Implemented | Indicator contributions, domain scores, percentile, indicator-completeness gauge, external MPI cross-check, and local human-verification SQLite audit log. |
| 4. Action Planner | Implemented (budget splitter + counterfactual index recalculation) | Allocates a user-entered budget across four service-access proxies, computes illustrative unit-cost reach, and recalculates the district's counterfactual composite index, national percentile, and screening tier shift. |
| 5. SchemeGap Intelligence | Six-proxy view + workflow cards implemented | Maps 6 NFHS-5 indicators (clean fuel, sanitation, health insurance, improved water, child stunting, female school attainment) to programme areas with 4-step verification cards. Administrative MIS enrollment data is still required for actual beneficiary gaps. |
| 6. PRAGATI AI Assistant | Implemented (TF-IDF vector RAG + contextual map/cards + optional LLM) | Uses character/word n-gram TF-IDF embeddings and cosine similarity over 707 district profiles and 6 scheme proxies, paired with a contextual peer locator map, KPI cards, offline English/Hindi templates, and optional Gemini/OpenAI synthesis. |
| XGBoost baseline (PPT slide 7) | Implemented (`XGBRegressor` + fallback) | `src/kshetra/model.py` runs `xgboost.XGBRegressor` (5-fold CV R² = 0.849, MAE = 5.02 pp) when `xgboost` is installed, with automatic fallback to `HistGradientBoostingRegressor` (R² = 0.844). |
| K-Means district profiles | Implemented | Descriptive clustering on the 3 NFHS domain scores (`outputs/profile_clusters.csv`). |
| Satellite CV / STAC helper | Implemented (spectral/pixel indices + STAC query) | `compute_image_change_features()` extracts interpretable spectral/pixel shifts on uploaded imagery; `scripts/fetch_satellite_stac.py` queries Sentinel-2 L2A scenes via Planetary Computer STAC. End-to-end CNN/ViT/LSTM training requires labeled historical time series. |
| Field-verification loop | Local audit-log prototype implemented | Aggregate-only SQLite form; no login/permissions or cross-user deployment controls. Do not enter personal data. |

## Next priorities for production deployment

1. Confirm the MPI file's source, year and licensing, and replace name-based crosswalks with official Census/LGD district codes.
2. Select an authoritative administrative boundary GeoJSON matching the 707 NFHS-5 districts.
3. Schedule automated zonal raster aggregation from CHIRPS, VIIRS Black Marble, Sentinel-2, and NASA FIRMS over verified district polygons.
4. Integrate dated district-level MIS beneficiary denominators from PMUY, SBM-G, PM-JAY, JJM, POSHAN, and UDISE+.
5. Add authentication, role-based access, retention and review controls before multi-user field verification.
