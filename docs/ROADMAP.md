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
| XGBoost baseline (PPT slide 7) | Implemented (`XGBRegressor` + fallback) | `src/kshetra/model.py` runs `xgboost.XGBRegressor` (5-fold CV on the bundled v8 data: R² = 0.845, MAE = 5.12 pp) when `xgboost` is installed, with automatic fallback to `HistGradientBoostingRegressor` (R² = 0.844). |
| K-Means district profiles | Implemented | Descriptive clustering on the 3 NFHS domain scores (`outputs/profile_clusters.csv`). |
| Satellite CV / STAC helper | Implemented as code, not validated on real imagery | `compute_image_change_features()` (ΔExG, ΔNGRDI, pixel change) runs on uploaded images. `scripts/fetch_satellite_stac.py` returns Sentinel-2 scene metadata only. `kshetra.zonal` and `scripts/stac_to_environment_csv.py` convert zonal means (computed externally) plus scene cloud cover into the Module 2 CSV. Clipping rasters to district polygons is an external GIS step; no raster reader is bundled. |
| Image-embedding change (CNN) | Optional dashboard section, not run in build environment | `kshetra.embeddings.embedding_difference()` accepts any extractor. `torchvision_extractor()` builds MobileNetV3-Small features; with no path it fetches official torchvision weights (network needed), otherwise it loads a local file with `weights_only=True`. The dashboard shows it only when a weights path is entered. |
| Drought transition baseline | Optional dashboard section (Module 2), no data bundled | `kshetra.forecast.drought_transition_baseline()` estimates P(deficit next month given current state) per district from consecutive months, with sample counts. This is a Markov baseline, not an LSTM or Transformer. It needs a monthly rainfall or VIIRS series that you supply. |
| Public MIS coverage comparison | Loader, diagnostic and optional dashboard section (Module 5); no data bundled | `kshetra.mis` validates a supplied MIS export (provenance, denominators, comparable schemes) and reports enrolment coverage against the NFHS-5 service proxy. Definitions and denominators differ, so the gap is a field-verification prompt, not a finding. |
| District boundary check | Checker implemented, no boundary file bundled | `scripts/check_district_geojson.py` reports size, feature count, exact-name coverage and unmatched features. No polygon file is shipped. |
| Field-verification loop | Local audit-log prototype implemented | Aggregate-only SQLite form; no login/permissions or cross-user deployment controls. Do not enter personal data. |

## Next priorities for production deployment

1. Confirm the MPI file's source, year and licensing, and replace name-based crosswalks with official Census/LGD district codes.
2. Select an authoritative administrative boundary GeoJSON matching the 707 NFHS-5 districts, check it with `scripts/check_district_geojson.py`, and bundle it under 3 MB.
3. Add a reproducible raster-clipping step (e.g. rasterio + district polygons) that writes the zonal-table input for `scripts/stac_to_environment_csv.py`, with CHIRPS, VIIRS, Sentinel-2 and FIRMS provenance preserved.
3b. Train or validate the drought baseline and any embedding model on real, documented time series before presenting skill metrics.
4. Integrate dated district-level MIS beneficiary denominators from PMUY, SBM-G, PM-JAY, JJM, POSHAN, and UDISE+.
5. Add authentication, role-based access, retention and review controls before multi-user field verification.
