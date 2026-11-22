# KSHETRA-AI

**From deprivation data to localized action.** KSHETRA-AI is a district-level deprivation screening and intervention-planning platform for India, aligned to the six modules in the project presentation.

> A score is a screening signal for review. It is not proof of poverty, a household eligibility determination, or a recommendation about an individual.

## Six modules

| Module | What is implemented | Data / limits |
|---|---|---|
| **1. Poverty Intelligence Atlas** | Interactive district choropleth (with GeoJSON) and default state-anchored spatial bubble map for all 707 districts, plus state summaries and ranked table. | Polygon choropleth requires a GeoJSON boundary file (exact name matches only). When no GeoJSON is uploaded, the map displays deterministic state-anchored markers for all 36 States/UTs. |
| **2. Risk & Early Warning** | Import validation, rule-based anomaly screening, risk radar, dated trend display, before/after satellite image computer-vision change metrics (ΔExG, ΔNGRDI, pixel diff, change heatmap), and STAC scene metadata query script (`scripts/fetch_satellite_stac.py`). | No environmental data is bundled by default. Upload attributed observations from CHIRPS, VIIRS, Sentinel-2 and/or FIRMS; no alerts appear without uploaded data. |
| **3. Explainable AI Insights** | Composite percentile, risk tier, indicator contributions, domain scores, indicator-completeness gauge, MPI cross-check, aggregate human review form. | Completeness gauge describes missingness/sample size, not calibrated model certainty. Local SQLite log; no user authentication or multi-user controls. |
| **4. Action Planner** | Hypothetical budget split across four service-access proxies, illustrative unit-cost reach benchmarks, and counterfactual composite index / percentile / tier recalculation. | Scenario simulation only: not a causal impact forecast or entitlement decision. Official scheme unit costs and administrative denominators must be verified locally. |
| **5. SchemeGap Intelligence** | Six NFHS-5 service/outcome proxies (clean fuel, sanitation, health-insurance, improved water, child stunting and female school attainment) with 4-step verification journey cards and official scheme links. | These are broad service/outcome proxies, not scheme enrollment or eligibility. Only four service-access proxies enter the budget-share scenario; nutrition/education outcomes are context only. |
| **6. PRAGATI AI Assistant** | TF-IDF vector RAG retrieval (cosine similarity) over 707 district profiles & 6 scheme proxies, contextual peer locator map, district KPI cards, English/Hindi synthesis templates, and optional Gemini / OpenAI API synthesis. | Runs 100% offline by default using deterministic templates; optional LLM synthesis is strictly constrained to retrieved evidence chunks. |

## Optional workflows (outside the dashboard)

- `scripts/stac_to_environment_csv.py`: converts externally computed district zonal means plus STAC scene cloud cover into the Module 2 CSV. Rows with unusable signals are reported separately. Raster clipping is not included.
- `scripts/check_district_geojson.py`: validates a boundary file's size and name coverage before it is bundled.
- `kshetra.embeddings`: image-embedding change distance. Requires PyTorch and a weights file you supply; not executed in the build environment.
- `kshetra.forecast`: per-district Markov drought-transition baseline with sample counts; requires a monthly series you supply.
- `kshetra.mis`: validates an administrative MIS export and compares enrolment coverage with NFHS-5 proxies.

These components have synthetic-data tests only. They are not yet used by the dashboard and have no validated skill metrics.

## Verified data currently in the project

The repository tracks the aggregate CSV files under `data/raw/` and derived profiles under `outputs/` so a fresh clone runs out of the box:

- **NFHS-5 (2019–21)**: 707 district fact-sheet rows across all 36 States/UTs. Eleven indicators make the deprivation composite; additional NFHS fields support service-coverage proxy views.
- **Census 2011 MPI candidate table**: 629 of 707 NFHS districts are linked after exact/unique fuzzy matching. Its source/year are not documented in the supplied CSV, so the HCR is treated strictly as an external cross-check.
- **Census SC/ST table**: only 84 rows are marked district-published (`PUBLISHED`). The other 623 state-average rows (`STATE_MEAN_ESTIMATED`) are excluded from district analysis.
- **Satellite / weather / scheme administrative data**: not bundled as static defaults. The project does not fabricate synthetic alerts.

See `docs/SOURCES.md` and `docs/DATA_QUALITY.md` for source links, caveats, and cleaning decisions.

## Model and evaluation

- **Explainable composite:** 11 NFHS indicators are direction-normalized and z-scored, averaged into living-standards, education, and health/nutrition domains, then equally averaged. Indicator contributions sum exactly to the composite.
- **Screening tiers:** percentile-based quartiles (`Very high`, `High`, `Moderate`, `Lower`); these are relative ranks within this dataset, not policy thresholds.
- **Confidence flag & gauge:** derived from indicator completeness and NFHS sample size; it is not predictive confidence.
- **Cross-check model:** `xgboost.XGBRegressor` (when `xgboost` is installed, with automatic fallback to scikit-learn `HistGradientBoostingRegressor`), predicting the supplied MPI HCR from NFHS-5 indicators. Five-fold CV on the bundled v8 data with XGBoost: **R² = 0.845**, **MAE = 5.12 percentage points**, **Spearman ρ = 0.89** on 629 districts (HistGradientBoosting fallback: R² = 0.844, MAE = 5.10 pp).
- **Clusters:** K-Means descriptive profiles on the three domain scores.

See `docs/VIVA_ALIGNMENT.md` for a detailed presentation and defense guide.

## Quick start

From the repository root in an activated Python 3.10+ virtual environment:

```bash
python -m pip install -e ".[dev,boost]"
python -m kshetra.pipeline --backend auto
python -m pytest -q
python -m streamlit run app/streamlit_app.py
```

Use `--backend hist_gb` for scikit-learn, or `--backend xgboost` to require XGBoost. For optional provider SDKs install the `llm` extra. Network, credentials and provider model availability are required for external synthesis; otherwise responses use the local template. The scene catalog helper requires `httpx` and accepts `--search-url`.

Verification notes are plain text. Set `KSHETRA_VERIFICATION_DB` to an absolute database path to reuse an existing log (including v5's `data/field_verifications.sqlite`). Otherwise the app uses `kshetra-ai/field_verifications.sqlite` under `XDG_DATA_HOME` or `~/.local/share`. Do not enter personal data.

For the polygon choropleth in Tab 1, place a GeoJSON boundary file at `data/geo/india_districts.geojson` or upload one in the UI (otherwise the default 707-district state-anchored spatial bubble map is shown). For environmental alerts in Tab 2, use the CSV template in `data/environmental/` or query STAC scene metadata with `python scripts/fetch_satellite_stac.py`.

## Repository layout

```
src/kshetra/        ingestion, crosswalk, index, model (XGBoost/HistGB), schemes, environment, verification, TF-IDF RAG
app/                six-module Streamlit dashboard
scripts/            boundary shapefile converter, GeoJSON checker, STAC metadata fetcher, STAC-to-signal bridge
tests/              unit and real-data pipeline tests
data/raw/           tracked NFHS-5, MPI candidate, and Census 2011 SC/ST datasets
data/geo/           optional attributed GeoJSON boundaries
data/environmental/ optional sourced district-date signals and schema
docs/               data quality, sources, roadmap, viva alignment
outputs/            generated district profiles, clusters and model metrics
```

## Responsible use and deployment limits

- Environmental signals indicate environmental stress; they are not poverty evidence.
- Never infer individual eligibility, benefit receipt, or entitlement from these district aggregates.
- Field verification is required before action. The prototype's local review log must not contain personal data.
