# KSHETRA-AI

**From deprivation data to localized action.** KSHETRA-AI is a district-level poverty-risk and intervention-intelligence platform for India. This repository holds the first build: a reproducible data pipeline on real district datasets, an explainable deprivation index, a cross-check model, and a dashboard prototype.

> A high score flags a district for **verification**. It does not show that people in the district are poor, and no decision about any individual should be based on it.

## What is built

| Component | Where | Status |
|---|---|---|
| Ingestion and cleaning of NFHS-5, Census 2011 MPI table, Census 2011 SC/ST table | `src/kshetra/ingest.py` | Working |
| Name matching across datasets (exact, then fuzzy with a unique-match rule) | `src/kshetra/crosswalk.py`, `src/kshetra/text.py` | Working: 629 of 707 districts linked to Census 2011 MPI |
| Explainable deprivation index (11 indicators, 3 domains) with driver breakdown and confidence | `src/kshetra/index.py` | Working |
| K-Means district profile clusters | `src/kshetra/index.py` | Working |
| Cross-check model: MPI from NFHS-5 indicators (gradient boosting) | `src/kshetra/model.py` | Working: 5-fold CV R² 0.84 |
| Dashboard: atlas, district profile, model check, data notes | `app/streamlit_app.py` | Prototype (not yet run in a browser) |
| Satellite signals (CHIRPS, VIIRS, Sentinel-2) | none yet | Not started. See `docs/ROADMAP.md` |
| SchemeGap, intervention planner, RAG assistant | none yet | Not started |

## Data sources

Real public datasets, placed by hand in `data/raw/` (not committed). See `data/raw/README.md`.

- **NFHS-5 (2019-21)**: district fact sheet, 707 districts. Indicators used are listed in `src/kshetra/config.py`.
- **Census 2011 MPI table**: district multidimensional-poverty headcount ratio (MPI HCR). Its source is not documented in the file, so confirm it before external use.
- **Census 2011 SC/ST table**: only 84 rows hold district values. The other 623 rows hold state averages and are ignored.

Every raw-data issue found, and how it was handled, is listed in `docs/DATA_QUALITY.md`.

## Method

1. Each NFHS-5 indicator is imputed with the national median when missing, z-scored across districts, and signed so that a positive value means more deprivation.
2. Indicators are averaged into three domains: living standards, education, and health & nutrition.
3. The composite index is the mean of the three domain scores. It is exactly the sum of per-indicator contributions, which drives the "why this district?" chart.
4. Risk tiers come from the composite percentile: Very high (top quarter), High, Moderate, Lower.
5. Confidence is High when all indicators are present and the survey has at least 500 households, Medium when at most two are missing and at least 200 households were surveyed, and Low otherwise. It measures data completeness, not model accuracy.

Validation: the composite ranks districts close to the Census 2011 MPI HCR (Spearman 0.89 on 629 districts). The MPI provenance is undocumented, so treat this as a consistency check, not proof of accuracy.

## Quick start

1. Create a virtual environment and install the packages listed in `requirements-dev.txt`.
2. Put the three CSVs in `data/raw/` (see `data/raw/README.md`).
3. Build the dataset, index and model outputs: `python -m kshetra.pipeline`
4. Run the tests: `python -m pytest -q`
5. Open the dashboard: `streamlit run app/streamlit_app.py`

Outputs are written to `outputs/`: `district_profiles.csv` (one row per district, all indicators, scores, tiers, drivers, confidence and MPI cross-check), `profile_clusters.csv`, and `model_metrics.json`.

## Repository layout

```
src/kshetra/      pipeline code (config, text, ingest, crosswalk, index, model, pipeline)
app/              Streamlit dashboard
tests/            pytest suite (real-data tests skip when CSVs are absent)
data/raw/         place source CSVs here (git-ignored)
docs/             data-quality log and roadmap
outputs/          generated outputs
```

## Responsible use

- Satellite and survey signals indicate **risk**, not proof of poverty.
- Every flagged district needs **local field verification** before action.
- No automatic decisions about individuals.
- Confidence and data limits are shown with every result.
