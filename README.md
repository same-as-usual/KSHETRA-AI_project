# KSHETRA-AI

**From deprivation data to localized action.** KSHETRA-AI is a district-level deprivation screening and intervention-planning prototype for India, aligned to the six modules in the project presentation.

> A score is a screening signal for review. It is not proof of poverty, a household eligibility determination, or a recommendation about an individual.

## Six modules

| Module | What is implemented | Data / limits |
|---|---|---|
| **1. Poverty Intelligence Atlas** | Interactive district choropleth, filters, state summaries and ranked table. | Requires a GeoJSON boundary file. Exact name matches only; boundary vintage and unmatched districts are disclosed. |
| **2. Risk & Early Warning** | Import validation, rule-based anomaly screening, dated trend display, optional before/after image comparison. | No environmental data is bundled or fetched automatically. Upload attributed observations from CHIRPS, VIIRS, Sentinel-2 and/or FIRMS; no alerts appear without uploaded data. |
| **3. Explainable AI Insights** | Composite percentile, risk tier, indicator contributions, domain scores, confidence flag, MPI cross-check, aggregate human review form. | Confidence describes missingness/sample size, not calibrated model certainty. Local SQLite log; no user authentication or multi-user controls. |
| **4. Action Planner** | User-entered hypothetical budget split in proportion to four service-access proxies; includes a user-assumed before/after target visualization. | Not a cost optimizer, beneficiary count, entitlement decision or impact forecast. Unit costs and eligible-population denominators are absent. |
| **5. SchemeGap Intelligence** | Six NFHS-5 service/outcome proxies (clean fuel, sanitation, health-insurance, improved water, child stunting and female school attainment) with official scheme links. | These are broad service/outcome proxies, not scheme enrollment or eligibility. Only four service-access proxies enter the budget-share scenario; nutrition/education outcomes are context only. Administrative scheme-coverage data is still needed. |
| **6. PRAGATI AI Assistant** | Local district retrieval with evidence cards and constrained answer templates; limited common Hindi intents. | No external LLM, API key or free-form generated advice is connected. |

## Verified data currently in the project

The bundle includes the aggregate CSV files supplied with the project. They are retained locally under `data/raw/` and ignored by Git by default; confirm source, licensing and redistribution terms before publishing them.

- **NFHS-5 (2019–21)**: 707 district fact-sheet rows. Eleven indicators make the deprivation composite; additional NFHS fields support service-coverage proxy views.
- **Census 2011 MPI candidate table**: 629 of 707 NFHS districts are linked after exact/unique fuzzy matching. Its source/year are not documented in the supplied CSV, so the HCR is only an unverified external cross-check.
- **Census SC/ST table**: only 84 rows are marked district-published. The other 623 state-average rows are excluded from district analysis.
- **Satellite / weather / scheme administrative data**: not included. The project does not fabricate sample signals or claim live ingestion.

See `docs/SOURCES.md` and `docs/DATA_QUALITY.md` for source links, caveats, and cleaning decisions.

## Model and evaluation

- **Explainable composite:** 11 NFHS indicators are direction-normalized and z-scored, averaged into living-standards, education, and health/nutrition domains, then equally averaged. Indicator contributions sum to the composite.
- **Screening tiers:** percentile-based quartiles (`Very high`, `High`, `Moderate`, `Lower`); these are relative ranks within this dataset, not policy thresholds.
- **Confidence flag:** derived from indicator missingness and NFHS sample size; it is not predictive confidence.
- **Cross-check model:** scikit-learn `HistGradientBoostingRegressor`, predicting the supplied MPI HCR from NFHS-5 indicators. Current five-fold CV: R² 0.84, MAE 5.1 percentage points on 629 districts. MPI provenance is undocumented, so this is not independent validation.
- **Clusters:** K-Means descriptive profiles on the three domain scores.

The presentation mentions XGBoost, CNN/ViT and temporal models. Those are **not** implemented in this build. The current model is not a livelihood-risk forecast. See `docs/VIVA_ALIGNMENT.md` for a careful demo talk track.

## Quick start

1. Use Python 3.10+ and set up the project environment from `pyproject.toml` (the `dev` extra includes the test runner).
2. Rebuild derived data and model outputs: `python -m kshetra.pipeline`.
3. Run checks: `python -m pytest -q`.
4. Start the dashboard: `streamlit run app/streamlit_app.py`.

For the district map, download a GeoJSON boundary file, convert a sourced shapefile with `scripts/convert_district_shapefile.py` (optional GeoPandas dependency), or upload a GeoJSON in the Atlas. Read `data/geo/README.md` first. For environmental alerts, use the empty CSV template under `data/environmental/` and follow its schema and provenance requirements.

Outputs are written to `outputs/`: district profiles with scores and contributions, cluster summaries, and cross-check metrics. Environmental observations and the field-verification SQLite database are not written to the public model outputs.

## Repository layout

```
src/kshetra/        ingestion, crosswalk, index, model, schemes, environment, verification, RAG
app/                six-module Streamlit dashboard
scripts/            optional boundary conversion utility
tests/              unit and real-data pipeline tests
data/raw/           supplied aggregate datasets (git-ignored)
data/geo/           optional attributed GeoJSON boundaries
data/environmental/ optional sourced district-date signals and schema
docs/               data quality, sources, roadmap, viva alignment
outputs/             generated district profiles, clusters and model metrics
```

## Responsible use and deployment limits

- Environmental signals indicate environmental stress; they are not poverty evidence.
- Never infer individual eligibility, benefit receipt, or entitlement from these district aggregates.
- Field verification is required before action. The prototype's local review log must not contain personal data.
- For operational deployment, add current administrative coverage, access controls, retention policy, independent validation, and monitored data-quality procedures.
