# Viva / presentation alignment notes

This note makes implementation boundaries explicit so your live demo and technical defense are 100% consistent with the codebase.

## Demo claims that are directly supported in code

- **1. Poverty Intelligence Atlas:** renders an out-of-the-box 707-district state-anchored spatial bubble map across all 36 States/UTs (including the normalized `MAHARASTRA -> MAHARASHTRA` alias fix), and upgrades to an exact-matched polygon choropleth when a GeoJSON FeatureCollection is uploaded.
- **2. Risk & Early Warning:** validates dated district environmental CSVs, renders a 3-axis polar **Risk Radar** (`go.Scatterpolar` for rainfall deficit, vegetation decline, and night-light decline), computes **computer-vision change-detection metrics** (`ΔExG`, `ΔNGRDI`, pixel intensity shift, changed area %, and terracotta/teal difference heatmap) on uploaded before/after satellite composites, and includes `scripts/fetch_satellite_stac.py` for querying Microsoft Planetary Computer STAC scenes.
- **3. Explainable AI Insights:** computes an equal-domain z-score composite where the 11 indicator contributions sum exactly to the composite score, alongside a radial **Indicator Completeness Gauge** and local SQLite human-verification audit log.
- **4. Action Planner:** splits hypothetical budgets across the 4 comparable service-access proxies, estimates illustrative units funded using reference unit costs, and runs a **counterfactual index simulation** (`simulate_counterfactual_profile`) that recalculates the district's composite score, national percentile shift, and screening tier change.
- **5. SchemeGap Intelligence:** covers 6 NFHS-5 proxies (PMUY, SBM-G, PM-JAY, Jal Jeevan Mission, POSHAN Abhiyaan, Samagra Shiksha), explicitly separating the 4 service-access proxies from the 2 nutrition/education outcome proxies, and displays the 4-step verification workflow.
- **6. PRAGATI AI Assistant:** implements a **TF-IDF vector RAG pipeline** (`TfidfVectorizer` + `cosine_similarity`) over 707 district profiles and 6 programme proxies, displays 4 district KPI cards and a **Contextual Peer Locator Map**, supports Devanagari and Romanized Hindi, and includes an optional **Gemini / OpenAI API key** synthesis toggle that falls back to deterministic templates offline.
- **ML Cross-Check (`src/kshetra/model.py`):** natively runs **`xgboost.XGBRegressor`** (5-fold CV **R² = 0.849**, **MAE = 5.02 pp**, **Spearman ρ = 0.89** across 629 MPI-linked districts), with automatic fallback to scikit-learn `HistGradientBoostingRegressor` (**R² = 0.844**).

## Claims that must not be overclaimed

- **No fabricated live satellite alerts:** the app never invents rainfall, NDVI, VIIRS, or fire numbers when no observation CSV is uploaded.
- **Optional modules are not part of the demo:** the zonal bridge, CNN-embedding path, drought transition baseline and MIS comparison are code paths with synthetic-data tests. None is wired into the dashboard, none ships with real data, and none has skill metrics. Say so explicitly if asked.
- **No black-box CNN/ViT/LSTM weights:** instead of shipping unverified neural weights without a labeled satellite ground-truth dataset, Module 2 uses transparent spectral/pixel change-detection indices (`ΔExG`, `ΔNGRDI`) and threshold rules.
- **Proxies vs. administrative MIS enrollment:** NFHS-5 indicators measure household survey coverage/outcomes (2019–21), not scheme beneficiary rolls or individual household eligibility.
- **MPI provenance:** `mpi_candidate.csv` is used strictly as an external cross-check benchmark until its exact publication vintage is documented.
