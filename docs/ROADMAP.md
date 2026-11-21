# Roadmap and implementation status

Aligned to the six modules in the KSHETRA-AI panel presentation. “Implemented” means a working code path exists; data-dependent modules remain empty until their real inputs are supplied.

| Presentation module / stage | Status | What works / what remains |
|---|---|---|
| 1. Poverty Intelligence Atlas | Implemented with optional boundary file | District filtering, ranking and interactive GeoJSON choropleth. Upload a FeatureCollection with state/district properties; the app exact-matches and excludes unmatched polygons. No boundary asset is bundled because available public boundaries have known vintage/quality caveats. |
| 2. Risk & Early Warning | Input/alert engine implemented; no signals bundled | Validates sourced dated district aggregates and calculates transparent anomaly screening scores. No CHIRPS/VIIRS/Sentinel-2/FIRMS measurements are bundled or fetched automatically. |
| 3. Explainable AI Insights | Implemented | Indicator contributions, domain scores, percentile, data-completeness confidence, external MPI cross-check, and local human-verification log. |
| 4. Action Planner | Scenario implemented | Allocates a user-entered budget share across four service-access proxies. Includes a user-assumed gap-reduction target chart; it is not a forecast or impact estimate. |
| 5. SchemeGap Intelligence | Six-proxy view implemented | NFHS-5 clean fuel, sanitation, any-health-insurance, improved-water, stunting, and female school-attainment indicators are mapped to programme areas. They are broad access/outcome proxies, not scheme enrollment or eligibility; only four comparable service-access proxies enter budget shares. Administrative programme data is still needed. |
| 6. PRAGATI AI Assistant | Local retrieval prototype implemented | Retrieves district facts and proxy evidence; answers with constrained templates in English and limited Hindi common intents. No generative LLM is connected. |
| XGBoost baseline (PPT slide 7) | Not implemented | Current cross-check uses scikit-learn HistGradientBoostingRegressor. Talk track and README must say so; do not call it XGBoost. |
| K-Means district profiles | Implemented | Descriptive clustering on the 3 NFHS domain scores. |
| CNN/ViT imagery model, LSTM/Transformer forecast | Not implemented | Do not claim these in a demo. Requires verified imagery/time series and a separate model evaluation. |
| Scheme-specific administrative coverage | Not implemented | Needed to replace the NFHS service proxies with true SchemeGap analysis. |
| Multilingual LLM / full RAG | Not implemented | Current assistant is retrieval + fixed templates, not generated long-form advice. |
| Field-verification loop | Local audit-log prototype implemented | Aggregate-only SQLite form; no login/permissions or cross-user deployment controls. Do not enter personal data. |

## Next priorities

1. Confirm the MPI file's source, year and licensing. Fix the Census district-name crosswalk with Census district codes, not name-only matching.
2. Select a current administrative boundary dataset and validate its license/vintage against NFHS-5.
3. Build a reproducible geospatial pipeline from source rasters to district-date anomalies (CHIRPS, VIIRS Black Marble, Sentinel-2 NDVI; FIRMS fire detections). Preserve quality flags and provenance.
4. Obtain dated scheme-administrative coverage by district, with definitions and denominators, before presenting enrollment gaps or allocating operational budgets.
5. Add authentication, role-based access, retention and review controls before multi-user field verification.
6. If needed, integrate an approved generative model only after evidence retrieval, citation display, safety, and multilingual quality are evaluated.
