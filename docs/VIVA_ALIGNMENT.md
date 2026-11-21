# Viva / presentation alignment notes

This note makes implementation boundaries explicit so a demo does not overclaim.

## Demo claims that are now supportable

- **Atlas:** an interactive choropleth can be shown when a GeoJSON FeatureCollection is uploaded. It performs exact normalized state/district matching and leaves unmatched polygons out. The supplied DataMeet boundary source warns that some shapes/names are outdated or shifted; see `data/geo/README.md`.
- **Risk & Early Warning:** a rule-based score runs only on dated district observations the user uploads. It is not a forecast and no satellite data ships in the current build.
- **Explainability:** the composite is a transparent equal-domain index. Driver contributions exactly sum to the composite. Confidence is a data-completeness/sample-size flag, not calibrated model uncertainty.
- **Action Planner:** the budget splitter allocates scenario shares according to four service-access proxies. Its before/after bar is a user-selected target assumption, not an optimized plan or impact estimate.
- **SchemeGap:** six NFHS-5 access/outcome indicators are mapped to programme areas. They do not prove eligibility or measure scheme enrollment; only four comparable service-access gaps are used in budget shares.
- **PRAGATI:** local retrieval plus constrained answer templates returns evidence cards. There is no live LLM, API key, or external generation.
- **Human verification:** the app can save a district-level review to a local SQLite audit log. It is not an authenticated case-management system.

## Claims that must not be made

- “We have a live satellite feed,” “real-time poverty map,” or “trained CNN/ViT/LSTM/Transformer.” None is present.
- “Our model uses XGBoost.” The tested baseline is scikit-learn `HistGradientBoostingRegressor`.
- “PMUY/SBM-G/PM-JAY beneficiaries/eligible households are identified.” Only NFHS-5 service proxies are present.
- “Budget predicts household reach or poverty reduction.” Unit costs, eligibility denominators and causal impact data are absent.
- “The MPI benchmark is official.” The supplied `mpi_candidate.csv` does not identify a source/year and must be verified.

## Recommended explanation of model mismatch

> “The slide proposed XGBoost, but our current reproducible baseline is scikit-learn HistGradientBoostingRegressor because that library is available in our environment. It predicts the supplied MPI HCR from NFHS-5 indicators as a cross-check; it is not an early-warning forecast. We report five-fold validation, but the MPI file provenance is not documented, so this is not independent validation. We have not implemented image or temporal neural networks.”

If the submitted slide must match the shipped code exactly, change the ML slide to say “scikit-learn gradient-boosted trees (baseline)” and move CNN/ViT/LSTM/Transformer from “implemented” to “future work.”
