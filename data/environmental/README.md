# Environmental observations input (optional)

No environmental/satellite observations are included with the project. This prevents a demo or placeholder from being mistaken for a measured signal. Use real values derived from the official dataset portals below and keep one row per district/date.

Required CSV columns:

```csv
state,district,observed_at,source,rainfall_deficit_pct,ndvi_change_pct,viirs_change_pct,fire_count
```

- `rainfall_deficit_pct`: percent rainfall shortfall against a documented seasonal normal; positive is below normal. Source and baseline interval should be recorded in `source`/your accompanying metadata.
- `ndvi_change_pct`: percent change from the same seasonal baseline; negative means vegetation decline.
- `viirs_change_pct`: percent change from the chosen VIIRS radiance baseline; negative means lower radiance. Never treat light as poverty proof.
- `fire_count`: number of NASA FIRMS detections aggregated to the district and time window. It is a shock flag, not a poverty measure.
- At least one numeric environmental measurement is required per row. Blank other values when unavailable.

KSHETRA-AI scales available rainfall, NDVI and VIIRS anomaly metrics using transparent screening thresholds. The resulting score is not a trained probability or poverty classification. Fire detections are shown separately and do not enter that score. Every row must preserve its source and observation date.

## Official sources

- CHIRPS v3 rainfall products (daily/pentad/monthly formats): [Climate Hazards Center](https://www.chc.ucsb.edu/data/chirps3).
- NASA FIRMS near-real-time active fire observations: [FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/).
- VIIRS Black Marble nighttime-light products: [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/science-domain/nighttime-lights/).
- Sentinel-2 data access and product documentation: [Copernicus Data Space](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html).

District aggregation from gridded/raster products needs boundary geometries, time-window choices, cloud/quality filtering, and reproducible processing metadata. The current app accepts the resulting district-level CSV and does not download or aggregate raw rasters. `scripts/stac_to_environment_csv.py` converts a zonal-means table (computed externally, e.g. in QGIS) and the STAC scene metadata into this schema. It blanks Sentinel-2 NDVI change when a scene is unknown or cloudier than `--max-cloud` (default 20%), and reports rejected rows.
