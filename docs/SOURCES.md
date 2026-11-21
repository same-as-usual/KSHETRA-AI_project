# Sources and attribution

## Data currently included

- **NFHS-5 district fact sheets, 2019-21**: International Institute for Population Sciences (IIPS) and Ministry of Health and Family Welfare. Open Government Data catalog: https://www.data.gov.in/catalog/national-family-health-survey-5-nfhs-5-india-districts-factsheet-data-provisional . The bundled file is a user-provided aggregate CSV; verify its exact release and column provenance before redistribution.
- **Census 2011 SC/ST district table**: Primary Census Abstract B-13/B-14, Census of India: https://censusindia.gov.in/census.website/data/census-tables . The supplied derived file labels 84 district rows `PUBLISHED` and 623 rows `STATE_MEAN_ESTIMATED`; only the former are treated as district values.
- **MPI candidate table**: supplied by the project owner. The CSV does not identify a source or reference year. Treat it as unverified until documented.

## Optional map / environmental sources

- **India districts**: DataMeet Community Maps Project, district boundaries page: https://projects.datameet.org/maps/districts/ . The page documents CC BY 2.5 India and cautions that some state boundaries are pre-delimitation, shifted, or have inconsistent names. Attribute DataMeet and inspect geometry vintage before use.
- **CHIRPS v3**: Climate Hazards Center, UCSB: https://www.chc.ucsb.edu/data/chirps3 . This source provides gridded rainfall products; a spatial aggregation step is still required.
- **VIIRS Black Marble / nighttime lights**: NASA LAADS DAAC: https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/science-domain/nighttime-lights/ . Night lights are an activity proxy, not a poverty label.
- **NASA FIRMS active fires**: https://firms.modaps.eosdis.nasa.gov/api/ . Data access may require a key; do not commit or paste secrets. Fire detections are a separate shock signal.
- **Sentinel-2**: Copernicus Data Space documentation: https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html . The app does not download imagery or compute NDVI from raw bands yet.

## Scheme information (not coverage data)

- PMUY official portal: https://www.pmuy.gov.in/
- SBM-G portal: https://swachhbharatmission.ddws.gov.in/
- PM-JAY portal: https://pmjay.gov.in/
- Jal Jeevan Mission: https://jaljeevanmission.gov.in/ . NFHS improved-water access is not a functional tap-connection count.
- POSHAN Abhiyaan: https://poshanabhiyaan.gov.in/ . NFHS stunting is an outcome indicator, not program coverage.
- Samagra Shiksha, Ministry of Education: https://samagra.education.gov.in/about.html . NFHS female ever-attended-school is an attainment proxy, not current enrollment or scheme coverage.

These pages describe programme information; the current project does not ingest beneficiary-level or district administrative coverage. NFHS fields in SchemeGap are only broad service coverage proxies.
