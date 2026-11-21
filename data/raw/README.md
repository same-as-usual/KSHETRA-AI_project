# Raw data

This local project bundle includes the three aggregate CSVs supplied by the project owner in `data/raw/`. They are git-ignored so they are not accidentally published. Confirm source URLs, licenses, and redistribution terms before committing or sharing them publicly.

| File | Source / note | Used for |
|---|---|---|
| `nfhs5_all_districts.csv` | NFHS-5 district fact sheet, 2019-21; see `docs/SOURCES.md` | Deprivation indicators and service-coverage proxies |
| `mpi_candidate.csv` | User-provided Census 2011 district MPI candidate table; the file itself has no source/year provenance | External benchmark only; confirm before external use |
| `census2011_scst_district.csv` | User-provided derived table referencing Census PCA B-13/B-14; 84 district values are marked published, 623 are state-level estimates | Published district SC/ST values only |

See `docs/DATA_QUALITY.md` and `docs/SOURCES.md` before using results.
