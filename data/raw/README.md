# Raw data

Place these three files here. They are **not committed to git** (see `.gitignore`), because their licences and redistribution terms should be checked first.

| File | Source | Used for |
|---|---|---|
| `nfhs5_all_districts.csv` | NFHS-5 (2019-21) district fact sheet, IIPS / Ministry of Health and Family Welfare. Download: https://www.nfhs.in | Deprivation indicators (11) and survey sample sizes |
| `mpi_candidate.csv` | Census 2011 district table with MPI headcount ratio. **Source and year not documented in the file: confirm before use.** | External reference (MPI HCR), SC/ST and literacy shares |
| `census2011_scst_district.csv` | Census 2011 PCA tables B-13 (SC) and B-14 (ST). Download: https://censusindia.gov.in | Published district SC/ST shares only (84 rows). Other rows are state estimates and are ignored. |

See `docs/DATA_QUALITY.md` for known issues in each file.
