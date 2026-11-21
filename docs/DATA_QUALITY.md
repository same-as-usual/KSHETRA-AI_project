# Data quality log

Every issue found in the three raw files, what it affects, and how the pipeline handles it.
Re-check these numbers if the raw files change: they were measured on the files supplied with the project.

## 1. Census 2011 SC/ST table (`census2011_scst_district.csv`)

| Issue | Measured | Handling |
|---|---|---|
| Most rows are state averages, not district values. `Source = STATE_MEAN_ESTIMATED` | 623 of 707 rows | Set to NaN unless `Source = PUBLISHED` (84 rows). The `census_scst_pct` column is empty for the other rows. |
| The `Note` column says the estimated rows use "Census 2011 state-level published means applied". | All estimated rows | Not used anywhere in scoring. |
| Published SC+ST shares disagree with SC+ST counts from the Census 2011 MPI table for the same districts. | 84 districts: correlation 0.91, mean absolute difference 5.9 percentage points | The MPI-table counts are used for SC/ST shares in the dashboard. The disagreement should be resolved with the original Census PCA tables before SC/ST shares are used for decisions. |

## 2. Census 2011 MPI table (`mpi_candidate.csv`)

| Issue | Measured | Handling |
|---|---|---|
| Trailing comma in the header creates an unnamed 33rd column. | Every row | Dropped on load. |
| `Households` does not equal rural + urban households. `Households.1` does. | 638 of 638 rows | `Households.1` is used as the household count. `Households` is ignored. |
| Literate males + literate females does not equal the literate total. The male/female literate columns are unreliable. | 636 of 638 rows | Only the total literate population is used (`census_literacy_pct`). Male/female literacy is not used. |
| Two district keys appear twice with different MPI HCR values (Hamirpur, Uttar Pradesh; Hamirpur, Himachal Pradesh). | 4 rows | Both rows for each conflicting key are excluded, with a warning. We cannot tell which value is correct. |
| Religion counts do not sum to total population. | 638 of 638 rows | Religion columns are not used. |
| MPI HCR is stored as text with a `%` sign. Its source and year are not documented in the file. | All rows | Parsed to a number. **Provenance must be confirmed before this is presented as official.** |
| Telangana districts sit under "Andhra Pradesh" (2011 boundaries). Some districts were renamed or split after 2011. | 31 Telangana districts in NFHS-5 (9 matched by fuzzy name) | Matched by exact name, then by fuzzy name (ratio >= 0.9, unique best match), searching Telangana for Andhra Pradesh rows. 12 districts matched by fuzzy name in total (9 in Telangana). Boundaries changed, so these values are approximate. |
| 5 Census 2011 districts have no clear NFHS-5 counterpart (boundaries split, or ambiguous names). | Barddhaman (West Bengal), Dakshin Bastar Dantewada (Chhattisgarh), Jaintia Hills (Meghalaya), Saiha (Mizoram), Warangal (Andhra Pradesh) | Left unmatched. They have no MPI value in the district frame. |

## 3. NFHS-5 district fact sheet (`nfhs5_all_districts.csv`)

| Issue | Measured | Handling |
|---|---|---|
| Suppressed cells are marked `*`. | 4,132 cells across all columns; only 13 districts are missing `fully_vaccinated_pct` among our 11 indicators | Parsed as missing. Missing values are imputed with the national median inside the index, and the count is shown per district (`indicators_missing`). |
| Values in parentheses are small-sample estimates. | 236 cells among our 11 indicators | Kept, because the value is still informative. They are not flagged individually yet. |
| Column names include footnote numbers (e.g. `...electricity...`, `...literate4 (%)`) and stray spaces. | Many columns | Matched by stripped name prefix. Each indicator prefix must match exactly one column, or loading fails. |
| District and state names differ from Census 2011 (e.g. `Maharastra`, `Odisha` vs `ORISSA`, `Telangana`, `Ladakh`, merged Dadra & Nagar Haveli and Daman & Diu). | Several | Normalised with `kshetra/text.py` and aliases in `kshetra/crosswalk.py`. |
| NFHS-5 (2019-21) and Census 2011 use different district boundaries for some areas. | Many districts | Joins on names. Values for renamed or split districts are approximate. |

## 4. How these issues affect the results

- **Scoring uses NFHS-5 only.** The composite index does not depend on the Census SC/ST table or on the MPI table, so the data problems above do not change it.
- **The MPI table is used as an external reference**, to check that the composite ranks districts the way an established poverty measure does (Spearman correlation 0.89 on 629 districts). Because the MPI provenance is undocumented and MPI and NFHS both measure similar deprivations, this check is supportive but **not independent proof of accuracy**.
- **Cross-check model**: the regression model estimates MPI from NFHS-5 indicators (5-fold cross-validated R² 0.84). Its estimates for the 78 districts without an MPI record are model outputs, not observations.
- **Confidence labels** describe data completeness and survey sample size only. They do not measure how accurate the index is.

## 5. What is still missing

- Satellite signals (CHIRPS rainfall, VIIRS night lights, Sentinel-2 land use) are not in these files and were not integrated.
- No scheme-coverage data (the SchemeGap module) was provided.
- The MPI file needs a documented source and reference year.
