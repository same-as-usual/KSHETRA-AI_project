"""Paths and indicator definitions shared by the pipeline and the dashboard."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

# Expected raw file names. Place the downloaded files in data/raw/ with these names.
NFHS_FILE = RAW_DIR / "nfhs5_all_districts.csv"
CENSUS_SCST_FILE = RAW_DIR / "census2011_scst_district.csv"
MPI_FILE = RAW_DIR / "mpi_candidate.csv"

# Deprivation indicators drawn from NFHS-5 (2019-21), district level.
# Each entry: short name -> (domain, direction, label prefix of the NFHS column).
# direction = +1 when a HIGHER value means MORE deprivation (e.g. stunting),
#             -1 when a higher value means LESS deprivation (e.g. electricity coverage).
# The label prefix is matched against the NFHS column names; each must match exactly one column.
INDICATORS: dict[str, tuple[str, int, str]] = {
    "electricity_pct": ("living_standards", -1, "Population living in households with electricity"),
    "improved_water_pct": ("living_standards", -1, "Population living in households with an improved drinking-water"),
    "improved_sanitation_pct": ("living_standards", -1, "Population living in households that use an improved sanitation"),
    "clean_cooking_fuel_pct": ("living_standards", -1, "Households using clean fuel for cooking"),
    "female_school_pct": ("education", -1, "Female population age 6 years and above who ever attended school"),
    "women_literate_pct": ("education", -1, "Women (age 15-49) who are literate"),
    "stunted_pct": ("health_nutrition", +1, "Children under 5 years who are stunted"),
    "wasted_pct": ("health_nutrition", +1, "Children under 5 years who are wasted"),
    "child_anaemia_pct": ("health_nutrition", +1, "Children age 6-59 months who are anaemic"),
    "institutional_birth_pct": ("health_nutrition", -1, "Institutional births (in the 5 years"),
    "fully_vaccinated_pct": ("health_nutrition", -1, "Children age 12-23 months fully vaccinated based on information from either"),
}

# Additional NFHS-5 fields for scheme-gap proxy views. Kept out of the deprivation
# composite and MPI cross-check model so that the original model specification is stable.
# short name -> (NFHS column-label prefix, human label)
AUX_INDICATORS: dict[str, tuple[str, str]] = {
    "health_insurance_pct": (
        "Households with any usual member covered under a health insurance/financing scheme",
        "Households with any health insurance / financing cover (%)",
    ),
}

# Human readable labels for the dashboard.
INDICATOR_LABELS: dict[str, str] = {
    "electricity_pct": "Households with electricity (%)",
    "improved_water_pct": "Population with improved drinking water (%)",
    "improved_sanitation_pct": "Population with improved sanitation (%)",
    "clean_cooking_fuel_pct": "Households with clean cooking fuel (%)",
    "female_school_pct": "Females (6+) who ever attended school (%)",
    "women_literate_pct": "Women (15-49) who are literate (%)",
    "stunted_pct": "Children under 5 who are stunted (%)",
    "wasted_pct": "Children under 5 who are wasted (%)",
    "child_anaemia_pct": "Children 6-59 months who are anaemic (%)",
    "institutional_birth_pct": "Institutional births (%)",
    "fully_vaccinated_pct": "Children 12-23 months fully vaccinated (%)",
}

DOMAIN_LABELS: dict[str, str] = {
    "living_standards": "Living standards",
    "education": "Education",
    "health_nutrition": "Health & nutrition",
}
