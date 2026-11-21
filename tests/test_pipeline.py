"""Tests for the KSHETRA-AI data pipeline.

Run with:  python -m pytest -q        (pytest is listed in requirements-dev.txt)
The tests use the raw CSVs in data/raw/ and are skipped when they are absent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kshetra.config import INDICATORS, MPI_FILE, NFHS_FILE  # noqa: E402
from kshetra.index import compute_index  # noqa: E402
from kshetra.text import norm_name, norm_state, parse_number  # noqa: E402

HAS_DATA = NFHS_FILE.exists() and MPI_FILE.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="raw CSVs not present in data/raw/")


# ---------- text helpers ----------

@pytest.mark.parametrize("raw, expected", [
    ("Y.S.R.  ", "YSR"),
    ("Andaman & Nicobar Islands", "ANDAMAN AND NICOBAR ISLANDS"),
    ("  Kanniyakumari ", "KANNIYAKUMARI"),
])
def test_norm_name(raw, expected):
    assert norm_name(raw) == expected


def test_norm_state_aliases():
    assert norm_state("ORISSA") == "ODISHA"
    assert norm_state("Pondicherry") == "PUDUCHERRY"


@pytest.mark.parametrize("raw, expected", [
    ("(29.5)", 29.5), ("27.12%", 27.12), ("1,234", 1234.0), (" 82.7 ", 82.7), (3, 3.0),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["*", "", None, "NA"])
def test_parse_number_missing(raw):
    assert math.isnan(parse_number(raw))


# ---------- index (synthetic data, no files needed) ----------

def _synthetic_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {name: rng.uniform(0, 100, n) for name in INDICATORS}
    data["survey_households"] = np.full(n, 800)
    return pd.DataFrame(data)


def test_index_contributions_sum_to_composite():
    out = compute_index(_synthetic_frame(), n_clusters=3)
    contrib_cols = [f"contrib_{k}" for k in INDICATORS]
    total = out[contrib_cols].sum(axis=1)
    assert np.allclose(total, out["composite_index"], atol=1e-9)


def test_direction_higher_deprivation_scores_higher():
    base = _synthetic_frame()
    worse = base.copy()
    # Stunting is a deprivation indicator: a higher value for district 0 must raise its composite.
    worse.loc[0, "stunted_pct"] += 1000
    # Electricity is a coverage indicator: lower coverage for district 1 must raise its composite.
    worse.loc[1, "electricity_pct"] -= 1000
    a = compute_index(base, n_clusters=3)
    b = compute_index(worse, n_clusters=3)
    assert b.loc[0, "composite_index"] > a.loc[0, "composite_index"]
    assert b.loc[1, "composite_index"] > a.loc[1, "composite_index"]


def test_missing_values_are_imputed_and_counted():
    frame = _synthetic_frame()
    frame.loc[0, ["stunted_pct", "wasted_pct"]] = np.nan
    out = compute_index(frame, n_clusters=3)
    assert out.loc[0, "indicators_missing"] == 2
    assert np.isfinite(out["composite_index"]).all()
    assert out.loc[0, "confidence"] != "High"


# ---------- real data ----------

@needs_data
def test_nfhs_loads_707_districts_with_all_indicators():
    from kshetra.ingest import load_nfhs
    nfhs = load_nfhs()
    assert len(nfhs) == 707
    assert nfhs[list(INDICATORS)].notna().mean().min() > 0.9


@needs_data
def test_mpi_conflicting_duplicates_are_excluded():
    from kshetra.ingest import load_mpi
    mpi = load_mpi()
    assert not mpi.duplicated(["state_key", "district_key"]).any()
    assert not ((mpi.state_key == "UTTAR PRADESH") & (mpi.district_key == "HAMIRPUR")).any()


@needs_data
def test_frame_has_one_row_per_nfhs_district_and_no_fake_scst_values():
    from kshetra.crosswalk import build_district_frame
    df = build_district_frame()
    assert len(df) == 707
    assert not df.duplicated(["state_key", "district_key"]).any()
    # Census SC/ST values from the state-estimated rows must not leak into the frame.
    assert df["census_scst_pct"].notna().sum() == df["scst_published"].sum() == 84
    assert df["mpi_hcr_pct"].notna().sum() >= 620

# ---------- six-module helpers ----------

def test_scheme_gap_is_labeled_as_proxy_and_budget_sums():
    from kshetra.schemes import plan_budget, scheme_gap_frame
    profile = pd.DataFrame([{
        "state_raw": "State A", "district_raw": "District A", "state_key": "STATE A", "district_key": "DISTRICT A",
        "clean_cooking_fuel_pct": 40.0, "improved_sanitation_pct": 60.0,
        "health_insurance_pct": 25.0, "survey_households": 700,
    }])
    gaps = scheme_gap_frame(profile)
    assert len(gaps) == 3
    assert set(gaps["survey_period"]) == {"NFHS-5, 2019-21"}
    assert gaps["gap_proxy_pct"].between(0, 100).all()
    plan = plan_budget(gaps, 10_000_000)
    assert plan["allocation_inr"].sum() == pytest.approx(10_000_000)
    assert plan["interpretation"].str.contains("not").all()


def test_environmental_rule_score_uses_only_uploaded_observations():
    from kshetra.environment import score_environmental_risk
    observed = pd.DataFrame([{
        "state": "State A", "district": "District A", "observed_at": "2025-07-01",
        "source": "CHIRPS v3 + Sentinel-2 L2A, aggregated per district",
        "rainfall_deficit_pct": 40, "ndvi_change_pct": -30,
        "viirs_change_pct": None, "fire_count": 0,
    }])
    scored = score_environmental_risk(observed)
    assert scored.loc[0, "environmental_score"] == pytest.approx(100)
    assert scored.loc[0, "alert_level"] == "High signal"
    assert scored.loc[0, "requires_field_verification"]
    assert scored.loc[0, "source"].startswith("CHIRPS")


def test_environmental_input_rejects_missing_provenance():
    from kshetra.environment import validate_environmental_signals
    bad = pd.DataFrame([{"state": "A", "district": "B", "observed_at": "2025-01-01",
                         "rainfall_deficit_pct": 20}])
    with pytest.raises(ValueError, match="source"):
        validate_environmental_signals(bad)


def test_local_assistant_retrieves_district_evidence():
    from kshetra.index import compute_index
    from kshetra.pragati import answer_query
    base = _synthetic_frame(n=40, seed=7)
    base["state_raw"] = ["State A"] * 40
    base["district_raw"] = [f"District {i}" for i in range(40)]
    base["state_key"] = ["STATE A"] * 40
    base["district_key"] = [f"DISTRICT {i}" for i in range(40)]
    base["mpi_hcr_pct"] = np.nan
    profiles = compute_index(base, n_clusters=3)
    reply = answer_query("Why is this district flagged?", profiles, district="District 0", state="State A")
    assert "District 0" in reply["answer"]
    assert reply["evidence"]
    assert any("NFHS-5" in e["source"] for e in reply["evidence"])


def test_geojson_exact_join_does_not_guess():
    from kshetra.geo import match_geojson_to_profiles
    profiles = pd.DataFrame([{
        "state_raw": "State A", "district_raw": "District A", "state_key": "STATE A",
        "district_key": "DISTRICT A", "composite_percentile": 75.0,
    }])
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"ST_NM": "State A", "DISTRICT": "District A"},
         "geometry": {"type": "Polygon", "coordinates": []}},
        {"type": "Feature", "properties": {"ST_NM": "State A", "DISTRICT": "Unknown"},
         "geometry": {"type": "Polygon", "coordinates": []}},
    ]}
    matched_geo, matched = match_geojson_to_profiles(geo, profiles, "DISTRICT", "ST_NM")
    assert len(matched) == 1
    assert len(matched_geo["features"]) == 1
    assert matched_geo["features"][0]["properties"]["_kshetra_key"] == "STATE A::DISTRICT A"


def test_verification_log_is_append_only_and_local(tmp_path):
    from kshetra.verification import add_verification, list_verifications
    path = tmp_path / "verification.sqlite"
    add_verification("State A", "District A", "Verified concern", "Education", "Aggregate review", "Block review", path)
    rows = list_verifications(path)
    assert len(rows) == 1
    assert rows.loc[0, "district"] == "District A"
    assert rows.loc[0, "status"] == "Verified concern"
