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
    assert norm_state(" Lakshadweep ") == "LAKSHADWEEP"


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
        "health_insurance_pct": 25.0, "improved_water_pct": 70.0,
        "stunted_pct": 30.0, "female_school_pct": 80.0, "survey_households": 700,
    }])
    gaps = scheme_gap_frame(profile)
    assert len(gaps) == 6
    assert set(gaps["survey_period"]) == {"NFHS-5, 2019-21"}
    assert gaps["gap_proxy_pct"].between(0, 100).all()
    assert gaps["scheme_key"].nunique() == 6
    plan = plan_budget(gaps, 10_000_000)
    assert len(plan) == 4  # only comparable service-access proxies enter budget shares
    assert plan["planner_eligible"].all()
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


def test_model_supports_auto_and_hist_gb_backend():
    from kshetra.model import available_backends, build_model, resolve_backend
    backends = available_backends()
    assert "hist_gb" in backends
    chosen = resolve_backend("auto")
    assert chosen in {"xgboost", "hist_gb"}
    m = build_model(backend="hist_gb")
    assert hasattr(m, "fit") and hasattr(m, "predict")


def test_image_change_features_computes_exg_and_heatmap():
    from kshetra.environment import compute_image_change_features
    before = np.zeros((20, 20, 3), dtype=np.uint8)
    before[:, :, 1] = 200  # high green
    after = np.zeros((20, 20, 3), dtype=np.uint8)
    after[:, :, 0] = 200   # high red (vegetation loss)
    res = compute_image_change_features(before, after)
    assert res["delta_exg_pct_points"] < -20
    assert res["changed_area_pct"] == pytest.approx(100.0)
    assert res["change_heatmap_rgb"].shape == (20, 20, 3)


def test_attach_approximate_coordinates_covers_states():
    from kshetra.geo import attach_approximate_coordinates
    df = pd.DataFrame([
        {"state_raw": "Bihar", "district_raw": "Araria", "state_key": "BIHAR", "district_key": "ARARIA"},
        {"state_raw": "Bihar", "district_raw": "Patna", "state_key": "BIHAR", "district_key": "PATNA"},
    ])
    out = attach_approximate_coordinates(df)
    assert out["approx_lat"].notna().all() and out["approx_lon"].notna().all()
    assert out.loc[0, "approx_lat"] != out.loc[1, "approx_lat"]


def test_pragati_tfidf_vector_search_and_roman_hindi():
    from kshetra.index import compute_index
    from kshetra.pragati import answer_query, vector_search_corpus
    base = _synthetic_frame(n=25, seed=11)
    base["state_raw"] = ["Bihar"] * 25
    base["district_raw"] = [f"District {i}" for i in range(25)]
    base["state_key"] = ["BIHAR"] * 25
    base["district_key"] = [f"DISTRICT {i}" for i in range(25)]
    base["mpi_hcr_pct"] = np.nan
    profiles = compute_index(base, n_clusters=3)
    hits = vector_search_corpus("clean cooking fuel PMUY sanitation", profiles, top_k=4)
    assert hits and all("similarity" in h for h in hits)
    hi_reply = answer_query("District 0 kyon flagged hai?", profiles, district="District 0", state="Bihar")
    assert "District 0" in hi_reply["answer"]
    assert hi_reply["intent"] == "drivers"


def test_counterfactual_simulation_improves_composite_and_percentile():
    from kshetra.index import compute_index
    from kshetra.schemes import simulate_counterfactual_profile
    base = _synthetic_frame(n=40, seed=21)
    base["state_raw"] = ["Maharashtra"] * 40
    base["district_raw"] = [f"District {i}" for i in range(40)]
    base["state_key"] = ["MAHARASHTRA"] * 40
    base["district_key"] = [f"DISTRICT {i}" for i in range(40)]
    base.loc[0, ["clean_cooking_fuel_pct", "improved_sanitation_pct", "improved_water_pct"]] = 15.0
    profiles = compute_index(base, n_clusters=3)
    sim = simulate_counterfactual_profile(profiles.loc[0], profiles, gap_closure_pct=50.0)
    assert sim["simulated_composite"] < sim["baseline_composite"]
    assert sim["simulated_percentile"] <= sim["baseline_percentile"]
    assert sim["composite_delta"] < 0


def test_optional_llm_fallback_returns_deterministic_template():
    from kshetra.pragati import synthesize_with_optional_llm
    evidence = [{"source": "NFHS-5", "district": "Araria / Bihar", "fact": "percentile=99.0", "similarity": 1.0}]
    ans, engine = synthesize_with_optional_llm("Why is Araria flagged?", evidence, "Template answer", api_key="")
    assert ans == "Template answer"
    assert engine == "deterministic_template"


def test_maharastra_spelling_normalizes_to_maharashtra_and_gets_coords():
    from kshetra.geo import attach_approximate_coordinates
    from kshetra.text import norm_state
    assert norm_state("Maharastra") == "MAHARASHTRA"
    df = pd.DataFrame([{"state_raw": "Maharastra", "district_raw": "Pune"}])
    out = attach_approximate_coordinates(df)
    assert out["approx_lat"].notna().all() and out["approx_lon"].notna().all()


def test_geo_coordinates_accept_nondefault_and_duplicate_index():
    from kshetra.geo import attach_approximate_coordinates
    df = pd.DataFrame([
        {"state_raw": "Bihar", "district_raw": "Araria"},
        {"state_raw": "Bihar", "district_raw": "Patna"},
    ], index=[101, 101])
    out = attach_approximate_coordinates(df)
    assert out["approx_lat"].notna().all()
    assert out["approx_lon"].notna().all()


def test_budget_planner_parses_string_booleans_safely():
    from kshetra.schemes import plan_budget
    gaps = pd.DataFrame({"planner_eligible": ["True", "False", "false", "1"],
                         "gap_proxy_pct": [40, 30, 20, 10]})
    plan = plan_budget(gaps, 1000)
    assert len(plan) == 2
    assert set(plan["gap_proxy_pct"]) == {40, 10}
    assert plan["allocation_inr"].sum() == pytest.approx(1000)


def test_image_change_features_accept_grayscale_and_rgba():
    from kshetra.environment import compute_image_change_features
    gray = np.full((6, 8), 128, dtype=np.uint8)
    rgba = np.dstack([np.full((6, 8), 128, dtype=np.uint8)] * 3 + [np.zeros((6, 8), dtype=np.uint8)])
    result = compute_image_change_features(gray, rgba)
    assert result["change_heatmap_rgb"].shape == (6, 8, 3)


def test_pragati_reuses_cached_vectorizer_for_same_corpus():
    from kshetra.index import compute_index
    from kshetra.pragati import vector_search_corpus, _search_index
    base = _synthetic_frame(n=10, seed=41)
    base["state_raw"] = "Bihar"
    base["district_raw"] = [f"District {i}" for i in range(10)]
    base["state_key"] = "BIHAR"
    base["district_key"] = [f"DISTRICT {i}" for i in range(10)]
    profiles = compute_index(base, n_clusters=2)
    _search_index.cache_clear()
    vector_search_corpus("fuel", profiles)
    first = _search_index.cache_info()
    vector_search_corpus("sanitation", profiles)
    second = _search_index.cache_info()
    assert second.hits == first.hits + 1


def test_district_name_matching_obeys_word_boundaries():
    from kshetra.pragati import _district_row
    profiles = pd.DataFrame([{"district_key": "GOA", "state_key": "GOA", "district_raw": "Goa", "state_raw": "Goa State"}])
    assert _district_row(profiles, None, "Is Goa flagged?") is not None
    assert _district_row(profiles, None, "Is this a goat district?") is None


def test_scst_source_label_is_case_insensitive_and_warns_unknown(tmp_path):
    from kshetra.ingest import load_census_scst
    raw = pd.DataFrame({"State_NFHS5": ["Bihar", "Bihar"], "District_NFHS5": ["A", "B"],
                        "Source": ["published", "district"], "SCST_Combined_Pct": ["12", "15"]})
    path = tmp_path / "scst.csv"
    raw.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="unknown source labels"):
        out = load_census_scst(path)
    assert out["scst_published"].tolist() == [True, False]
    assert pd.isna(out.loc[1, "census_scst_pct"])


def test_evaluate_skips_oof_predictions_unless_requested():
    from kshetra.index import compute_index
    from kshetra.model import evaluate
    base = _synthetic_frame(n=20, seed=54)
    base["mpi_hcr_pct"] = np.linspace(1, 60, 20)
    prof = compute_index(base, n_clusters=2)
    metrics = evaluate(prof, n_splits=2, backend="hist_gb")
    assert "_oof_predictions" not in metrics
    metrics_oof = evaluate(prof, n_splits=2, backend="hist_gb", include_oof=True)
    assert len(metrics_oof["_oof_predictions"]) == 20


def test_stac_query_returns_metadata_and_rejects_bad_bbox():
    import importlib.util
    from unittest.mock import patch
    spec = importlib.util.spec_from_file_location("fetch_satellite_stac", ROOT / "scripts" / "fetch_satellite_stac.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {"features": [{"id": "scene-1", "properties": {"datetime": "2025-07-01T00:00:00Z", "eo:cloud_cover": 4.0, "platform": "sentinel-2a"}}]}
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, json): return Response()
    with patch.object(module.httpx, "Client", Client):
        rows = module.query_stac_scenes([87.1, 25.9, 87.7, 26.5], "2025-07-01/2025-07-31")
    assert rows[0]["id"] == "scene-1"
    assert rows[0]["cloud_cover"] == 4.0
    with pytest.raises(ValueError, match="min coordinates"):
        module.query_stac_scenes([88, 25, 87, 26], "2025-07-01/2025-07-31")


# ---------- v7: STAC bridge, GeoJSON audit, embeddings, transition baseline, MIS ----------

def test_zonal_mean_respects_mask_and_missing_threshold():
    from kshetra.zonal import zonal_mean
    values = np.array([[1.0, 2.0], [np.nan, np.nan]])
    mask = np.ones((2, 2), dtype=bool)
    assert zonal_mean(values, mask, min_valid_fraction=0.5) == pytest.approx(1.5)
    assert math.isnan(zonal_mean(values, mask, min_valid_fraction=0.9))
    with pytest.raises(ValueError):
        zonal_mean(values, np.zeros((2, 2), dtype=bool))


def test_ndvi_and_percent_change_signs():
    from kshetra.zonal import ndvi, pct_change, rainfall_deficit_pct
    assert ndvi([0.6], [0.2])[0] == pytest.approx(0.5)
    assert math.isnan(ndvi([0.0], [0.0])[0])
    assert pct_change(0.3, 0.5) == pytest.approx(-40.0)   # vegetation decline is negative
    assert rainfall_deficit_pct(60, 100) == pytest.approx(40.0)  # positive = below normal
    assert math.isnan(rainfall_deficit_pct(60, 0))


def test_zonal_bridge_blanks_ndvi_for_cloudy_scenes_and_keeps_schema():
    from kshetra.zonal import zonal_table_to_signals
    table = pd.DataFrame([
        {"state": "Bihar", "district": "Araria", "observed_at": "2025-07-31", "source": "CHIRPS + S2 zonal means",
         "scene_ids": "S2A_1;S2B_2", "ndvi_current": 0.30, "ndvi_baseline": 0.50,
         "viirs_current": 9.0, "viirs_baseline": 10.0, "rain_observed_mm": 60, "rain_normal_mm": 100, "fire_count": 2},
        {"state": "Bihar", "district": "Patna", "observed_at": "2025-07-31", "source": "CHIRPS + S2 zonal means",
         "scene_ids": "S2A_3", "ndvi_current": 0.30, "ndvi_baseline": 0.50,
         "viirs_current": 9.0, "viirs_baseline": 10.0, "rain_observed_mm": 60, "rain_normal_mm": 100, "fire_count": None},
    ])
    scenes = {"S2A_1": 5.0, "S2B_2": 35.0, "S2A_3": 10.0}
    signals, rejected = zonal_table_to_signals(table, scenes, max_cloud_cover=20)
    assert rejected.empty
    araria = signals[signals["district"] == "Araria"].iloc[0]
    patna = signals[signals["district"] == "Patna"].iloc[0]
    assert math.isnan(araria["ndvi_change_pct"])      # one scene at 35% cloud blanks NDVI
    assert patna["ndvi_change_pct"] == pytest.approx(-40.0)
    assert araria["rainfall_deficit_pct"] == pytest.approx(40.0)
    assert "S2A_1" in patna["source"] or "S2A_3" in patna["source"]
    assert list(signals.columns) == ["state", "district", "observed_at", "source", "rainfall_deficit_pct",
                                     "ndvi_change_pct", "viirs_change_pct", "fire_count"]


def test_zonal_bridge_rejects_rows_without_any_usable_signal():
    from kshetra.zonal import zonal_table_to_signals
    table = pd.DataFrame([{
        "state": "Bihar", "district": "Nowhere", "observed_at": "2025-07-31", "source": "x", "scene_ids": "missing",
        "ndvi_current": 0.3, "ndvi_baseline": 0.5, "viirs_current": np.nan, "viirs_baseline": np.nan,
        "rain_observed_mm": np.nan, "rain_normal_mm": np.nan,
    }])
    signals, rejected = zonal_table_to_signals(table, {})
    assert signals.empty
    assert rejected.loc[0, "reason"] == "no usable numeric anomaly"


def test_bridge_output_passes_environment_validator():
    from kshetra.environment import score_environmental_risk
    from kshetra.zonal import zonal_table_to_signals
    table = pd.DataFrame([{
        "state": "Bihar", "district": "Araria", "observed_at": "2025-07-31", "source": "CHIRPS",
        "scene_ids": "S2A_1", "ndvi_current": 0.30, "ndvi_baseline": 0.50, "viirs_current": 9.0,
        "viirs_baseline": 10.0, "rain_observed_mm": 60, "rain_normal_mm": 100,
    }])
    signals, _ = zonal_table_to_signals(table, {"S2A_1": 4.0})
    scored = score_environmental_risk(signals)
    assert scored.loc[0, "environmental_score"] > 0


def test_geojson_audit_reports_size_coverage_and_unmatched_without_repair():
    from kshetra.geo import audit_district_geojson
    profiles = pd.DataFrame([
        {"state_raw": "Bihar", "district_raw": "Araria", "state_key": "BIHAR", "district_key": "ARARIA"},
        {"state_raw": "Bihar", "district_raw": "Patna", "state_key": "BIHAR", "district_key": "PATNA"},
    ])
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"ST_NM": "Bihar", "DISTRICT": "Araria"}, "geometry": None},
        {"type": "Feature", "properties": {"ST_NM": "Bihar", "DISTRICT": "Unknown"}, "geometry": None},
    ]}
    report = audit_district_geojson(geo, profiles, "DISTRICT", "ST_NM", file_bytes=2_000_000, max_mb=3.0)
    assert report["matched_profile_districts"] == 1
    assert report["profile_districts_without_polygon"] == 1
    assert report["unmatched_feature_count"] == 1
    assert report["within_size_limit"] is True
    too_big = audit_district_geojson(geo, profiles, "DISTRICT", "ST_NM", file_bytes=5_000_000, max_mb=3.0)
    assert too_big["within_size_limit"] is False


def test_embedding_difference_uses_injected_extractor_and_checks_inputs():
    from kshetra.embeddings import embedding_difference
    def mean_colour(rgb):
        return rgb.reshape(-1, 3).mean(axis=0)
    before = np.full((4, 4, 3), 100, dtype=np.uint8)
    after = np.full((4, 4, 3), 100, dtype=np.uint8)
    after[:, :, 0] = 200
    res = embedding_difference(before, after, mean_colour)
    assert res["embedding_dim"] == 3
    assert res["l2_distance"] > 0
    assert res["cosine_distance"] >= 0
    same = embedding_difference(before, before, mean_colour)
    assert same["l2_distance"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        embedding_difference(before, after, lambda rgb: np.ones(3) if rgb[0, 0, 0] == 100 else np.ones(2))


def test_torchvision_extractor_fails_clearly_without_torch(tmp_path):
    from kshetra.embeddings import torchvision_extractor
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; the missing-dependency path is not reachable")
    with pytest.raises(RuntimeError, match="PyTorch"):
        torchvision_extractor(tmp_path / "missing.pt")


def _monthly_series(pattern: list[float], district: str = "ARARIA", years: int = 4) -> pd.DataFrame:
    months = pd.period_range("2019-01", periods=12 * years, freq="M")
    values = [pattern[i % 12] for i in range(len(months))]
    return pd.DataFrame({"district_key": district, "month": months.to_timestamp(), "value": values})


def test_transition_baseline_forecasts_from_latest_state_and_flags_short_history():
    from kshetra.forecast import drought_transition_baseline
    rng = np.random.default_rng(0)
    # Persistent deficits: a deficit month is followed by another deficit most of the time.
    rows = []
    for year in range(4):
        for month in range(12):
            base = 100.0
            dry = (year % 2 == 0 and month in (5, 6, 7, 8))
            rows.append({"district_key": "ARARIA", "month": pd.Timestamp(2019 + year, month + 1, 1),
                         "value": (50.0 if dry else base) + rng.normal(0, 1)})
    data = pd.DataFrame(rows)
    res = drought_transition_baseline(data, deficit_pct=25.0, min_months=24)
    row = res.iloc[0]
    assert row["sufficient_history"]
    assert 0 <= row["forecast_deficit_prob"] <= 1
    assert np.isfinite(row["p_deficit_given_normal"])
    short = drought_transition_baseline(data.iloc[:6], deficit_pct=25.0, min_months=24).iloc[0]
    assert not short["sufficient_history"]
    assert math.isnan(short["forecast_deficit_prob"])


def test_transition_baseline_skips_gaps_between_months():
    from kshetra.forecast import drought_transition_baseline
    data = _monthly_series([100.0] * 12, years=3)
    gapped = data.drop(index=[5, 6, 7, 8, 9, 10])  # remove a six-month gap
    res = drought_transition_baseline(gapped, min_months=10)
    assert res.iloc[0]["months_observed"] == len(gapped)


def test_mis_loader_requires_provenance_and_comparable_schemes():
    from kshetra.mis import load_mis_table
    base = {"state": "Bihar", "district": "Araria", "scheme": "pmuy", "period": "2025-03-31",
            "enrolled": 1000, "denominator": 5000, "denominator_type": "households (census 2011)", "source": "MIS export 2025-03"}
    load_mis_table(pd.DataFrame([base]))
    with pytest.raises(ValueError, match="source"):
        load_mis_table(pd.DataFrame([{**base, "source": " "}]))
    with pytest.raises(ValueError, match="not comparable"):
        load_mis_table(pd.DataFrame([{**base, "scheme": "poshan"}]))
    with pytest.raises(ValueError, match="non-negative"):
        load_mis_table(pd.DataFrame([{**base, "enrolled": -5}]))


def test_mis_vs_survey_reports_difference_and_blocks_zero_denominator():
    from kshetra.mis import load_mis_table, mis_vs_survey
    profiles = pd.DataFrame([{"state_raw": "Bihar", "district_raw": "Araria", "state_key": "BIHAR",
                              "district_key": "ARARIA", "clean_cooking_fuel_pct": 40.0}])
    rows = pd.DataFrame([
        {"state": "Bihar", "district": "Araria", "scheme": "pmuy", "period": "2025-03-31", "enrolled": 3000,
         "denominator": 5000, "denominator_type": "households", "source": "MIS export"},
        {"state": "Bihar", "district": "Araria", "scheme": "pmuy", "period": "2025-06-30", "enrolled": 10,
         "denominator": 0, "denominator_type": "none", "source": "MIS export"},
    ])
    out = mis_vs_survey(load_mis_table(rows), profiles)
    ok = out[out["period"] == "2025-03-31"].iloc[0]
    assert ok["mis_coverage_pct"] == pytest.approx(60.0)
    assert ok["proxy_minus_mis_points"] == pytest.approx(-20.0)
    blocked = out[out["period"] == "2025-06-30"].iloc[0]
    assert math.isnan(blocked["mis_coverage_pct"])


def test_zero_climatology_zero_rain_is_zero_anomaly_not_missing():
    import pandas as pd
    from kshetra.forecast import monthly_anomalies
    data = pd.DataFrame({
        "district_key": ["d", "d", "d"],
        "month": ["2020-01", "2021-01", "2020-06"],
        "value": [0.0, 0.0, 10.0],
    })
    out = monthly_anomalies(data)
    jan = out[out["_cal_month"] == 1]
    assert (jan["anomaly_pct"] == 0.0).all(), jan
    assert out["anomaly_pct"].notna().all(), out


def test_mis_vs_survey_matches_row_by_row_and_flags_duplicate_profiles():
    import numpy as np
    import pandas as pd
    from kshetra.mis import mis_vs_survey
    from kshetra.schemes import SCHEME_PROXIES
    scheme = next(s for s in SCHEME_PROXIES if s.planner_eligible)
    profiles = pd.DataFrame({
        "state_key": ["BIHAR", "BIHAR", "BIHAR"],
        "district_key": ["ARARIA", "PATNA", "PATNA"],
        "state_raw": ["Bihar"] * 3, "district_raw": ["Araria", "Patna", "Patna"],
        scheme.nfhs_indicator: [40.0, 55.0, 60.0],
    })
    mis = pd.DataFrame({
        "state": ["Bihar", "Bihar"], "district": ["Araria", "Patna"], "scheme": [scheme.key, scheme.key],
        "period": ["2025-03", "2025-03"], "enrolled": [30, 50], "denominator": [100, 0],
        "denominator_type": ["households", "households"], "source": ["test", "test"],
    })
    out = mis_vs_survey(mis, profiles)
    araria = out[out["district"] == "Araria"].iloc[0]
    patna = out[out["district"] == "Patna"].iloc[0]
    assert araria["profile_match"] and araria["nfhs_proxy_pct"] == 40.0
    assert abs(araria["mis_coverage_pct"] - 30.0) < 1e-9
    assert abs(araria["proxy_minus_mis_points"] - 10.0) < 1e-9
    assert not patna["profile_match"] and np.isnan(patna["nfhs_proxy_pct"])
    assert np.isnan(patna["mis_coverage_pct"])


def test_torchvision_loader_uses_weights_only_and_optional_path():
    import inspect
    from kshetra import embeddings
    src = inspect.getsource(embeddings.torchvision_extractor)
    assert "weights_only=True" in src
    assert "weights_path: Path | str | None = None" in src
