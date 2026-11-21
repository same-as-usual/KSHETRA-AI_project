"""KSHETRA-AI six-module dashboard.

Run from the repository root with: streamlit run app/streamlit_app.py
The app only reports locally supplied survey/environment data; it does not fabricate satellite observations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kshetra.config import DOMAIN_LABELS, INDICATOR_LABELS, INDICATORS  # noqa: E402
import numpy as np
from kshetra.environment import attach_profile_scores, compute_image_change_features, validate_environmental_signals  # noqa: E402
from kshetra.geo import attach_approximate_coordinates, match_geojson_to_profiles  # noqa: E402
from kshetra.pragati import answer_query  # noqa: E402
from kshetra.schemes import SCHEME_PROXIES, plan_budget, scheme_gap_frame  # noqa: E402
from kshetra.verification import STATUSES, add_verification, list_verifications  # noqa: E402

PROFILES = ROOT / "outputs" / "district_profiles.csv"
METRICS = ROOT / "outputs" / "model_metrics.json"
BOUNDARIES = ROOT / "data" / "geo" / "india_districts.geojson"

st.set_page_config(page_title="KSHETRA-AI | District intelligence", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {border-left: 3px solid #d88a38; padding-left: .75rem;}
.small-muted {color: #667085; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_profiles(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def district_options(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return [(f"{r.district_raw} — {r.state_raw}", pos) for pos, r in enumerate(frame.itertuples())]


def district_picker(frame: pd.DataFrame, label: str, key: str, default_index: int = 0) -> pd.Series:
    options = district_options(frame)
    selected = st.selectbox(label, options, index=min(max(default_index, 0), len(options) - 1),
                            format_func=lambda pair: pair[0], key=key)
    return frame.iloc[selected[1]]


def risk_colour(value: str) -> str:
    return {"Very high": "#ad3f24", "High": "#d77b32", "Moderate": "#e4bb54", "Lower": "#468a72"}.get(value, "#777777")


def money_inr(value: float) -> str:
    if value >= 1e7:
        return f"₹{value / 1e7:,.2f} crore"
    if value >= 1e5:
        return f"₹{value / 1e5:,.2f} lakh"
    return f"₹{value:,.0f}"


st.title("KSHETRA-AI")
st.caption("From deprivation data to localized action · district-level screening, explainable drivers, and human verification")
st.warning("A risk signal is not proof of poverty or individual eligibility. Verify locally before any intervention.", icon="⚠️")

if not PROFILES.exists():
    st.error("District profiles are missing. Place the three CSV files in data/raw/ and run `python -m kshetra.pipeline`.")
    st.stop()

profiles = load_profiles(str(PROFILES))
metrics = load_metrics(str(METRICS)) if METRICS.exists() else {"model": {}}

with st.sidebar:
    st.markdown("## KSHETRA-AI")
    st.caption("Evidence-led district screening")
    states = sorted(profiles["state_raw"].dropna().unique())
    state_filter = st.multiselect("State / UT", states, default=[], key="global_state_filter")
    tier_order = ["Very high", "High", "Moderate", "Lower"]
    tier_filter = st.multiselect("Risk tier", tier_order, default=tier_order, key="global_tier_filter")
    filtered = profiles[profiles["risk_tier"].isin(tier_filter)]
    if state_filter:
        filtered = filtered[filtered["state_raw"].isin(state_filter)]
    st.divider()
    st.caption(f"Data: NFHS-5 2019–21 · {len(profiles):,} districts")
    st.caption("Satellite inputs are optional and are not included by default.")

if filtered.empty:
    st.info("No districts match the selected filters. Adjust the sidebar filters.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Districts in view", f"{len(filtered):,}")
m2.metric("Very high screening tier", f"{(filtered.risk_tier == 'Very high').sum():,}")
m3.metric("Census-2011 MPI linked", f"{filtered.mpi_hcr_pct.notna().sum():,}")
m4.metric("Cross-check CV R²", f"{metrics.get('model', {}).get('cv_r2', 'n/a')}")

# Six modules exactly match the project presentation.
tabs = st.tabs([
    "1 · Poverty Intelligence Atlas",
    "2 · Risk & Early Warning",
    "3 · Explainable AI Insights",
    "4 · Action Planner",
    "5 · SchemeGap Intelligence",
    "6 · PRAGATI AI Assistant",
])

# MODULE 1 — Atlas
with tabs[0]:
    st.subheader("District deprivation atlas")
    left, right = st.columns([1.35, 1])
    with left:
        upload = st.file_uploader("Upload district boundaries (GeoJSON FeatureCollection)", type=["geojson", "json"], key="atlas_geojson")
        geojson = None
        if upload:
            try:
                import json as _json
                geojson = _json.loads(upload.getvalue().decode("utf-8"))
            except Exception as exc:
                st.error(f"Could not read the GeoJSON: {exc}")
        elif BOUNDARIES.exists():
            try:
                geojson = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
            except Exception as exc:
                st.error(f"Could not read data/geo/india_districts.geojson: {exc}")

        if geojson:
            feats = geojson.get("features", [])
            props = list((feats[0].get("properties") or {}).keys()) if feats else []
            if not props:
                st.warning("No feature properties found in the boundary file.")
            else:
                state_guess = next((p for p in props if p.lower() in {"st_nm", "state", "state_name", "state_ut"}), props[0])
                district_guess = next((p for p in props if p.lower() in {"district", "dtname", "district_name", "district_nm"}), props[-1])
                s_col, d_col = st.columns(2)
                state_prop = s_col.selectbox("State name property", props, index=props.index(state_guess), key="geo_state_prop")
                district_prop = d_col.selectbox("District name property", props, index=props.index(district_guess), key="geo_district_prop")
                try:
                    map_geo, map_rows = match_geojson_to_profiles(geojson, filtered, district_prop, state_prop)
                    if map_rows.empty:
                        st.warning("No exact state/district name matches. Check the property fields or boundary vintage; unmatched features are not guessed.")
                    else:
                        fig_map = px.choropleth_mapbox(
                            map_rows, geojson=map_geo, locations="_kshetra_key",
                            featureidkey="properties._kshetra_key", color="composite_percentile",
                            hover_name="district_raw",
                            hover_data={"state_raw": True, "risk_tier": True, "primary_domain": True,
                                        "confidence": True, "composite_percentile": ":.1f", "_kshetra_key": False},
                            color_continuous_scale=[[0, "#4d8b73"], [0.5, "#e4bb54"], [1, "#ad3f24"]],
                            range_color=(0, 100), mapbox_style="open-street-map", zoom=3.25,
                            center={"lat": 22.8, "lon": 79.0}, opacity=0.72,
                            labels={"composite_percentile": "Deprivation percentile", "state_raw": "State"},
                            title="District screening percentile · survey data, not live poverty",
                        )
                        fig_map.update_layout(height=600, margin={"r": 0, "t": 45, "l": 0, "b": 0})
                        st.plotly_chart(fig_map, use_container_width=True)
                        st.caption(f"Exact boundary matches: {len(map_rows)} of {len(filtered)} filtered districts. Boundary attribution and limitations are in data/geo/README.md.")
                except Exception as exc:
                    st.error(f"Could not join GeoJSON to district profiles: {exc}")
        else:
            st.caption("No district boundary GeoJSON is loaded; displaying a state-anchored spatial bubble map. Upload a GeoJSON FeatureCollection above for polygon choropleth boundaries.")
            spatial_df = attach_approximate_coordinates(filtered).dropna(subset=["approx_lat", "approx_lon"]).copy()
            spatial_df["bubble_size"] = spatial_df["composite_percentile"].clip(lower=15.0)
            fig_bubble = px.scatter_geo(
                spatial_df,
                lat="approx_lat",
                lon="approx_lon",
                color="composite_percentile",
                size="bubble_size",
                size_max=11,
                hover_name="district_raw",
                hover_data={
                    "state_raw": True, "risk_tier": True, "primary_domain": True,
                    "confidence": True, "composite_percentile": ":.1f",
                    "approx_lat": False, "approx_lon": False, "bubble_size": False,
                },
                color_continuous_scale=[[0, "#4d8b73"], [0.5, "#e4bb54"], [1, "#ad3f24"]],
                range_color=(0, 100),
                labels={"composite_percentile": "Deprivation percentile", "state_raw": "State"},
                title="District deprivation screening · state-anchored spatial view (upload GeoJSON for polygons)",
            )
            fig_bubble.update_geos(
                visible=True, resolution=50, scope="asia",
                center={"lat": 22.8, "lon": 80.0}, projection_scale=4.2,
                showcountries=True, countrycolor="#cbd5e1",
                showland=True, landcolor="#f8f6f0",
                showocean=True, oceancolor="#eef4f7",
                lataxis_range=[6.5, 37.5], lonaxis_range=[67.5, 98.0],
            )
            fig_bubble.update_layout(height=600, margin={"r": 0, "t": 45, "l": 0, "b": 0})
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.caption("Markers are deterministically arranged around state/UT reference centroids so all 707 districts remain inspectable without bundling unverified boundary polygons.")

    with right:
        state_means = filtered.groupby("state_raw", as_index=False).agg(
            mean_percentile=("composite_percentile", "mean"), districts=("district_raw", "count"))
        state_means = state_means.sort_values("mean_percentile", ascending=True).tail(15)
        fig_state = px.bar(state_means, x="mean_percentile", y="state_raw", orientation="h",
                           hover_data={"districts": True},
                           labels={"mean_percentile": "Mean district percentile", "state_raw": ""},
                           title="Highest state/UT means in current view",
                           color="mean_percentile", color_continuous_scale=["#e4bb54", "#ad3f24"])
        fig_state.update_layout(height=600, coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_state, use_container_width=True)

    ranked = filtered.sort_values("composite_percentile", ascending=False)[
        ["state_raw", "district_raw", "composite_percentile", "risk_tier", "primary_domain", "top_drivers", "confidence", "mpi_hcr_pct"]
    ].rename(columns={"state_raw": "State", "district_raw": "District", "composite_percentile": "Percentile",
                      "risk_tier": "Tier", "primary_domain": "Main domain", "top_drivers": "Top drivers",
                      "confidence": "Confidence", "mpi_hcr_pct": "Census-2011 MPI HCR (%)"})
    st.dataframe(ranked.round(1), use_container_width=True, hide_index=True)

# MODULE 2 — Risk and Early Warning
with tabs[1]:
    st.subheader("Risk & Early Warning")
    st.markdown("Upload **observed, district-aggregated** rainfall / vegetation / VIIRS / fire data. The prototype does not contain synthetic alerts or fetch live rasters.")
    env_template = pd.DataFrame(columns=["state", "district", "observed_at", "source", "rainfall_deficit_pct", "ndvi_change_pct", "viirs_change_pct", "fire_count"])
    st.download_button("Download empty CSV template", env_template.to_csv(index=False).encode("utf-8"),
                       file_name="environment_signals_template.csv", mime="text/csv")
    env_upload = st.file_uploader("Upload sourced environmental signals CSV", type=["csv"], key="env_csv")
    if env_upload:
        try:
            raw_env = pd.read_csv(env_upload)
            env = attach_profile_scores(raw_env, profiles)
            st.caption("Screening thresholds are transparent rule-based scales, not trained probabilities. Fire detections are shown separately and do not enter the score.")
            e1, e2, e3 = st.columns(3)
            e1.metric("Observation rows", f"{len(env):,}")
            e2.metric("Rows requiring verification", f"{env.requires_field_verification.sum():,}")
            e3.metric("Exact district profile matches", f"{env.profile_match.sum():,}")
            env["district_label"] = env["district"] + " — " + env["state"]
            env_options = sorted(env["district_label"].unique())
            selected_env = st.selectbox("Trend district", env_options, key="env_district")
            tr = env[env["district_label"] == selected_env].sort_values("observed_at")
            signal_cols = [c for c in ["rainfall_deficit_pct", "ndvi_change_pct", "viirs_change_pct"] if c in tr and tr[c].notna().any()]
            if signal_cols:
                long = tr.melt(id_vars=["observed_at"], value_vars=signal_cols, var_name="signal", value_name="value").dropna()
                fig = px.line(long, x="observed_at", y="value", color="signal", markers=True,
                              title="Observed environmental anomalies over time",
                              labels={"observed_at": "Observation date", "value": "Source value (%)", "signal": "Signal"})
                fig.update_layout(height=380, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            latest = tr.sort_values("observed_at").iloc[-1]
            radar_fields = [
                ("Rainfall stress", "score_rainfall"),
                ("Vegetation decline", "score_vegetation"),
                ("Night-light decline", "score_night_lights"),
            ]
            radar_labels = [label for label, field in radar_fields if field in latest.index and pd.notna(latest[field])]
            radar_values = [float(latest[field]) for label, field in radar_fields if field in latest.index and pd.notna(latest[field])]
            if len(radar_values) >= 3:
                radar = go.Figure(go.Scatterpolar(
                    r=radar_values + [radar_values[0]], theta=radar_labels + [radar_labels[0]],
                    fill="toself", line={"color": "#ad3f24", "width": 2},
                    fillcolor="rgba(173,63,36,0.18)", name="Normalized stress score",
                ))
                radar.update_layout(
                    title=f"Latest environmental stress components · {latest['observed_at'].date()}",
                    polar={"radialaxis": {"visible": True, "range": [0, 100], "ticksuffix": "%"}},
                    showlegend=False, height=380, margin={"l": 30, "r": 30, "t": 55, "b": 20},
                )
                st.plotly_chart(radar, use_container_width=True)
                st.caption("Radar axes are transparent rescalings of rainfall deficit (40% scale), vegetation decline (30%), and VIIRS decline (30%). Fire detections remain separate.")
            elif radar_values:
                radar_bar = pd.DataFrame({"Signal": radar_labels, "Normalized stress score": radar_values})
                st.plotly_chart(px.bar(radar_bar, x="Normalized stress score", y="Signal", orientation="h",
                                       range_x=[0, 100], title="Available environmental stress components"), use_container_width=True)
                st.caption("Risk radar requires all three anomaly baselines; only available metrics are shown here.")
            st.dataframe(env.sort_values(["observed_at", "environmental_score"], ascending=[False, False]),
                         use_container_width=True, hide_index=True)
            st.caption("Scores identify environmental stress only. They are not poverty labels and do not trigger automatic action.")
        except Exception as exc:
            st.error(f"Environmental input rejected: {exc}")
    else:
        st.info("No environmental observation file supplied. There are no live or simulated alerts to display.")
        st.markdown("Use CHIRPS for rainfall, VIIRS Black Marble for nighttime-light change, Sentinel-2 for vegetation/land change, and NASA FIRMS for active-fire observations. Document district aggregation, baseline, quality filtering, source and observation dates; see data/environmental/README.md.")

    with st.expander("Optional Sentinel-2 before / after image comparison"):
        st.caption("Upload actual, quality-screened composites and enter their source/date in the captions. This visual does not perform change detection by itself.")
        img_left, img_right = st.columns(2)
        before_file = img_left.file_uploader("Before image", type=["png", "jpg", "jpeg"], key="s2_before")
        after_file = img_right.file_uploader("After image", type=["png", "jpg", "jpeg"], key="s2_after")
        before_details, after_details = st.columns(2)
        before_label = before_details.text_input("Before date / source", placeholder="Date and product/source", key="s2_before_details")
        after_label = after_details.text_input("After date / source", placeholder="Date and product/source", key="s2_after_details")
        if before_file and after_file:
            try:
                from PIL import Image, ImageOps
                before_img = Image.open(before_file).convert("RGB")
                after_img = Image.open(after_file).convert("RGB")
                size = (min(before_img.width, after_img.width), min(before_img.height, after_img.height))
                before_img = ImageOps.fit(before_img, size)
                after_img = ImageOps.fit(after_img, size)
                blend = st.slider("After-image blend", min_value=0, max_value=100, value=50, key="s2_blend") / 100
                compare = Image.blend(before_img, after_img, blend)
                cv_stats = compute_image_change_features(np.asarray(before_img), np.asarray(after_img))
                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.metric("Greenness index ΔExG", f"{cv_stats['delta_exg_pct_points']:+.2f} pp")
                ic2.metric("Spectral index ΔNGRDI", f"{cv_stats['delta_ngrdi']:+.4f}")
                ic3.metric("Mean pixel intensity Δ", f"{cv_stats['mean_pixel_change_pct']:.1f}%")
                ic4.metric("Changed area fraction", f"{cv_stats['changed_area_pct']:.1f}%")
                x1, x2, x3, x4 = st.columns(4)
                x1.image(before_img, caption=f"Before · {before_label or before_file.name}", use_container_width=True)
                x2.image(after_img, caption=f"After · {after_label or after_file.name}", use_container_width=True)
                x3.image(compare, caption=f"Blend · after weight {blend:.0%}", use_container_width=True)
                x4.image(cv_stats["change_heatmap_rgb"], caption="Change heatmap (terracotta = greenness loss, teal = gain)", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not compare uploaded images: {exc}")

# MODULE 3 — Explainable AI
with tabs[2]:
    st.subheader("Explainable AI Insights")
    explain_default = int(filtered["composite_percentile"].to_numpy().argmax())
    row = district_picker(filtered, "District profile", "explain_district", explain_default)
    a, b, c, d = st.columns(4)
    a.metric("Composite percentile", f"{row['composite_percentile']:.0f}")
    b.metric("Screening tier", str(row["risk_tier"]), delta_color="off")
    c.metric("Confidence", str(row["confidence"]), delta_color="off")
    d.metric("Missing indicators", int(row["indicators_missing"]))
    st.markdown(f"**District:** {row['district_raw']}, {row['state_raw']} · **Main domain:** {row['primary_domain']} · **NFHS-5 survey households:** {int(row['survey_households']) if pd.notna(row['survey_households']) else 'n/a'}")
    completeness = max(0.0, 100.0 * (len(INDICATORS) - int(row["indicators_missing"])) / len(INDICATORS))
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=completeness, number={"suffix": "%", "valueformat": ".0f"},
        title={"text": "Indicator completeness<br><span style='font-size:0.75em'>not predictive confidence</span>"},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": "#d88a38"},
               "steps": [{"range": [0, 70], "color": "#f2e5df"}, {"range": [70, 90], "color": "#efe7ce"},
                         {"range": [90, 100], "color": "#dcebe3"}],
               "threshold": {"line": {"color": "#344054", "width": 2}, "thickness": 0.75, "value": completeness}},
    ))
    gauge.update_layout(height=225, margin={"l": 30, "r": 30, "t": 55, "b": 15, "pad": 3})
    st.plotly_chart(gauge, use_container_width=True)
    st.caption(f"This gauge is the share of the {len(INDICATORS)} NFHS indicators available for this row. The High/Medium/Low badge also considers survey sample size.")
    contributions = pd.DataFrame({
        "Indicator": [INDICATOR_LABELS[k] for k in INDICATORS],
        "Contribution": [float(row[f"contrib_{k}"]) for k in INDICATORS],
        "Observed NFHS-5 value (%)": [row[k] for k in INDICATORS],
    }).sort_values("Contribution", ascending=True)
    fig_contrib = px.bar(contributions, x="Contribution", y="Indicator", orientation="h",
                         hover_data={"Observed NFHS-5 value (%)": ":.1f"},
                         color="Contribution", color_continuous_scale=["#4d8b73", "#f3efe4", "#ad3f24"],
                         title="Driver contribution to the district composite (positive = more deprivation)")
    fig_contrib.update_layout(height=510, coloraxis_showscale=False)
    st.plotly_chart(fig_contrib, use_container_width=True)
    domain_scores = pd.DataFrame({"Domain": [DOMAIN_LABELS[x] for x in DOMAIN_LABELS],
                                  "Standardized score": [row[f"score_{x}"] for x in DOMAIN_LABELS]})
    st.dataframe(domain_scores.round(2), use_container_width=True, hide_index=True)
    if pd.notna(row["mpi_hcr_pct"]):
        st.info(f"External cross-check only: supplied Census-2011 MPI candidate HCR is {row['mpi_hcr_pct']:.1f}% ({row['mpi_match']} name match). Provenance must be confirmed.")
    else:
        st.info(f"No linked Census-2011 MPI value. The model estimate is {row['proxy_mpi_hcr_pct']:.1f}% and is explicitly a prediction, not an observation.")

    st.divider()
    st.subheader("Human field-verification loop")
    st.caption("Log an aggregate district review only. Do not enter names, phone numbers, addresses, case details, or other personal data. Records stay in a local SQLite file on the app server.")
    with st.form("verification_form", clear_on_submit=True):
        v1, v2 = st.columns(2)
        status = v1.selectbox("Review outcome", [s for s in STATUSES if s != "Pending"])
        domain = v2.selectbox("Domain reviewed", list(DOMAIN_LABELS.values()) + ["Environmental signal", "Other"])
        note = st.text_area("Aggregate field note (max 2,000 characters; no personal data)", max_chars=2000)
        method = st.text_input("Verification method / source", placeholder="e.g. block-level administrative review")
        submitted = st.form_submit_button("Save field review")
        if submitted:
            try:
                add_verification(str(row["state_raw"]), str(row["district_raw"]), status, domain, note, method)
                st.success("Review saved to the local audit log.")
            except Exception as exc:
                st.error(f"Could not save review: {exc}")
    try:
        verifications = list_verifications()
        if not verifications.empty:
            st.dataframe(verifications, use_container_width=True, hide_index=True)
        else:
            st.caption("No field reviews have been recorded yet.")
    except Exception as exc:
        st.warning(f"Field log unavailable: {exc}")

# MODULE 4 — Action planner
with tabs[3]:
    st.subheader("Action Planner · budget-share scenario")
    st.warning("Scenario only: allocations are proportional to service-gap proxies. This is not a cost optimizer, beneficiary estimate, entitlement decision, or impact forecast.")
    planner_row = district_picker(filtered, "District for scenario", "planner_district")
    district_gaps = scheme_gap_frame(pd.DataFrame([planner_row]))
    budget_crore = st.number_input("Hypothetical district budget (₹ crore)", min_value=0.0, value=1.0, step=0.1, key="budget_crore")
    plan = plan_budget(district_gaps, budget_crore * 1e7)
    scenario_share = st.slider("Illustrative share of each service gap targeted", min_value=0, max_value=100, value=20, step=5, key="scenario_gap_share")
    if plan.empty:
        st.info("No service-access proxy values are available for this district.")
    else:
        plan_view = plan[["scheme", "coverage_proxy_pct", "gap_proxy_pct", "budget_share_pct", "allocation_inr", "interpretation"]].copy()
        plan_view["scenario_coverage_pct"] = (plan_view["coverage_proxy_pct"] + plan_view["gap_proxy_pct"] * scenario_share / 100).clip(upper=100)
        plan_view["allocation"] = plan_view["allocation_inr"].map(money_inr)
        fig_plan = px.bar(plan_view, x="allocation_inr", y="scheme", orientation="h", color="gap_proxy_pct",
                          hover_data={"budget_share_pct": ":.1f", "gap_proxy_pct": ":.1f", "coverage_proxy_pct": ":.1f", "allocation_inr": False},
                          labels={"allocation_inr": "Scenario allocation (₹)", "scheme": "Programme area", "gap_proxy_pct": "Gap proxy (%)"},
                          title="Scenario budget shares · weighted by service-access gap proxies",
                          color_continuous_scale=["#e4bb54", "#ad3f24"])
        fig_plan.update_layout(height=360)
        st.plotly_chart(fig_plan, use_container_width=True)
        before_after = pd.concat([
            plan_view[["scheme", "coverage_proxy_pct"]].rename(columns={"coverage_proxy_pct": "Proxy coverage (%)"}).assign(Scenario="Observed NFHS-5 proxy"),
            plan_view[["scheme", "scenario_coverage_pct"]].rename(columns={"scenario_coverage_pct": "Proxy coverage (%)"}).assign(Scenario=f"Illustrative target (+{scenario_share}% of gap)"),
        ], ignore_index=True)
        fig_scenario = px.bar(before_after, x="scheme", y="Proxy coverage (%)", color="Scenario", barmode="group",
                              range_y=[0, 100], title="Before / after planning scenario (assumed target, not a forecast)",
                              color_discrete_sequence=["#587f78", "#d88a38"])
        fig_scenario.update_layout(height=370, xaxis_tickangle=-18)
        st.plotly_chart(fig_scenario, use_container_width=True)
        st.dataframe(plan_view[["scheme", "coverage_proxy_pct", "gap_proxy_pct", "scenario_coverage_pct", "budget_share_pct", "allocation", "interpretation"]].round(1), use_container_width=True, hide_index=True)
        st.caption(f"The after value is a user-selected scenario target: observed proxy coverage plus {scenario_share}% of the observed gap. It is not a predicted or causal impact and is not linked to the entered budget. Budget shares cover service-access proxies only; no eligible-population denominator or unit cost was provided.")

# MODULE 5 — SchemeGap
with tabs[4]:
    st.subheader("SchemeGap Intelligence · service coverage proxies")
    st.markdown("This view compares district NFHS-5 service outcomes to highlight where administrative scheme data should be checked. It does **not** show scheme enrollment or household eligibility.")
    gaps_all = scheme_gap_frame(filtered)
    scheme_choice = st.selectbox("Programme proxy", [s.key for s in SCHEME_PROXIES],
                                 format_func=lambda key: next(s.name for s in SCHEME_PROXIES if s.key == key), key="scheme_choice")
    sg = gaps_all[gaps_all.scheme_key == scheme_choice].dropna(subset=["gap_proxy_pct"])
    if sg.empty:
        st.info("No proxy observations available for this selection.")
    else:
        definition = next(s for s in SCHEME_PROXIES if s.key == scheme_choice)
        c1, c2, c3 = st.columns(3)
        c1.metric("Districts with proxy", f"{len(sg):,}")
        if definition.proxy_kind == "Service-access proxy":
            c2.metric("Mean service-access proxy", f"{sg.coverage_proxy_pct.mean():.1f}%")
            c3.metric("Mean uncovered proxy", f"{sg.gap_proxy_pct.mean():.1f}%")
        else:
            c2.metric("Mean outcome proxy", f"{sg.outcome_proxy_pct.mean():.1f}%")
            c3.metric("Mean burden / access-gap proxy", f"{sg.gap_proxy_pct.mean():.1f}%")
        st.markdown(f"**Proxy type:** {definition.proxy_kind}")
        st.write(f"**Interpretation:** {definition.interpretation}")
        st.markdown(f"**Official scheme information:** [{definition.name}]({definition.url})")
        journey = st.columns(4)
        journey_steps = [
            ("1 · NFHS proxy", "Screen district-level access or outcome signal"),
            ("2 · Check MIS", "Compare against current programme records"),
            ("3 · Field verify", "Confirm the local context and data vintage"),
            ("4 · Authorized review", "Decide through the responsible agency; no automatic eligibility"),
        ]
        for card, (heading, detail) in zip(journey, journey_steps):
            card.markdown(f"**{heading}**" + "\n\n" + detail)
        gap_rank = sg.sort_values("gap_proxy_pct", ascending=False).head(30)
        fig_gap = px.bar(gap_rank, x="gap_proxy_pct", y="district_raw", color="state_raw", orientation="h",
                         hover_data={"proxy_measure_pct": ":.1f", "survey_households": True},
                         labels={"gap_proxy_pct": definition.gap_label + " (%)", "district_raw": "District", "state_raw": "State"},
                         title=f"Highest {definition.gap_label.lower()} values in filtered districts")
        fig_gap.update_layout(height=680, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_gap, use_container_width=True)
        st.dataframe(sg.sort_values("gap_proxy_pct", ascending=False)[
            ["state_raw", "district_raw", "proxy_kind", "proxy_measure_label", "proxy_measure_pct", "gap_label", "gap_proxy_pct", "survey_households", "survey_period", "interpretation"]
        ].round(1), use_container_width=True, hide_index=True)
    st.caption("NFHS-5 reference period is 2019–21. Proxies do not establish scheme coverage or eligibility. Update with current administrative data before operational planning.")

# MODULE 6 — PRAGATI Assistant
with tabs[5]:
    st.subheader("PRAGATI · evidence-grounded district assistant (TF-IDF RAG)")
    st.caption("TF-IDF vector retrieval over the 707 district profiles and 6 programme-proxy definitions paired with grounded English/Hindi synthesis templates. No external API key or hallucinated claims.")
    rag_row = district_picker(filtered, "Context district", "rag_district")
    ctx_left, ctx_right = st.columns([1.15, 1])
    with ctx_left:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Screening tier", str(rag_row["risk_tier"]))
        p2.metric("Deprivation pct", f"{rag_row['composite_percentile']:.0f}th")
        p3.metric("Main domain", str(rag_row["primary_domain"]))
        p4.metric("Confidence", str(rag_row["confidence"]))
        st.markdown(f"**Top indicator drivers:** {rag_row['top_drivers']}")
        rag_gaps = scheme_gap_frame(pd.DataFrame([rag_row])).dropna(subset=["gap_proxy_pct"])
        if not rag_gaps.empty:
            st.dataframe(
                rag_gaps[["scheme", "proxy_kind", "proxy_measure_pct", "gap_proxy_pct"]].rename(
                    columns={"scheme": "Programme proxy", "proxy_kind": "Type", "proxy_measure_pct": "Observed (%)", "gap_proxy_pct": "Gap / burden (%)"}
                ).round(1),
                use_container_width=True, hide_index=True,
            )
    with ctx_right:
        rag_spatial = attach_approximate_coordinates(
            profiles[profiles["state_raw"] == rag_row["state_raw"]]
        ).dropna(subset=["approx_lat", "approx_lon"]).copy()
        if not rag_spatial.empty:
            rag_spatial["focus"] = np.where(
                rag_spatial["district_raw"] == rag_row["district_raw"], "Selected district", "Same-state peer"
            )
            rag_spatial["marker_size"] = np.where(rag_spatial["focus"] == "Selected district", 16, 8)
            fig_ctx = px.scatter_geo(
                rag_spatial, lat="approx_lat", lon="approx_lon", color="focus", size="marker_size",
                hover_name="district_raw",
                hover_data={"composite_percentile": ":.1f", "risk_tier": True, "approx_lat": False, "approx_lon": False, "marker_size": False},
                color_discrete_map={"Selected district": "#ad3f24", "Same-state peer": "#587f78"},
                title=f"Contextual locator · {rag_row['district_raw']} & {rag_row['state_raw']} peers",
            )
            c_lat = float(rag_spatial.loc[rag_spatial["focus"] == "Selected district", "approx_lat"].iloc[0])
            c_lon = float(rag_spatial.loc[rag_spatial["focus"] == "Selected district", "approx_lon"].iloc[0])
            fig_ctx.update_geos(
                visible=True, scope="asia", center={"lat": c_lat, "lon": c_lon}, projection_scale=11,
                showcountries=True, countrycolor="#cbd5e1", showland=True, landcolor="#f8f6f0",
            )
            fig_ctx.update_layout(height=275, margin={"r": 0, "t": 35, "l": 0, "b": 0}, legend_title_text="")
            st.plotly_chart(fig_ctx, use_container_width=True)

    st.markdown("Try: **Why is this district flagged?** · **What scheme gaps are visible?** · **What does the MPI cross-check say?** · **यह जिला क्यों चिन्हित है?** · **योजना कवरेज क्या है?**")
    if "pragati_history" not in st.session_state:
        st.session_state["pragati_history"] = []
    with st.form("pragati_chat_form", clear_on_submit=True):
        prompt = st.text_input("Ask about the selected district (English or Hindi)", placeholder="Ask about drivers, screening tier, MPI, budget scenario, or scheme proxies")
        ask = st.form_submit_button("Ask PRAGATI")
    if ask and prompt.strip():
        response = answer_query(prompt.strip(), profiles, district=str(rag_row["district_raw"]), state=str(rag_row["state_raw"]))
        st.session_state["pragati_history"].append({"question": prompt.strip(), **response})
    for item in reversed(st.session_state["pragati_history"][-8:]):
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            with st.expander("Retrieved evidence & TF-IDF similarity"):
                if item["evidence"]:
                    st.dataframe(pd.DataFrame(item["evidence"]), use_container_width=True, hide_index=True)
                else:
                    st.caption("No local evidence was retrieved.")
    if st.button("Clear this session's chat", key="clear_pragati"):
        st.session_state["pragati_history"] = []
        st.rerun()
    st.info("RAG architecture note: uses character/word n-gram TF-IDF embeddings with cosine-similarity ranking over the verified district corpus and constrained English/Hindi answer templates to prevent statistical hallucination.")

# Shared technical notes, not a seventh module.
with st.expander("Model check and data provenance"):
    model = metrics.get("model", {})
    model_name = model.get("model_name", "Gradient-Boosted Decision Trees (XGBRegressor / HistGradientBoostingRegressor)")
    st.write(f"Cross-check model ({model_name}): 5-fold CV R² **{model.get('cv_r2', 'n/a')}**, MAE **{model.get('cv_mae_pct_points', 'n/a')} percentage points**, n={model.get('n_districts_with_mpi', 'n/a')}. Uses `xgboost.XGBRegressor` when installed and falls back to `HistGradientBoostingRegressor` otherwise.")
    st.caption("NFHS-5 is 2019–21. The MPI file's source is undocumented. Census SC/ST district estimates are published for only a subset. See docs/DATA_QUALITY.md and docs/VIVA_ALIGNMENT.md.")
    st.markdown("Sources: [NFHS-5 (IIPS / MoHFW)](https://www.data.gov.in/catalog/national-family-health-survey-5-nfhs-5-india-districts-factsheet-data-provisional) · [CHIRPS v3](https://www.chc.ucsb.edu/data/chirps3) · [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/) · [NASA VIIRS Black Marble](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/science-domain/nighttime-lights/) · [Copernicus Sentinel-2](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html)")
