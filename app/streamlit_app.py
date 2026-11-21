"""KSHETRA-AI dashboard (prototype): district deprivation atlas and explainable profiles.

Run:  streamlit run app/streamlit_app.py
Requires the pipeline output in outputs/ (run `python -m kshetra.pipeline` first).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kshetra.config import DOMAIN_LABELS, INDICATOR_LABELS, INDICATORS  # noqa: E402

PROFILES = ROOT / "outputs" / "district_profiles.csv"
METRICS = ROOT / "outputs" / "model_metrics.json"

st.set_page_config(page_title="KSHETRA-AI", layout="wide")


@st.cache_data
def load_profiles() -> pd.DataFrame:
    return pd.read_csv(PROFILES)


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS, encoding="utf-8") as fh:
        return json.load(fh)


st.title("KSHETRA-AI")
st.caption("District deprivation atlas. A risk signal for verification, not a finding that an area is poor.")

if not PROFILES.exists():
    st.error("No outputs found. Run `python -m kshetra.pipeline` after placing the raw CSVs in data/raw/.")
    st.stop()

df = load_profiles()
metrics = load_metrics()

with st.sidebar:
    st.header("Filters")
    states = sorted(df["state_raw"].unique())
    chosen_states = st.multiselect("State / UT", states, default=[])
    tiers = ["Very high", "High", "Moderate", "Lower"]
    chosen_tiers = st.multiselect("Risk tier", tiers, default=tiers)
    chosen_conf = st.multiselect("Confidence", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])

view = df[df["risk_tier"].isin(chosen_tiers) & df["confidence"].isin(chosen_conf)]
if chosen_states:
    view = view[view["state_raw"].isin(chosen_states)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Districts shown", f"{len(view):,}")
c2.metric("Very high risk tier", f"{(view['risk_tier'] == 'Very high').sum():,}")
c3.metric("Districts with Census-2011 MPI", f"{view['mpi_hcr_pct'].notna().sum():,}")
c4.metric("Model CV R² (MPI from NFHS-5)", f"{metrics['model']['cv_r2']:.2f}")

tab_atlas, tab_district, tab_model, tab_data = st.tabs(["Atlas", "District profile", "Model check", "Data notes"])

with tab_atlas:
    by_state = (
        view.groupby("state_raw")["composite_index"].mean().sort_values(ascending=False).head(20).reset_index()
    )
    fig = px.bar(by_state, x="composite_index", y="state_raw", orientation="h",
                 labels={"composite_index": "Mean composite deprivation (z)", "state_raw": ""},
                 title="Top 20 states / UTs by mean composite deprivation (filtered districts)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
    st.plotly_chart(fig, use_container_width=True)

    ranked = view.sort_values("composite_index", ascending=False)[
        ["state_raw", "district_raw", "composite_percentile", "risk_tier", "primary_domain",
         "top_drivers", "confidence", "mpi_hcr_pct"]
    ].rename(columns={
        "state_raw": "State", "district_raw": "District", "composite_percentile": "Percentile",
        "risk_tier": "Tier", "primary_domain": "Main domain", "top_drivers": "Top drivers",
        "confidence": "Confidence", "mpi_hcr_pct": "Census-2011 MPI HCR (%)",
    })
    st.dataframe(ranked.round(1), use_container_width=True, hide_index=True)

with tab_district:
    options = (df["district_raw"] + " - " + df["state_raw"]).tolist()
    pick = st.selectbox("District", options, index=options.index(
        (df.sort_values("composite_index", ascending=False)["district_raw"] + " - "
         + df.sort_values("composite_index", ascending=False)["state_raw"]).iloc[0]))
    row = df.iloc[options.index(pick)]

    a, b, c = st.columns(3)
    a.metric("Composite percentile", f"{row['composite_percentile']:.0f}")
    b.metric("Risk tier", row["risk_tier"])
    c.metric("Confidence", row["confidence"])
    st.write(f"**Main domain:** {row['primary_domain']}  |  **Indicators missing:** {int(row['indicators_missing'])}"
             f"  |  **NFHS-5 households surveyed:** {int(row['survey_households']) if pd.notna(row['survey_households']) else 'n/a'}")

    contrib = pd.DataFrame({
        "Indicator": [INDICATOR_LABELS[k] for k in INDICATORS],
        "Contribution to composite": [row[f"contrib_{k}"] for k in INDICATORS],
        "Raw value (%)": [row[k] for k in INDICATORS],
    }).sort_values("Contribution to composite")
    fig = px.bar(contrib, x="Contribution to composite", y="Indicator", orientation="h",
                 hover_data=["Raw value (%)"], title="Why this district scores as it does (positive = more deprived)")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    domain_scores = pd.DataFrame({
        "Domain": [DOMAIN_LABELS[d] for d in DOMAIN_LABELS],
        "Score (z)": [row[f"score_{d}"] for d in DOMAIN_LABELS],
    })
    st.dataframe(domain_scores.round(2), hide_index=True)

    if pd.notna(row["mpi_hcr_pct"]):
        st.info(f"Census-2011 MPI headcount ratio (external reference, {row['mpi_match']} name match): {row['mpi_hcr_pct']:.1f}%")
    else:
        st.warning(f"No Census-2011 MPI record linked. Model estimate: {row['proxy_mpi_hcr_pct']:.1f}% (not an observation).")
    st.caption("Next step for any flagged district: local field verification before any decision.")

with tab_model:
    st.subheader("Cross-check model: MPI from NFHS-5 indicators")
    st.write(
        f"Cross-validated R² **{metrics['model']['cv_r2']}**, MAE **{metrics['model']['cv_mae_pct_points']} pp** "
        f"on {metrics['model']['n_districts_with_mpi']} districts. Spearman correlation between the explainable "
        f"composite and the MPI headcount ratio: **{metrics['model']['spearman_composite_vs_mpi']}**."
    )
    imp = pd.DataFrame({"Indicator": list(metrics["model"]["permutation_importance"].keys()),
                        "Importance": list(metrics["model"]["permutation_importance"].values())})
    st.plotly_chart(px.bar(imp, x="Importance", y="Indicator", orientation="h",
                           title="Permutation importance (held-out folds)").update_layout(
        yaxis={"categoryorder": "total ascending"}, height=460), use_container_width=True)
    st.caption("The MPI file's provenance is not documented, so high agreement may partly reflect shared underlying "
               "survey data. Treat this as a consistency check, not independent validation.")

with tab_data:
    st.markdown("""
- **NFHS-5 (2019-21)**: district indicators for 707 districts (health, nutrition, education, living standards).
- **Census 2011 (MPI table)**: district MPI headcount ratio and SC/ST counts for 629 districts matched to NFHS-5 names.
- **Census 2011 SC/ST table**: only 84 rows are district-published; the other 623 hold state averages and are excluded.
- **Satellite signals (CHIRPS, VIIRS, Sentinel-2)**: not yet integrated; see the README roadmap.

See `docs/DATA_QUALITY.md` for every issue found in the raw files and how it was handled.
""")
