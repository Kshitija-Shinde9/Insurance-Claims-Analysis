"""Streamlit dashboard for final ML results and expected claim cost insights."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "fremtpl2_policy_claims.csv"
METRICS_FILE = ROOT / "outputs" / "fremtpl2" / "modeling" / "metrics_summary.csv"
RISK_DECILE_FILE = ROOT / "outputs" / "fremtpl2" / "modeling" / "risk_decile_calibration.csv"
IMPORTANCE_FILE = ROOT / "outputs" / "fremtpl2" / "modeling" / "permutation_importance.csv"


st.set_page_config(
    page_title="Expected Claim Cost Dashboard",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        st.error(
            "Processed data file is missing. Run "
            "`python scripts/build_fremtpl2_modeling_file.py` first."
        )
        st.stop()
    return pd.read_csv(DATA_FILE)


@st.cache_data(show_spinner=False)
def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


df = load_data()
metrics = load_optional_csv(METRICS_FILE)
risk_deciles = load_optional_csv(RISK_DECILE_FILE)
importance = load_optional_csv(IMPORTANCE_FILE)

st.title("Expected Claim Cost Dashboard")
st.caption(
    "Final project dashboard for freMTPL2 motor insurance policies, "
    "pure premium, claim frequency, and model explainability."
)

with st.sidebar:
    st.header("Filters")
    regions = st.multiselect(
        "Region",
        sorted(df["Region"].dropna().unique()),
        default=sorted(df["Region"].dropna().unique())[:8],
    )
    areas = st.multiselect(
        "Area",
        sorted(df["Area"].dropna().unique()),
        default=sorted(df["Area"].dropna().unique()),
    )
    gases = st.multiselect(
        "Vehicle Gas",
        sorted(df["VehGas"].dropna().unique()),
        default=sorted(df["VehGas"].dropna().unique()),
    )
    bonus = st.multiselect(
        "Bonus-Malus Band",
        sorted(df["BonusMalusBand"].dropna().unique()),
        default=sorted(df["BonusMalusBand"].dropna().unique()),
    )

filtered = df[
    df["Region"].isin(regions)
    & df["Area"].isin(areas)
    & df["VehGas"].isin(gases)
    & df["BonusMalusBand"].isin(bonus)
].copy()

if filtered.empty:
    st.warning("No policies match the selected filters.")
    st.stop()

total_exposure = filtered["Exposure"].sum()
claim_frequency = filtered["ClaimNb"].sum() / total_exposure if total_exposure else 0
pure_premium = filtered["TotalClaimAmount"].sum() / total_exposure if total_exposure else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Policies", f"{len(filtered):,}")
k2.metric("Exposure", f"{total_exposure:,.0f}")
k3.metric("Claims", f"{filtered['ClaimNb'].sum():,.0f}")
k4.metric("Claim Frequency", f"{claim_frequency:.4f}")
k5.metric("Pure Premium", f"{pure_premium:,.2f}")

tab1, tab2, tab3 = st.tabs(["Portfolio Risk", "ML Results", "Explainability"])

with tab1:
    c1, c2 = st.columns(2)
    region_summary = (
        filtered.groupby("Region", as_index=False)
        .agg(
            Policies=("IDpol", "count"),
            Exposure=("Exposure", "sum"),
            Claims=("ClaimNb", "sum"),
            TotalClaimAmount=("TotalClaimAmount", "sum"),
        )
    )
    region_summary["PurePremium"] = (
        region_summary["TotalClaimAmount"] / region_summary["Exposure"]
    )
    region_summary["ClaimFrequency"] = region_summary["Claims"] / region_summary["Exposure"]
    region_summary = region_summary.sort_values("PurePremium", ascending=False)

    with c1:
        st.plotly_chart(
            px.bar(
                region_summary.head(15),
                x="PurePremium",
                y="Region",
                orientation="h",
                title="Top Regions by Pure Premium",
                color="PurePremium",
                color_continuous_scale="Reds",
            ),
            use_container_width=True,
        )
    with c2:
        bm = (
            filtered.groupby("BonusMalusBand", as_index=False)
            .agg(
                Exposure=("Exposure", "sum"),
                Claims=("ClaimNb", "sum"),
                TotalClaimAmount=("TotalClaimAmount", "sum"),
            )
        )
        bm["PurePremium"] = bm["TotalClaimAmount"] / bm["Exposure"]
        st.plotly_chart(
            px.line(
                bm,
                x="BonusMalusBand",
                y="PurePremium",
                markers=True,
                title="Pure Premium by Bonus-Malus Band",
            ),
            use_container_width=True,
        )

    c3, c4 = st.columns(2)
    with c3:
        age = (
            filtered.groupby("DriverAgeBand", as_index=False)
            .agg(Exposure=("Exposure", "sum"), Claims=("ClaimNb", "sum"))
        )
        age["ClaimFrequency"] = age["Claims"] / age["Exposure"]
        st.plotly_chart(
            px.bar(
                age,
                x="DriverAgeBand",
                y="ClaimFrequency",
                title="Claim Frequency by Driver Age",
            ),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            px.histogram(
                filtered.sample(min(len(filtered), 80_000), random_state=42),
                x="PurePremium",
                nbins=80,
                title="Pure Premium Distribution",
                log_y=True,
            ),
            use_container_width=True,
        )

with tab2:
    if metrics is None:
        st.info("Run `python scripts/model_expected_claim_cost.py` to create model metrics.")
    else:
        st.subheader("Model Metrics")
        st.dataframe(metrics, use_container_width=True)
        st.plotly_chart(
            px.bar(
                metrics.sort_values("RMSE"),
                x="RMSE",
                y="Model",
                orientation="h",
                title="Model Comparison by RMSE",
            ),
            use_container_width=True,
        )

    if risk_deciles is not None:
        long = risk_deciles.melt(
            id_vars=["RiskDecile"],
            value_vars=["ActualPurePremium", "PredictedPurePremium"],
            var_name="Series",
            value_name="PurePremium",
        )
        st.plotly_chart(
            px.line(
                long,
                x="RiskDecile",
                y="PurePremium",
                color="Series",
                markers=True,
                title="Actual vs Predicted Pure Premium by Risk Decile",
            ),
            use_container_width=True,
        )

with tab3:
    if importance is None:
        st.info("Run `python scripts/model_expected_claim_cost.py` to create feature importance.")
    else:
        st.plotly_chart(
            px.bar(
                importance.sort_values("Importance", ascending=False).head(12),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Permutation Feature Importance",
            ),
            use_container_width=True,
        )

    st.subheader("Interpretation")
    st.write(
        "The dashboard connects policy risk characteristics to expected claim cost. "
        "Higher predicted pure premium segments identify policies that are expected "
        "to generate larger claim costs per exposure unit. This supports pricing, "
        "portfolio monitoring, and risk segmentation."
    )

