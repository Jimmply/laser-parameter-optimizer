"""Laser Parameter Optimizer Dashboard — Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import GenConfig, LaserParamGenerator
from materials import MATERIALS
from optimizer import LaserParameterOptimizer, OptimizeRequest

st.set_page_config(page_title="Laser Parameter Optimizer", page_icon="🔆", layout="wide")

GRADE_COLORS = {"A": "#2ecc71", "B": "#f39c12", "C": "#e67e22", "Reject": "#e74c3c"}


@st.cache_resource
def load_model() -> tuple[pd.DataFrame, LaserParameterOptimizer, object]:
    df = LaserParamGenerator(GenConfig(n_samples=8000)).generate()
    opt = LaserParameterOptimizer()
    results = opt.fit(df)
    return df, opt, results


df, optimizer, results = load_model()

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("🔆 Laser Parameter Optimizer")
st.sidebar.markdown(
    "Select a material and target weld depth — the surrogate model finds "
    "the optimal Nd:YAG / LASAG SLS 200 parameters."
)

material_name = st.sidebar.selectbox("Material", list(MATERIALS.keys()))
mat = MATERIALS[material_name]
st.sidebar.markdown(f"*{mat.notes}*")
st.sidebar.markdown(f"**Shielding gas:** {mat.preferred_gas}")

target_pen = st.sidebar.slider(
    "Target Penetration Depth (μm)", 50, 2000, 500, step=25
)
thickness  = st.sidebar.slider("Material Thickness (mm)", 0.1, 4.0, 1.0, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model Penetration MAE:** {results.penetration_mae_um:.0f} μm")

# ── Main ───────────────────────────────────────────────────────
st.title("Laser Welding Parameter Optimizer")
st.markdown(
    "XGBoost surrogate model + scipy optimization for pulsed Nd:YAG and LASAG SLS 200 systems. "
    "Covers the full LSR Welding material repertoire: Ti-6Al-4V, Inconel, Cobalt-Cr, stainless, aluminum."
)

if st.button("⚡ Find Optimal Parameters", type="primary"):
    with st.spinner("Optimizing..."):
        req = OptimizeRequest(
            material_name=material_name,
            target_penetration_um=float(target_pen),
            thickness_mm=float(thickness),
        )
        res = optimizer.recommend(req)

    grade_color = GRADE_COLORS.get(res.quality_grade, "#999")

    st.markdown("### Recommended Parameters")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Power (W)",        f"{res.power_w}")
    c2.metric("Pulse (ms)",       f"{res.pulse_ms}")
    c3.metric("Frequency (Hz)",   f"{res.frequency_hz}")
    c4.metric("Travel Speed (mm/s)", f"{res.travel_speed_mm_s}")
    c5.metric("Spot Size (μm)",   f"{res.spot_size_um:.0f}")

    st.markdown("### Predicted Outcomes")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Penetration Depth", f"{res.predicted_penetration_um:.0f} μm",
              delta=f"{res.predicted_penetration_um - target_pen:+.0f} μm vs target")
    o2.metric("Defect Probability", f"{res.predicted_defect_prob:.1%}")
    o3.markdown(
        f"<div style='background:{grade_color};padding:12px;border-radius:8px;"
        f"text-align:center;color:white;font-weight:bold;font-size:18px'>"
        f"Quality Grade<br>{res.quality_grade}</div>",
        unsafe_allow_html=True,
    )
    o4.metric("Model Confidence", f"{res.confidence:.1%}")

    st.markdown("---")

# ── Material comparison scatter ────────────────────────────────
st.subheader("Process Window — Power vs Travel Speed by Material")
sample = df.sample(min(3000, len(df)), random_state=1)
fig = px.scatter(
    sample, x="travel_speed_mm_s", y="power_w",
    color="material", symbol="quality_grade",
    opacity=0.5,
    labels={"travel_speed_mm_s": "Travel Speed (mm/s)", "power_w": "Power (W)"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

# ── Quality distribution per material ─────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Quality Grade by Material")
    grade_dist = df.groupby(["material", "quality_grade"]).size().reset_index(name="count")
    fig2 = px.bar(
        grade_dist, x="material", y="count", color="quality_grade",
        color_discrete_map=GRADE_COLORS, barmode="stack",
        labels={"count": "Weld Trials", "material": ""},
    )
    fig2.update_layout(height=300, xaxis_tickangle=-30)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Feature Importance (Penetration Model)")
    fi = results.feature_importances
    fig3 = px.bar(fi, orientation="h", color=fi.values, color_continuous_scale="Oranges",
                  labels={"value": "Importance", "index": ""})
    fig3.update_layout(showlegend=False, coloraxis_showscale=False, height=300)
    st.plotly_chart(fig3, use_container_width=True)

# ── Penetration vs power for selected material ─────────────────
st.subheader(f"Penetration Depth vs Power — {material_name}")
mat_df = df[df["material"] == material_name]
fig4 = px.scatter(
    mat_df, x="power_w", y="penetration_um", color="quality_grade",
    color_discrete_map=GRADE_COLORS, opacity=0.6,
    labels={"power_w": "Laser Power (W)", "penetration_um": "Penetration Depth (μm)"},
)
fig4.update_layout(height=300)
st.plotly_chart(fig4, use_container_width=True)
