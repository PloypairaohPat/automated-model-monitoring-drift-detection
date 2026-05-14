import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="ML Model Monitor", page_icon="📊", layout="wide")
st.title("Fraud Detection Model — Live Monitoring Dashboard")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_path = os.path.join(BASE, "outputs", "monitoring_results.csv")

if not os.path.exists(results_path):
    st.error("No monitoring results found. Run `monitor.py` first.")
    st.stop()

results = pd.read_csv(results_path)

# ── Row 1: KPI cards ──────────────────────────────────────────────────────────
latest = results.iloc[-1]
col1, col2, col3, col4 = st.columns(4)

with col1:
    status = (
        "OK" if latest["max_psi"] < 0.10
        else "WARNING" if latest["max_psi"] < 0.20
        else "CRITICAL"
    )
    colour_map = {"OK": "normal", "WARNING": "off", "CRITICAL": "inverse"}
    st.metric("Alert Status", status)

with col2:
    delta = (
        f"{latest['f1'] - results.iloc[-2]['f1']:.4f}" if len(results) > 1 else None
    )
    st.metric("Current F1 Score", f"{latest['f1']:.4f}", delta=delta)

with col3:
    st.metric("Max PSI This Week", f"{latest['max_psi']:.3f}")

with col4:
    st.metric("Drifted Features", int(latest["n_drifted"]))

st.divider()

# ── Row 2: F1 trend chart ─────────────────────────────────────────────────────
st.subheader("Model Performance Over Time")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results["week"], y=results["f1"],
    mode="lines+markers", name="F1 Score",
    line=dict(color="#1A3A6B", width=2.5),
    marker=dict(size=8),
))
fig.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="Alert threshold (0.80)")
fig.update_layout(
    xaxis_title="Production Week", yaxis_title="F1 Score",
    height=350, xaxis=dict(dtick=1),
)
st.plotly_chart(fig, use_container_width=True)

# ── Row 3: PSI trend bar chart ────────────────────────────────────────────────
st.subheader("Feature Drift — Max PSI per Week")
fig2 = go.Figure()
bar_colours = [
    "red" if p >= 0.20 else "orange" if p >= 0.10 else "steelblue"
    for p in results["max_psi"]
]
fig2.add_trace(go.Bar(
    x=results["week"], y=results["max_psi"],
    marker_color=bar_colours,
    name="Max PSI",
))
fig2.add_hline(y=0.10, line_dash="dot", line_color="orange",
               annotation_text="Warning (0.10)", annotation_position="bottom right")
fig2.add_hline(y=0.20, line_dash="dash", line_color="red",
               annotation_text="Critical (0.20)", annotation_position="top left")
fig2.update_layout(
    xaxis_title="Production Week", yaxis_title="Max PSI",
    height=300, xaxis=dict(dtick=1),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Row 4: Alert log ──────────────────────────────────────────────────────────
alert_log_path = os.path.join(BASE, "outputs", "alert_log.jsonl")
if os.path.exists(alert_log_path):
    st.subheader("Alert Log")
    alerts = []
    with open(alert_log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                alerts.append(json.loads(line))
    if alerts:
        alert_df = pd.DataFrame(alerts)[["severity", "type", "message", "action", "timestamp"]]
        st.dataframe(alert_df, use_container_width=True)
    else:
        st.info("No alerts fired yet.")

# ── Row 5: Latest Evidently HTML report ───────────────────────────────────────
st.subheader("Latest Evidently Drift Report")
latest_week = int(results["week"].max())
report_path = os.path.join(BASE, f"reports/evidently/week_{latest_week:02d}_drift.html")

if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=700, scrolling=True)
else:
    st.info(f"No Evidently report found for week {latest_week}.")
