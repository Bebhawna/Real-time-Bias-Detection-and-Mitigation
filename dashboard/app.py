import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


import streamlit as st
# from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time
from datetime import datetime

# from src.db_config import fetch_latest_record s
from src.fairness_matrics import compute_fairness_metrics

import plotly.express as px

st.set_page_config(page_title="AI Bias Monitoring Dashboard", layout="wide")

st.title("AI Bias Detection & Mitigation Monitoring")

import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

from src.db_config import (
    fetch_latest_records,
    fetch_final_records
)

# -----------------------------
# CONFIG
# -----------------------------
WINDOW_SIZE = 100

# -----------------------------
# AUTO REFRESH (LIVE DASHBOARD)
# -----------------------------
st_autorefresh(interval=5000, key="refresh")

# -----------------------------
# TITLE
# -----------------------------
st.title("AI Bias Detection & Mitigation Monitoring Dashboard")

# -----------------------------
# FETCH DATA
# -----------------------------
raw_records = fetch_latest_records(WINDOW_SIZE)
final_records = fetch_final_records(WINDOW_SIZE)

raw_df = pd.DataFrame(raw_records)
final_df = pd.DataFrame(final_records)

# -----------------------------
# HANDLE EMPTY DATA
# -----------------------------
if raw_df.empty or final_df.empty:
    st.warning("No streaming data available yet...")
    st.stop()

# -----------------------------
# FAIRNESS METRIC FUNCTION S
# -----------------------------
def compute_dpd(df):
    """
    Demographic Parity Difference
    """
    rates = df.groupby("gender")["prediction"].mean()
    return rates.max() - rates.min()

def compute_di(df):
    """
    Disparate Impact
    """
    rates = df.groupby("gender")["prediction"].mean()
    return rates.min() / rates.max()

# -----------------------------
# CALCULATE FAIRNESS
# -----------------------------
dpd_raw = compute_dpd(raw_df)
dpd_final = compute_dpd(final_df)

di_raw = compute_di(raw_df)
di_final = compute_di(final_df)

# -----------------------------
# METRIC DISPLAY
# -----------------------------
st.subheader("Fairness Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("DPD (RAW)", round(dpd_raw, 3))
col2.metric("DPD (FINAL)", round(dpd_final, 3))

col3.metric("DI (RAW)", round(di_raw, 3))
col4.metric("DI (FINAL)", round(di_final, 3))



# -----------------------------
# Function to compute metrics overtime
# -----------------------------

import pandas as pd

def compute_metrics_over_time(df, step=10):
    dpd_values = []
    di_values = []
    steps = []

    for i in range(step, len(df) + 1, step):
        subset = df.iloc[:i]

        dpd = compute_dpd(subset)
        di = compute_di(subset)

        dpd_values.append(dpd)
        di_values.append(di)
        steps.append(i)

    metrics_df = pd.DataFrame({
        "records_processed": steps,
        "dpd": dpd_values,
        "di": di_values
    })

    return metrics_df

raw_df = pd.DataFrame(fetch_latest_records(WINDOW_SIZE))
final_df = pd.DataFrame(fetch_final_records(WINDOW_SIZE))


raw_metrics = compute_metrics_over_time(raw_df)
final_metrics = compute_metrics_over_time(final_df)

st.subheader("DPD Over Time")

dpd_plot = pd.DataFrame({
    "RAW_DPD": raw_metrics["dpd"],
    "FINAL_DPD": final_metrics["dpd"]
})

st.line_chart(dpd_plot)

st.subheader("DI Over Time")

di_plot = pd.DataFrame({
    "RAW_DI": raw_metrics["di"],
    "FINAL_DI": final_metrics["di"]
})

st.line_chart(di_plot)

# -----------------------------
# MITIGATION INDICATOR
# -----------------------------
st.subheader("Mitigation Status")

if abs(dpd_raw) > 0.1:
    st.error("⚠ Bias detected in RAW predictions")
else:
    st.success("RAW predictions are within fairness threshold")

if abs(dpd_final) < abs(dpd_raw):
    st.success("✔ Mitigation improved fairness")
else:
    st.warning("Mitigation not improving fairness yet")

# -----------------------------
# LATEST RECORDS TABLE
# -----------------------------
st.subheader("Latest Predictions")

st.dataframe(final_df.tail(20))

# -----------------------------
# RAW VS FINAL COMPARISON
# -----------------------------
st.subheader("Before vs After Mitigation")

comparison = pd.DataFrame({
    "Metric": ["DPD", "DI"],
    "RAW": [dpd_raw, di_raw],
    "FINAL": [dpd_final, di_final]
})

st.table(comparison)

# -----------------------------
# FOOTER
# -----------------------------
st.caption("Live bias monitoring system with automatic mitigation.")




st.subheader("Fairness Improvement Check")

if abs(dpd_final) < abs(dpd_raw):
    st.success("✅ DPD Improved After Mitigation")
else:
    st.error("❌ DPD Worse After Mitigation")

if abs(1 - di_final) < abs(1 - di_raw):
    st.success("✅ DI Improved After Mitigation")
else:
    st.error("❌ DI Worse After Mitigation") 