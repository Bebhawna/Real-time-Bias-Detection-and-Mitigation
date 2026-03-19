import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time
from datetime import datetime


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
# FAIRNESS METRIC FUNCTIONS
# -----------------------------

def compute_dpd(df, column):
    """
    Demographic Parity Difference
    """
    rates = df.groupby(column)["prediction"].mean()
    return rates.max() - rates.min()


def compute_di(df, column):
    """
    Disparate Impact
    """
    rates = df.groupby(column)["prediction"].mean()

    if rates.max() == 0:
        return 0

    return rates.min() / rates.max()


# -----------------------------
# CALCULATE FAIRNESS
# -----------------------------

dpd_gender_raw = compute_dpd(raw_df, "gender")
dpd_gender_final = compute_dpd(final_df, "gender")

di_gender_raw = compute_di(raw_df, "gender")
di_gender_final = compute_di(final_df, "gender")


dpd_race_raw = compute_dpd(raw_df, "race")
dpd_race_final = compute_dpd(final_df, "race")

di_race_raw = compute_di(raw_df, "race")
di_race_final = compute_di(final_df, "race")




# -----------------------------
# METRIC DISPLAY
# -----------------------------
st.subheader("Fairness Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("DPD (RAW)", round(dpd_gender_raw, 3))
col2.metric("DPD (FINAL)", round(dpd_gender_final, 3))

col3.metric("DI (RAW)", round(di_gender_raw, 3))
col4.metric("DI (FINAL)", round(di_gender_final, 3))



# -----------------------------
# Function to compute metrics overtime
# -----------------------------

import pandas as pd

def compute_metrics_over_time(df, column, step=10):
    dpd_values = []
    di_values = []
    steps = []

    for i in range(step, len(df) + 1, step):
        subset = df.iloc[:i]

        dpd = compute_dpd(subset,column)
        di = compute_di(subset,column)

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


raw_metrics_gender = compute_metrics_over_time(raw_df,"gender")
final_metrics_gender = compute_metrics_over_time(final_df,"gender")

raw_metrics_race = compute_metrics_over_time(raw_df,"race")
final_metrics_race = compute_metrics_over_time(final_df,"race")

st.subheader("DPD Over Time")

dpd_plot_gender = pd.DataFrame({
    "RAW_DPD": raw_metrics_gender["dpd"],
    "FINAL_DPD": final_metrics_gender["dpd"]
})

dpd_plot_race = pd.DataFrame({
    "RAW_DPD": raw_metrics_race["dpd"],
    "FINAL_DPD": final_metrics_race["dpd"]
})

st.line_chart(dpd_plot_gender)
# st.line_chart(dpd_plot_race)


st.subheader("DI Over Time")

di_plot_gender = pd.DataFrame({
    "RAW_DI": raw_metrics_gender["di"],
    "FINAL_DI": final_metrics_gender["di"]
})

di_plot_race = pd.DataFrame({
    "RAW_DI": raw_metrics_race["di"],
    "FINAL_DI": final_metrics_race["di"]
})

st.line_chart(di_plot_gender)


# st.line_chart(di_plot_race)

# -----------------------------
# MITIGATION INDICATOR
# -----------------------------
st.subheader("Mitigation Status")

if abs(dpd_gender_raw) > 0.05:
    st.error("⚠ Bias detected in RAW predictions")
else:
    st.success("RAW predictions are within fairness threshold")

if abs(dpd_gender_final) < abs(dpd_gender_raw):
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
    "RAW": [dpd_gender_raw, di_gender_raw],
    "FINAL": [dpd_gender_final, di_gender_final]
})

st.table(comparison)

# -----------------------------
# FOOTER
# -----------------------------
st.caption("Live bias monitoring system with automatic mitigation.")


st.subheader("Fairness Improvement Check")

if abs(dpd_gender_final) < abs(dpd_gender_raw):
    st.success("✅ DPD Improved After Mitigation")
else:
    st.error("❌ DPD Worse After Mitigation")

if abs(1 - di_gender_final) < abs(1 - di_gender_raw):
    st.success("✅ DI Improved After Mitigation")
else:
    st.error("❌ DI Worse After Mitigation") 