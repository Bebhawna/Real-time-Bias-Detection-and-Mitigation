import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


import streamlit as st
# from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time
from datetime import datetime

from src.db_config import fetch_latest_records
from src.fairness_matrics import compute_fairness_metrics

import plotly.express as px

st.set_page_config(page_title="AI Bias Monitoring Dashboard", layout="wide")

st.title("AI Bias Detection & Mitigation Monitoring")
# st_autorefresh(interval=5000, key="datarefresh")

# WINDOW_SIZE = 100

# if "dpd_history" not in st.session_state:
#     st.session_state.dpd_history = []

# if "di_history" not in st.session_state:
#     st.session_state.di_history = []

# if "timestamps" not in st.session_state:
#     st.session_state.timestamps = []

# placeholder = st.empty()

# while True:

#     records = fetch_latest_records(WINDOW_SIZE)

#     if records:

#         df = pd.DataFrame(records)

#         metrics = compute_fairness_metrics(df,
#                                            group_col="race",
#                                            positive_col="prediction")

#         dpd = metrics["demographic_parity_difference"]
#         di = metrics["disparate_impact"]

#         now = datetime.now().strftime("%H:%M:%S")

#         st.session_state.dpd_history.append(dpd)
#         st.session_state.di_history.append(di)
#         st.session_state.timestamps.append(now)

#         with placeholder.container():

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.metric("Demographic Parity Difference", round(dpd, 4))

#             with col2:
#                 st.metric("Disparate Impact", round(di, 4))

#             st.subheader("DPD Trend")

#             dpd_df = pd.DataFrame({
#                 "Time": st.session_state.timestamps,
#                 "DPD": st.session_state.dpd_history
#             })

#             fig1 = px.line(dpd_df, x="Time", y="DPD")
#             st.plotly_chart(fig1, use_container_width=True)

#             st.subheader("DI Trend")

#             di_df = pd.DataFrame({
#                 "Time": st.session_state.timestamps,
#                 "DI": st.session_state.di_history
#             })

#             fig2 = px.line(di_df, x="Time", y="DI")
#             st.plotly_chart(fig2, use_container_width=True)

#             st.subheader("Latest Predictions")

#             st.dataframe(df.tail(10))

#             st.subheader("Mitigation Status")

#             if dpd < 0.1 and di > 0.8:
#                 st.success("Mitigation working correctly")
#             else:
#                 st.warning("Potential bias detected")

#     time.sleep(5)


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
# DPD TREND GRAPH
# -----------------------------
st.subheader("DPD Over Time")

dpd_trend = pd.DataFrame({
    "RAW_DPD": [compute_dpd(raw_df)],
    "FINAL_DPD": [compute_dpd(final_df)]
})

# st.line_chart(dpd_trend)

import pandas as pd

dpd_df = pd.DataFrame({
    "Type": ["RAW", "FINAL"],
    "DPD": [dpd_raw, dpd_final]
})


st.subheader("DPD Comparison")
st.bar_chart(dpd_df.set_index("Type"))


st.write("RAW rows:", len(raw_df))
st.write("FINAL rows:", len(final_df))



st.write("RAW DPD values:", dpd_raw)
st.write("FINAL DPD values:", dpd_final)

st.write("RAW DI values:", di_raw)
st.write("FINAL DI values:", di_final)

st.write(raw_df.groupby("gender")["prediction"].mean())
# -----------------------------
# DI TREND GRAPH
# -----------------------------
st.subheader("DI Over Time")

di_trend = pd.DataFrame({
    "RAW_DI": [compute_di(raw_df)],
    "FINAL_DI": [compute_di(final_df)]
})

# st.line_chart(di_trend)
di_df = pd.DataFrame({
    "Type": ["RAW", "FINAL"],
    "DI": [di_raw, di_final]
})

st.subheader("DI Comparison")
st.bar_chart(di_df.set_index("Type"))
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


import pandas as pd

dpd_df = pd.DataFrame({
    "Step": [1, 2],
    "RAW_DPD": [dpd_raw, dpd_raw],
    "FINAL_DPD": [dpd_raw, dpd_final]
})

di_df = pd.DataFrame({
    "Step": [1, 2],
    "RAW_DI": [di_raw, di_raw],
    "FINAL_DI": [di_raw, di_final]
})

st.subheader("DPD Over Time")
st.line_chart(dpd_df.set_index("Step"))

st.subheader("DI Over Time")
st.line_chart(di_df.set_index("Step"))


st.subheader("Fairness Improvement Check")

if abs(dpd_final) < abs(dpd_raw):
    st.success("✅ DPD Improved After Mitigation")
else:
    st.error("❌ DPD Worse After Mitigation")

if abs(1 - di_final) < abs(1 - di_raw):
    st.success("✅ DI Improved After Mitigation")
else:
    st.error("❌ DI Worse After Mitigation")