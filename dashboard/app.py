import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


import streamlit as st
import pandas as pd
import time
from datetime import datetime

from src.db_config import fetch_latest_records
from src.fairness_matrics import compute_fairness_metrics

import plotly.express as px

st.set_page_config(page_title="AI Bias Monitoring Dashboard", layout="wide")

st.title("AI Bias Detection & Mitigation Monitoring")

WINDOW_SIZE = 100

if "dpd_history" not in st.session_state:
    st.session_state.dpd_history = []

if "di_history" not in st.session_state:
    st.session_state.di_history = []

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

placeholder = st.empty()

while True:

    records = fetch_latest_records(WINDOW_SIZE)

    if records:

        df = pd.DataFrame(records)

        metrics = compute_fairness_metrics(df,
                                           group_col="race",
                                           positive_col="prediction")

        dpd = metrics["demographic_parity_difference"]
        di = metrics["disparate_impact"]

        now = datetime.now().strftime("%H:%M:%S")

        st.session_state.dpd_history.append(dpd)
        st.session_state.di_history.append(di)
        st.session_state.timestamps.append(now)

        with placeholder.container():

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Demographic Parity Difference", round(dpd, 4))

            with col2:
                st.metric("Disparate Impact", round(di, 4))

            st.subheader("DPD Trend")

            dpd_df = pd.DataFrame({
                "Time": st.session_state.timestamps,
                "DPD": st.session_state.dpd_history
            })

            fig1 = px.line(dpd_df, x="Time", y="DPD")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("DI Trend")

            di_df = pd.DataFrame({
                "Time": st.session_state.timestamps,
                "DI": st.session_state.di_history
            })

            fig2 = px.line(di_df, x="Time", y="DI")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Latest Predictions")

            st.dataframe(df.tail(10))

            st.subheader("Mitigation Status")

            if dpd < 0.1 and di > 0.8:
                st.success("Mitigation working correctly")
            else:
                st.warning("Potential bias detected")

    time.sleep(5)