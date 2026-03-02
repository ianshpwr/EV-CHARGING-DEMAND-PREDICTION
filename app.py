import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from src.preprocessing import load_and_prepare_hourly, build_daily_dataset
from src.eda_analysis import (
    plot_historical_trend,
    plot_monthly_trend,
    plot_weekday_heatmap,
    plot_top_stations,
    plot_demand_distribution,
    plot_system_trend
)

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(page_title="EV Demand Dashboard", layout="wide")

st.title("⚡ Intelligent EV Charging Demand Dashboard")

# ------------------------------------------------
# Initialize Session State
# ------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "lag_1" not in st.session_state:
    st.session_state.lag_1 = None

if "rolling_7" not in st.session_state:
    st.session_state.rolling_7 = None

# ------------------------------------------------
# Load Data (Cached)
# ------------------------------------------------
@st.cache_data
def load_data():
    hourly_data = load_and_prepare_hourly("data/caltech_full.csv")
    daily_data = build_daily_dataset(hourly_data)
    return daily_data

daily_data = load_data()

# Load model + encoder
model = joblib.load("models/daily_model.pkl")
le = joblib.load("models/station_encoder.pkl")

# ------------------------------------------------
# Sidebar Controls
# ------------------------------------------------
stations = sorted(daily_data["stationID"].unique())

col1, col2 = st.columns(2)

with col1:
    selected_station = st.selectbox("Select Station", stations)

with col2:
    selected_date = st.date_input("Forecast Date")

# Always prepare station dataframe
station_df = daily_data[daily_data["stationID"] == selected_station]
station_df = station_df.sort_values("date")

# ------------------------------------------------
# Prediction Logic
# ------------------------------------------------
if st.button("Generate Forecast"):

    forecast_date = pd.to_datetime(selected_date)
    hist_df = station_df[station_df["date"] < forecast_date]

    if len(hist_df) < 3:
        st.error("Not enough historical data.")
    else:
        last_days = hist_df.tail(14)

        lag_1 = last_days["daily_kwh"].iloc[-1]
        lag_7 = last_days["daily_kwh"].iloc[-7] if len(last_days) >= 7 else last_days["daily_kwh"].mean()
        lag_14 = last_days["daily_kwh"].iloc[0] if len(last_days) >= 14 else last_days["daily_kwh"].mean()

        rolling_7 = last_days["daily_kwh"].tail(7).mean()
        rolling_14 = last_days["daily_kwh"].mean()

        day_of_week = forecast_date.dayofweek
        month = forecast_date.month
        is_weekend = 1 if day_of_week >= 5 else 0

        station_encoded = le.transform([selected_station])[0]

        input_data = pd.DataFrame([{
            "station_encoded": station_encoded,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "rolling_7": rolling_7,
            "rolling_14": rolling_14,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend
        }])

        pred_log = model.predict(input_data)
        prediction = float(np.expm1(pred_log)[0])

        # Store in session state
        st.session_state.prediction = prediction
        st.session_state.lag_1 = lag_1
        st.session_state.rolling_7 = rolling_7

# ------------------------------------------------
# KPI SECTION
# ------------------------------------------------
st.subheader("📊 Forecast Summary")

k1, k2, k3 = st.columns(3)

if st.session_state.prediction is not None:
    k1.metric("Predicted Demand (kWh)", round(st.session_state.prediction, 2))
    k2.metric("Last Day Demand", round(st.session_state.lag_1, 2))
    k3.metric("7-Day Avg", round(st.session_state.rolling_7, 2))
else:
    k1.metric("Predicted Demand (kWh)", "—")
    k2.metric("Last Day Demand", "—")
    k3.metric("7-Day Avg", "—")

# ------------------------------------------------
# Historical Trend Chart
# ------------------------------------------------
st.subheader("📈 Historical Trend")

fig = plot_historical_trend(
    station_df,
    selected_date,
    st.session_state.prediction
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Demand Level Indicator
# ------------------------------------------------
if st.session_state.prediction is not None:

    st.subheader("⚠ Demand Level")

    if st.session_state.prediction < 5:
        st.success("🟢 Low Demand Expected")
    elif st.session_state.prediction < 20:
        st.warning("🟡 Moderate Demand Expected")
    else:
        st.error("🔴 High Demand / Congestion Risk")

# ------------------------------------------------
# Monthly Trend
# ------------------------------------------------
st.subheader("📅 Monthly Average Demand")
st.plotly_chart(plot_monthly_trend(daily_data), use_container_width=True)

# ------------------------------------------------
# Weekday Heatmap
# ------------------------------------------------
st.subheader("🔥 Weekly Demand Pattern")
st.plotly_chart(plot_weekday_heatmap(station_df), use_container_width=True)

# ------------------------------------------------
# Top Stations
# ------------------------------------------------
st.subheader("🏆 Top 10 High Demand Stations")
st.plotly_chart(plot_top_stations(daily_data), use_container_width=True)
# ------------------------------------------------
# Demand Distribution
# ------------------------------------------------
st.subheader("📊 Demand Distribution")
st.plotly_chart(plot_demand_distribution(station_df), use_container_width=True)
# ------------------------------------------------
# System-Wide Trend
# ------------------------------------------------
st.subheader("📈 System-Wide Demand Trend")
st.plotly_chart(plot_system_trend(daily_data), use_container_width=True)