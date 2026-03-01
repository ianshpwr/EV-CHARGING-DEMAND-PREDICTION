import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.preprocessing import load_and_prepare_hourly, build_daily_dataset

st.set_page_config(page_title="EV Demand Forecast", layout="centered")

st.title("🔋 EV Charging Demand Forecast")
st.write("Predict next-day charging demand for a selected station.")

# Load data
@st.cache_data
def load_data():
    hourly_data = load_and_prepare_hourly("data/caltech_full.csv")
    daily_data = build_daily_dataset(hourly_data)
    return daily_data

daily_data = load_data()

# Load model
model = joblib.load("models/daily_model.pkl")
le = joblib.load("models/station_encoder.pkl")

# Dropdown station list
stations = sorted(daily_data["stationID"].unique())
selected_station = st.selectbox("Select Station ID", stations)

# Date selector
selected_date = st.date_input("Select Forecast Date")

if st.button("Predict Demand"):

    forecast_date = pd.to_datetime(selected_date)

    station_df = daily_data[daily_data["stationID"] == selected_station]
    station_df = station_df.sort_values("date")
    station_df = station_df[station_df["date"] < forecast_date]

    if len(station_df) < 3:
        st.error("Not enough historical data.")
    else:
        last_days = station_df.tail(14)

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
        prediction = np.expm1(pred_log)[0]

        st.success(f"Predicted Demand: {round(prediction, 2)} kWh")

        if prediction < 5:
            st.info("Low Demand Expected")
        elif prediction < 20:
            st.warning("Moderate Demand Expected")
        else:
            st.error("High Demand / Congestion Risk")