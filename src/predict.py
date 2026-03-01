import joblib
import pandas as pd
import numpy as np


def predict_demand(station_id, forecast_date, daily_data):

    model = joblib.load("models/daily_model.pkl")
    le = joblib.load("models/station_encoder.pkl")

    forecast_date = pd.to_datetime(forecast_date)

    station_df = daily_data[daily_data["stationID"] == station_id]
    station_df = station_df.sort_values("date")

    if station_df.empty:
        return "Station not found in dataset."

    # Only use historical data BEFORE forecast date
    station_df = station_df[station_df["date"] < forecast_date]

    if len(station_df) < 3:
        return "Need at least 3 historical days."

    last_days = station_df.tail(14)  # take up to 14 if available

    # Safe lag extraction
    lag_1 = last_days["daily_kwh"].iloc[-1]

    lag_7 = last_days["daily_kwh"].iloc[-7] if len(last_days) >= 7 else last_days["daily_kwh"].mean()

    lag_14 = last_days["daily_kwh"].iloc[0] if len(last_days) >= 14 else last_days["daily_kwh"].mean()

    rolling_7 = last_days["daily_kwh"].tail(7).mean()
    rolling_14 = last_days["daily_kwh"].mean()

    day_of_week = forecast_date.dayofweek
    month = forecast_date.month
    is_weekend = 1 if day_of_week >= 5 else 0

    station_encoded = le.transform([station_id])[0]

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

    return round(float(prediction), 2)