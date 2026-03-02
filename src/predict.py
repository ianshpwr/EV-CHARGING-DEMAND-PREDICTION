def forecast_until_today(station_id, daily_data):

    model = joblib.load("models/daily_model.pkl")
    le = joblib.load("models/station_encoder.pkl")

    station_df = daily_data[daily_data["stationID"] == station_id].copy()
    station_df = station_df.sort_values("date")

    last_date = station_df["date"].max()
    today = pd.Timestamp.today().normalize()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        end=today,
        freq="D"
    )

    simulated = station_df.copy()

    for future_date in future_dates:

        last_days = simulated.tail(14)

        lag_1 = last_days["daily_kwh"].iloc[-1]
        lag_7 = last_days["daily_kwh"].iloc[-7] if len(last_days) >= 7 else last_days["daily_kwh"].mean()
        lag_14 = last_days["daily_kwh"].iloc[0] if len(last_days) >= 14 else last_days["daily_kwh"].mean()

        rolling_7 = last_days["daily_kwh"].tail(7).mean()
        rolling_14 = last_days["daily_kwh"].mean()

        day_of_week = future_date.dayofweek
        month = future_date.month
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

        new_row = pd.DataFrame([{
            "stationID": station_id,
            "date": future_date,
            "daily_kwh": prediction
        }])

        simulated = pd.concat([simulated, new_row], ignore_index=True)

    return simulated