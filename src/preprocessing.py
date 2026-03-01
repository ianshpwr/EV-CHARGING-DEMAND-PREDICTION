import pandas as pd
import numpy as np

def load_and_prepare_hourly(path):
    """
    Load raw charging session data and convert it to
    continuous hourly energy consumption per station.
    """
    # -----------------------------
    # 1. Load and parse datetime
    # -----------------------------
    df = pd.read_csv(path)
    df["connectionTime"] = pd.to_datetime(df["connectionTime"])

    df["hour_timestamp"] = df["connectionTime"].dt.floor("h")
    # -----------------------------
    # 2. Aggregate kWh per hour
    # -----------------------------
    hourly_sum = (
        df.groupby(["stationID", "hour_timestamp"])["kWhDelivered"]
        .sum()
        .reset_index()
        .rename(columns={"kWhDelivered": "total_kwh"})
    )
    # -----------------------------
    # 3. Create continuous hourly timeline per station
    # -----------------------------

    full_data = []

    for station in hourly_sum["stationID"].unique():

        station_df = hourly_sum[hourly_sum["stationID"] == station]

        full_range = pd.date_range(
            start=station_df["hour_timestamp"].min(),
            end=station_df["hour_timestamp"].max(),
            freq="h"
        )

        full_station = pd.DataFrame({
            "stationID": station,
            "hour_timestamp": full_range
        })

        merged = full_station.merge(
            station_df,
            on=["stationID", "hour_timestamp"],
            how="left"
        )

        merged["total_kwh"] = merged["total_kwh"].fillna(0)
        full_data.append(merged)

    hourly_data = pd.concat(full_data, ignore_index=True)
    return hourly_data


def build_daily_dataset(hourly_data):
    """
    Convert hourly station data into daily dataset
    with lag, rolling, and calendar features.
    """
    # -----------------------------
    # 1. Aggregate to daily level
    # -----------------------------
    daily_data = (
        hourly_data
        .groupby([
            "stationID",
            hourly_data["hour_timestamp"].dt.date
        ])["total_kwh"]
        .sum()
        .reset_index()
    )

    daily_data.columns = ["stationID", "date", "daily_kwh"]
    daily_data["date"] = pd.to_datetime(daily_data["date"])
    daily_data = daily_data.sort_values(["stationID", "date"])
    # -----------------------------
    # 2. Calendar features
    # -----------------------------
    daily_data["day_of_week"] = daily_data["date"].dt.dayofweek
    daily_data["month"] = daily_data["date"].dt.month
    daily_data["is_weekend"] = (daily_data["day_of_week"] >= 5).astype(int)
    # -----------------------------
    # 3. Lag features
    # -----------------------------
    daily_data["lag_1"] = daily_data.groupby("stationID")["daily_kwh"].shift(1)
    daily_data["lag_7"] = daily_data.groupby("stationID")["daily_kwh"].shift(7)
    daily_data["lag_14"] = daily_data.groupby("stationID")["daily_kwh"].shift(14)
    # -----------------------------
    # 4. Rolling features (shifted to avoid leakage)
    # -----------------------------
    daily_data["rolling_7"] = (
        daily_data.groupby("stationID")["daily_kwh"]
        .rolling(7)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    daily_data["rolling_14"] = (
        daily_data.groupby("stationID")["daily_kwh"]
        .rolling(14)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    # -----------------------------
    # 5. Next-day target
    # -----------------------------
    daily_data["target_next_day"] = (
        daily_data.groupby("stationID")["daily_kwh"].shift(-1)
    )

    return daily_data.dropna()