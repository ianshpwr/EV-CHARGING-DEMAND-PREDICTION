"""
src/models/demand_model.py
==========================
ML prediction layer — pure, stateless, side-effect-free.

Public API:
    predict_station_demand(df: pd.DataFrame) -> dict[str, float]
        Returns {station_id: predicted_next_day_kwh} for every station in df.
"""

import pickle
import pandas as pd
import numpy as np

# Paths to trained artifacts (relative to project root)
_MODEL_PATH    = "models/ev_demand_model.pkl"
_FEATURES_PATH = "models/features.pkl"


def _load_artifacts():
    """Load model and feature list from disk. Called once per prediction run."""
    with open(_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(_FEATURES_PATH, "rb") as f:
        feature_list = pickle.load(f)
    return model, feature_list


def predict_station_demand(df: pd.DataFrame) -> dict:
    """
    Predict next-day EV charging demand for every station in df.

    Args:
        df: Raw charging session DataFrame with columns:
            ['connectionTime', 'stationID', 'kWhDelivered']

    Returns:
        dict mapping station_id (str) → predicted_next_day_kwh (float).
        Returns {} on validation failure.
    """
    # ------------------------------------------------------------------
    # 0. Load artifacts (inside function → no stale globals on hot-reload)
    # ------------------------------------------------------------------
    model, feature_list = _load_artifacts()

    # ------------------------------------------------------------------
    # 1. Validate input columns
    # ------------------------------------------------------------------
    required = {"connectionTime", "stationID", "kWhDelivered"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    # ------------------------------------------------------------------
    # 2. Parse datetime & drop bad rows
    # ------------------------------------------------------------------
    df = df.copy()
    df["connectionTime"] = pd.to_datetime(df["connectionTime"], errors="coerce")
    df = df.dropna(subset=["connectionTime"])

    if df.empty:
        return {}

    # ------------------------------------------------------------------
    # 3. Daily aggregation per station
    # ------------------------------------------------------------------
    df_idx = df.set_index("connectionTime")
    daily_kwh = (
        df_idx.groupby("stationID")["kWhDelivered"]
        .resample("D")
        .sum()
        .reset_index()
        .rename(columns={"kWhDelivered": "daily_kwh"})
    )

    # ------------------------------------------------------------------
    # 4. Feature engineering per station group
    # ------------------------------------------------------------------
    def _engineer_features(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("connectionTime").copy()

        # Lag features
        g["lag_1"]  = g["daily_kwh"].shift(1)
        g["lag_7"]  = g["daily_kwh"].shift(7)
        g["lag_14"] = g["daily_kwh"].shift(14)

        # Rolling averages (shifted to prevent leakage)
        g["rolling_7"]  = g["daily_kwh"].shift(1).rolling(7).mean()
        g["rolling_14"] = g["daily_kwh"].shift(1).rolling(14).mean()

        # Short-term trend (% change of rolling means)
        g["trend_3"] = g["daily_kwh"].shift(1).rolling(3).mean().pct_change()
        g["trend_7"] = g["daily_kwh"].shift(1).rolling(7).mean().pct_change()

        return g

    frames = []
    for station, grp in daily_kwh.groupby("stationID"):
        engineered = _engineer_features(grp.copy())
        engineered["stationID"] = station
        frames.append(engineered)

    features_df = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Calendar features
    # ------------------------------------------------------------------
    features_df["connectionTime"] = pd.to_datetime(features_df["connectionTime"])
    features_df["day_of_week"]    = features_df["connectionTime"].dt.dayofweek
    features_df["month"]          = features_df["connectionTime"].dt.month
    features_df["is_weekend"]     = (features_df["day_of_week"] >= 5).astype(int)

    # ------------------------------------------------------------------
    # 6. Take the most-recent row per station (latest known state)
    # ------------------------------------------------------------------
    latest = (
        features_df
        .sort_values("connectionTime")
        .groupby("stationID")
        .tail(1)
        .set_index("stationID")
        .fillna(0)
    )

    # ------------------------------------------------------------------
    # 7. Align with training features
    # ------------------------------------------------------------------
    # current_kwh is the alias used during training (= daily_kwh)
    latest["current_kwh"] = latest["daily_kwh"]

    # station_encoded: ordinal from current index (same relative ordering)
    latest["station_encoded"] = latest.index.astype("category").codes

    # Ensure every training feature exists (handle any missing with 0)
    for col in feature_list:
        if col not in latest.columns:
            latest[col] = 0

    X_pred = latest[feature_list]

    # ------------------------------------------------------------------
    # 8. Predict (model predicts delta; add to current demand)
    # ------------------------------------------------------------------
    delta_pred = model.predict(X_pred)
    latest["predicted_delta"] = delta_pred
    latest["predicted_kwh"]   = (latest["daily_kwh"] + latest["predicted_delta"]).clip(lower=0)

    return latest["predicted_kwh"].to_dict()
