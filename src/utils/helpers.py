"""
src/utils/helpers.py
====================
Shared utility functions — data loading, station key resolution, stats.

All functions are pure (no side effects) and independently testable.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.preprocessing import load_and_prepare_hourly, build_daily_dataset


# ------------------------------------------------------------------
# Data loading (Streamlit-cached)
# ------------------------------------------------------------------
@st.cache_data(show_spinner="📦 Loading daily dataset…")
def load_daily_data(csv_path: str = "data/caltech_full.csv") -> pd.DataFrame:
    """
    Load raw CSV, build hourly data, then aggregate to daily features.
    Result is cached by Streamlit so it only runs once per session.

    Returns:
        DataFrame with columns: stationID, date, daily_kwh, lag_*, rolling_*, …
    """
    hourly = load_and_prepare_hourly(csv_path)
    daily  = build_daily_dataset(hourly)
    return daily


@st.cache_data(show_spinner="📦 Loading raw CSV…")
def load_raw_csv(csv_path: str = "data/caltech_full.csv") -> pd.DataFrame:
    """
    Load the raw charging session CSV without any transformation.
    Used as input to predict_station_demand().
    """
    return pd.read_csv(csv_path)


# ------------------------------------------------------------------
# Station key resolution
# ------------------------------------------------------------------
def resolve_station_key(station: str, predictions: dict) -> str | None:
    """
    Look up a station's prediction from the predictions dict, handling
    the common dash-vs-underscore format mismatch between UI and model.

    Tries (in order):
        1. Exact match:          predictions["CA-329"]
        2. Underscore variant:   predictions["CA_329"]
        3. Dash variant:         predictions["CA-329"]

    Args:
        station:     Station ID as shown in the UI selectbox
        predictions: Full predictions dict from predict_station_demand()

    Returns:
        The matching key string if found, else None.
    """
    if not predictions:
        return None

    candidates = [
        station,
        station.replace("-", "_"),
        station.replace("_", "-"),
    ]
    for key in candidates:
        if key in predictions:
            return key

    return None


# ------------------------------------------------------------------
# Per-station statistics
# ------------------------------------------------------------------
def get_station_stats(station_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for a single station's historical data.

    Args:
        station_df: Filtered daily_data for one station, sorted by date.

    Returns:
        dict with keys:
            last_demand  (float | None) — most recent day's kWh
            avg_7        (float | None) — 7-day rolling average
            avg_30       (float | None) — 30-day rolling average
            peak         (float | None) — all-time peak kWh
    """
    if station_df.empty:
        return {"last_demand": None, "avg_7": None, "avg_30": None, "peak": None}

    kwh = station_df["daily_kwh"]

    return {
        "last_demand": float(kwh.iloc[-1])          if len(kwh) >= 1  else None,
        "avg_7":       float(kwh.tail(7).mean())     if len(kwh) >= 7  else float(kwh.mean()),
        "avg_30":      float(kwh.tail(30).mean())    if len(kwh) >= 30 else float(kwh.mean()),
        "peak":        float(kwh.max()),
    }


# ------------------------------------------------------------------
# Demand status helper
# ------------------------------------------------------------------
def classify_demand(predicted_kwh: float) -> dict:
    """
    Classify a predicted demand value into a status with display metadata.

    Returns dict with keys: label, color, emoji, streamlit_fn
    """
    if predicted_kwh < 50:
        return {"label": "Normal",     "color": "#1a9e5c", "bg": "#e6faf1", "emoji": "🟢"}
    elif predicted_kwh < 150:
        return {"label": "High Load",  "color": "#c28500", "bg": "#fffbe6", "emoji": "🟡"}
    else:
        return {"label": "Overloaded", "color": "#c0392b", "bg": "#fdecea", "emoji": "🔴"}
