import pandas as pd
import numpy as np
import pickle

# Load model + features
with open('models/ev_demand_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/features.pkl', 'rb') as f:
    feature_list = pickle.load(f)


def predict_station_demand(df):
    """
    Predict next-day EV demand per station
    """

    # -----------------------------
    # VALIDATION
    # -----------------------------
    required_cols = ['connectionTime', 'stationID', 'kWhDelivered']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing columns. Required: {required_cols}")

    # -----------------------------
    # CLEAN DATETIME
    # -----------------------------
    df['connectionTime'] = pd.to_datetime(df['connectionTime'], errors='coerce')
    df = df.dropna(subset=['connectionTime'])

    # -----------------------------
    # DAILY AGGREGATION
    # -----------------------------
    df = df.set_index('connectionTime')

    daily_kwh = (
        df.groupby('stationID')['kWhDelivered']
        .resample('D')
        .sum()
        .reset_index()
    )

    daily_kwh.rename(columns={'kWhDelivered': 'daily_kwh'}, inplace=True)

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    def create_features(group):
        group = group.sort_values('connectionTime').copy()

        # Lag features
        group['lag_1'] = group['daily_kwh'].shift(1)
        group['lag_7'] = group['daily_kwh'].shift(7)
        group['lag_14'] = group['daily_kwh'].shift(14)

        # Rolling
        group['rolling_7'] = group['daily_kwh'].shift(1).rolling(7).mean()
        group['rolling_14'] = group['daily_kwh'].shift(1).rolling(14).mean()

        # Trend
        group['trend_3'] = group['daily_kwh'].shift(1).rolling(3).mean().pct_change()
        group['trend_7'] = group['daily_kwh'].shift(1).rolling(7).mean().pct_change()

        return group

    features_df = []

    for station, group in daily_kwh.groupby("stationID"):
        g = create_features(group.copy())
        g["stationID"] = station  # ensure column exists
        features_df.append(g)

    features_df = pd.concat(features_df, ignore_index=True)

    # -----------------------------
    # TIME FEATURES
    # -----------------------------
    features_df['connectionTime'] = pd.to_datetime(features_df['connectionTime'])

    features_df['day_of_week'] = features_df['connectionTime'].dt.dayofweek
    features_df['month'] = features_df['connectionTime'].dt.month
    features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)

    # -----------------------------
    # GET LATEST PER STATION
    # -----------------------------
    latest_data = (
        features_df
        .sort_values('connectionTime')
        .groupby('stationID')
        .tail(1)
        .set_index('stationID')
    )

    # Fill NaNs
    latest_data = latest_data.fillna(0)

    # -----------------------------
    # 🔥 CRITICAL FIX (FEATURE ALIGNMENT)
    # -----------------------------

    # Add required features used during training
    latest_data["current_kwh"] = latest_data["daily_kwh"]

    latest_data["station_encoded"] = (
        latest_data.index.astype("category").codes
    )

    # Ensure all features exist
    for col in feature_list:
        if col not in latest_data.columns:
            latest_data[col] = 0

    # Select features in correct order
    X_pred = latest_data[feature_list]

    # -----------------------------
    # PREDICT
    # -----------------------------
    delta_pred = model.predict(X_pred)

    latest_data['predicted_delta'] = delta_pred
    latest_data['predicted_kwh'] = (
        latest_data['daily_kwh'] + latest_data['predicted_delta']
    ).clip(lower=0)

    return latest_data['predicted_kwh'].to_dict()


# -----------------------------
# TEST
# -----------------------------
if __name__ == '__main__':
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    np.random.seed(42)

    data = []
    for station in ['A', 'B']:
        for d in dates:
            data.append({
                'connectionTime': d,
                'stationID': station,
                'kWhDelivered': np.random.uniform(5, 30)
            })

    df_test = pd.DataFrame(data)

    try:
        preds = predict_station_demand(df_test)
        print("Predicted Demand:", preds)
    except Exception as e:
        print("Error:", e)