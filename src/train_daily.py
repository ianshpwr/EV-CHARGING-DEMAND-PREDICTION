import os 
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_daily_model(daily_model_df):
    """
    Train XGBoost model for next-day forecasting.
    """
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    upper_limit = daily_model_df["target_next_day"].quantile(0.95)
    # -----------------------------
    # 1. Clip extreme target values
    # -----------------------------
    daily_model_df["target_clipped"] = np.clip(
        daily_model_df["target_next_day"],
        0,
        upper_limit
    )
    # -----------------------------
    # 2. Encode station ID
    # -----------------------------

    le = LabelEncoder()
    daily_model_df["station_encoded"] = le.fit_transform(
        daily_model_df["stationID"]
    )
    # -----------------------------
    # 3. Feature selection
    # -----------------------------
    features = [
        "station_encoded",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_7",
        "rolling_14",
        "day_of_week",
        "month",
        "is_weekend"
    ]
    # -----------------------------
    # 4. Train-test split (time-based)
    # -----------------------------
    split_index = int(len(daily_model_df) * 0.8)

    train_df = daily_model_df.iloc[:split_index]
    test_df = daily_model_df.iloc[split_index:]

    X_train = train_df[features]
    X_test = test_df[features]

    y_train = train_df["target_clipped"]
    y_test = test_df["target_clipped"]

    # -----------------------------
    # 5. Log transform target
    # -----------------------------
    y_train_log = np.log1p(y_train)
    # -----------------------------
    # 6. Initialize and train model
    # -----------------------------
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train_log,verbose=False)
    # -----------------------------
    # 7. Predict & inverse transform
    # -----------------------------
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)
    # -----------------------------
    # 8. Evaluation
    # -----------------------------
    print("Daily MAE:", mean_absolute_error(y_test, pred))
    print("Daily R2:", r2_score(y_test, pred))

    joblib.dump(model, "models/daily_model.pkl")
    joblib.dump(le, "models/station_encoder.pkl")
    # -----------------------------
    # 9. Save model & encoder
    # -----------------------------
    print("Model saved successfully.")
    return model, le