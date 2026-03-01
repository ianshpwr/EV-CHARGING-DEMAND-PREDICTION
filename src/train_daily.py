import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_daily_model(daily_model_df):

    upper_limit = daily_model_df["target_next_day"].quantile(0.95)

    daily_model_df["target_clipped"] = np.clip(
        daily_model_df["target_next_day"],
        0,
        upper_limit
    )

    le = LabelEncoder()
    daily_model_df["station_encoded"] = le.fit_transform(
        daily_model_df["stationID"]
    )

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

    split_index = int(len(daily_model_df) * 0.8)

    train_df = daily_model_df.iloc[:split_index]
    test_df = daily_model_df.iloc[split_index:]

    X_train = train_df[features]
    X_test = test_df[features]

    y_train = train_df["target_clipped"]
    y_test = test_df["target_clipped"]

    y_train_log = np.log1p(y_train)

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train_log)

    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)

    print("Daily MAE:", mean_absolute_error(y_test, pred))
    print("Daily R2:", r2_score(y_test, pred))

    joblib.dump(model, "models/daily_model.pkl")
    joblib.dump(le, "models/station_encoder.pkl")

    print("Model saved successfully.")