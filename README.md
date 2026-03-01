#  EV Charging Demand Forecasting

A machine learning-based time-series forecasting system designed to predict EV charging demand at station and system levels.

This project implements feature-engineered forecasting models using ensemble learning methods to estimate future energy demand (kWh) for electric vehicle charging infrastructure.

---

##  Project Overview

With increasing EV adoption, accurate charging demand forecasting is essential for:

- Grid stability
- Infrastructure scaling
- Load balancing
- Energy optimization
- Smart city planning

This project builds a complete pipeline including:

- Hourly demand aggregation
- Daily demand forecasting
- Demand occurrence classification
- Feature engineering with lag & rolling statistics
- Ensemble-based regression modeling

---

##  Feature Engineering

The forecasting models use advanced time-series features:

###  Time Features
- Hour of day
- Day of week
- Month
- Weekend indicator
- Cyclical encoding (sin/cos transformation)

### Lag Features
- lag_1 (previous period demand)
- lag_7 (weekly dependency)
- lag_24 (daily seasonality)
- lag_14 (extended weekly memory)

###  Rolling Statistics
- Rolling mean (7-day, 14-day, 24-hour)
- Volatility estimation

These features allow the model to capture temporal dependencies, seasonality, and trend behavior.

---

##  Modeling Approach

### 1️⃣ Demand Occurrence Prediction
Model: RandomForestClassifier  
Purpose: Predict whether demand will occur in the next period.

### 2️⃣ Demand Magnitude Forecasting
Model: XGBoost Regressor  
Purpose: Predict future energy demand (kWh).

Enhancements applied:
- Log transformation of target
- Outlier clipping (95th percentile)
- Temporal train-test split (no data leakage)

---

##  Model Performance

Below is the comparison between actual and predicted daily demand values:

![Actual vs Predicted Demand](images/actual_vs_predicted.png)

The model successfully captures overall trend and seasonality patterns in EV demand.

Evaluation Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

##  Project Structure
EV-Charging-Demand-Forecasting/
│
├── data/
│ └── caltech_full.csv
│
├── models/
│ ├── daily_model.pkl
│ └── station_encoder.pkl
│
├── src/
│ ├── main.py
│ ├── train.py
│ ├── train_daily.py
│ ├── preprocessing.py
│ ├── predict.py
│ ├── utils.py
│ └── eda_analysis.py
│
├── images/
│ └── actual_vs_predicted.png
│
├── README.md
├── requirements.txt
└── .gitignore


---

## Model Persistence

Trained models are saved using joblib:

```python
import joblib
model = joblib.load("daily_model.pkl")

