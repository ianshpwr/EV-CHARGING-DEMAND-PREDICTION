# Project 15: Intelligent EV Charging Demand Prediction & Agentic Infrastructure Planning

## From Usage Analytics to Autonomous Grid & Station Planning

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

## Exploratory Data Analysis

### 📈 Historical Demand Trend
![Historical Demand Trend](images/Historical_Trend.png)

### 📅 Monthly Demand Trend
![Monthly Demand Trend](images/Monthly_Average_Demand.png)

### 🔥 Weekly Demand Pattern
![Weekly Demand Pattern](images/Weekly_Demand_Pattern.png)

### Demand Distributions
![Top 10 High Demand Stations](images/Demand_Distribution.png)

---

##  Evaluation Metrics

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

## Deployed Link

[Hosted Application](https://evpreds.streamlit.app/)


## Model Persistence

Trained models are saved using joblib:

```python
import joblib
model = joblib.load("daily_model.pkl")



