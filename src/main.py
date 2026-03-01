from preprocessing import load_and_prepare_hourly, build_daily_dataset
from train_daily import train_daily_model
from predict import predict_demand
import os


print("MAIN FILE STARTED")

# Ensure models folder exists
if not os.path.exists("models"):
    os.makedirs("models")

print("Loading data...")
hourly_data = load_and_prepare_hourly("data/caltech_full.csv")

print("Building daily dataset...")
daily_data = build_daily_dataset(hourly_data)

# Train only if model doesn't exist
if not os.path.exists("models/daily_model.pkl"):
    print("Training model...")
    train_daily_model(daily_data)
else:
    print("Model already exists. Skipping training.")

# ---- USER INPUT ----
print("\n🔋 EV Charging Demand Forecast\n")

station_id = input("Enter Station ID: ")
forecast_date = input("Enter Forecast Date (YYYY-MM-DD): ")

prediction = predict_demand(
    station_id=station_id,
    forecast_date=forecast_date,
    daily_data=daily_data
)

print("\n📊 Forecast Result")
print("---------------------------")
print("Station:", station_id)
print("Date:", forecast_date)
print("Predicted Demand (kWh):", prediction)