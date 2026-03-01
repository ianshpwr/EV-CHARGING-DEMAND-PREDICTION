import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt
import seaborn as sns

def eda(df):
  # Plot 1: Energy Demand by Hour
  # Identifying the peak load periods for infrastructure planning
  plt.figure(figsize=(12, 5))
  plt.subplot(1, 2, 1)
  sns.lineplot(data=df, x='hour', y='kWhDelivered', estimator='mean')
  plt.title("Average Energy Demand (kWh) per Hour")
  plt.grid(True)

  # Plot 2: Connection Duration vs Energy
  #Identifying session inefficiencies.
  plt.subplot(1, 2, 2)
  sns.scatterplot(data=df, x='charging_duration_hr', y='kWhDelivered', alpha=0.3)
  plt.title("Connection Duration vs. kWh Delivered")

  plt.tight_layout()
  plt.show()


#Potting the graph for feature importance's result


def feature_importance_plot(model, feature_cols):
  importance =pd.Series(
      model.feature_importances_,
      index=feature_cols
  ).sort_values(ascending=False)
  print(importance)
  importance.head(10).plot(kind="barh", figsize=(8,5))
  plt.title("Top Feature Importance")
  plt.show()


#Potting the graph for regression's result
def regression_results_plot(y_test, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Actual kWh")
    plt.ylabel("Predicted kWh")
    plt.title("Actual vs Predicted Energy")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red"
    )

    plt.show()
