# train_prophet.py — Prophet + MLflow + Comet + WandB (FULL, FIXED)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os, json, time, joblib, psutil
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import json
# this is comet
from comet_ml import Experiment
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMET_CONFIG = PROJECT_ROOT / "comet-config" / "comet.json"
DATA_DIR = PROJECT_ROOT / "data"

with open(COMET_CONFIG, "r") as f:
    comet_cfg = json.load(f)

experiment = Experiment(
    api_key=comet_cfg["api_key"],
    project_name=comet_cfg["project_name"],
    workspace=comet_cfg["workspace"]
)

# weights and b
import wandb
wandb.init(project="carbon-price-forecasting", name="Prophet_Run")

# mflow part
import mlflow
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("Carbon_Forecasting_Prophet")

df = pd.read_csv(DATA_DIR / "carbon_trading_dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

#  KEEP ONLY NUMERIC TARGET BEFORE RESAMPLING , this is important
df = df[["Date", "Carbon_Price_USD_per_t"]]

#  SAFE RESAMPLING (Pandas 2.x compatible) or else will  get errors..
df = (
    df.set_index("Date")
      .resample("M")
      .mean(numeric_only=True)
      .reset_index()
)

# Prophet format
df = df.rename(columns={
    "Date": "ds",
    "Carbon_Price_USD_per_t": "y"
})

split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]

def metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred)
    )

# ---------------- Train ----------------
with mlflow.start_run(run_name="Prophet"):

    model = Prophet()
    model.fit(train)

    forecast = model.predict(test[["ds"]])
    y_pred = forecast["yhat"]
    y_test = test["y"]

    rmse, mae, mse = metrics(y_test, y_pred)

    # ---- Log Params ----
    mlflow.log_param("model", "Prophet")
    experiment.log_parameter("model", "Prophet")

    # ---- Log Metrics ----
    mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "MSE": mse})
    experiment.log_metrics({"RMSE": rmse, "MAE": mae, "MSE": mse})
    wandb.log({"Final_RMSE": rmse, "Final_MAE": mae, "Final_MSE": mse})

    # ---- Step-wise Metrics ----
    for i in range(1, len(y_pred) + 1):
        rmse_i, mae_i, mse_i = metrics(y_test.iloc[:i], y_pred.iloc[:i])

        wandb.log({
            "RMSE_over_time": rmse_i,
            "MAE_over_time": mae_i,
            "MSE_over_time": mse_i
        }, step=i)

        experiment.log_metric("RMSE_over_time", rmse_i, step=i)
        experiment.log_metric("MAE_over_time", mae_i, step=i)
        experiment.log_metric("MSE_over_time", mse_i, step=i)

    # ---- Resource Utilization ----
    for step in range(20):
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent

        wandb.log({"CPU_Usage_%": cpu, "RAM_Usage_%": ram}, step=step)
        experiment.log_metric("CPU_Usage_%", cpu, step=step)
        experiment.log_metric("RAM_Usage_%", ram, step=step)

        time.sleep(0.4)

    # ---- Save Model ----
    MODEL_DIR = r"C:\Users\prath\Downloads\MLOps_Backend\models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "prophet.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, "model")

# ---------------- Plot ----------------
plt.figure(figsize=(12, 6))
plt.plot(train["ds"], train["y"], label="Train")
plt.plot(test["ds"], y_test, label="Actual")
plt.plot(test["ds"], y_pred, label="Forecast")
plt.legend()
plt.title("Prophet Forecast")
plt.tight_layout()

os.makedirs("../outputs", exist_ok=True)
plot_path = "../outputs/prophet_plot.png"
plt.savefig(plot_path)

experiment.log_figure(figure=plt)
wandb.log({"Forecast_Plot": wandb.Image(plot_path)})
plt.close()

print(" PROPHET TRAINING COMPLETE")