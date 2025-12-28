# train_rf.py — RandomForest + MLflow + Comet + WandB (FULL, FIXED)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os, json, time, joblib, psutil
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from comet_ml import Experiment
from pathlib import Path
import json

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMET_CONFIG = PROJECT_ROOT / "comet-config" / "comet.json"
DATA_DIR = PROJECT_ROOT / "data"

if os.path.exists(COMET_CONFIG):
    with open(COMET_CONFIG, "r") as f:
        comet_cfg = json.load(f)
    experiment = Experiment(
        api_key=comet_cfg.get("api_key"),
        project_name=comet_cfg.get("project_name"),
        workspace=comet_cfg.get("workspace")
    )
else:
    experiment = Experiment(api_key=os.getenv("COMET_API_KEY"))

import wandb
wandb_key = os.getenv("WANDB_API_KEY")
if not wandb_key:
    os.environ["WANDB_MODE"] = "disabled"
    print("No WANDB_API_KEY found. WandB disabled.")
else:
    wandb.login(key=wandb_key)
wandb.init(project="carbon-price-forecasting", name="RF_Run")

import mlflow
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("Carbon_Forecasting_RF")

df = pd.read_csv(DATA_DIR / "raw" / "carbon_trading_dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

df = df[["Date", "Carbon_Price_USD_per_t"]]

# SAFE MONTHLY AGGREGATION (Pandas 2.x compatible)
df = (
    df.set_index("Date")
      .resample("M")
      .mean(numeric_only=True)
      .reset_index()
)

df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month

X = df[["year", "month"]]
y = df["Carbon_Price_USD_per_t"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

def metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred)
    )

with mlflow.start_run(run_name="RandomForest"):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse, mae, mse = metrics(y_test, y_pred)

    mlflow.log_params({
        "model": "RandomForest",
        "n_estimators": 200,
        "max_depth": 10
    })
    experiment.log_parameters({
        "model": "RandomForest",
        "n_estimators": 200,
        "max_depth": 10
    })

    mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "MSE": mse})
    experiment.log_metrics({"RMSE": rmse, "MAE": mae, "MSE": mse})
    wandb.log({"Final_RMSE": rmse, "Final_MAE": mae, "Final_MSE": mse})

    for i in range(1, len(y_pred) + 1):
        rmse_i, mae_i, mse_i = metrics(y_test.iloc[:i], y_pred[:i])

        wandb.log({
            "RMSE_over_time": rmse_i,
            "MAE_over_time": mae_i,
            "MSE_over_time": mse_i
        }, step=i)

        experiment.log_metric("RMSE_over_time", rmse_i, step=i)
        experiment.log_metric("MAE_over_time", mae_i, step=i)
        experiment.log_metric("MSE_over_time", mse_i, step=i)

    for step in range(20):
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent

        wandb.log({"CPU_Usage_%": cpu, "RAM_Usage_%": ram}, step=step)
        experiment.log_metric("CPU_Usage_%", cpu, step=step)
        experiment.log_metric("RAM_Usage_%", ram, step=step)

        time.sleep(0.4)

    MODEL_DIR = PROJECT_ROOT / "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "rf.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, "model")

print(" RANDOM FOREST TRAINING COMPLETE")
