# ================================================================
# train_sarimax.py — SARIMAX + MLflow + Comet + WandB (FULL, FIXED)
# ================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os, json, time, joblib, psutil
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------- COMET ----------------
from comet_ml import Experiment
with open("../comet-config/comet.json", "r") as f:
    comet_cfg = json.load(f)

experiment = Experiment(
    api_key=comet_cfg["api_key"],
    project_name=comet_cfg["project_name"],
    workspace=comet_cfg["workspace"]
)

# ---------------- W&B ----------------
import wandb
wandb.init(project="carbon-price-forecasting", name="SARIMAX_Run")

# ---------------- MLflow ----------------
import mlflow
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("Carbon_Forecasting_SARIMAX")

# ---------------- Load Data ----------------
df = pd.read_csv("../data/carbon_trading_dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# ✅ KEEP ONLY TARGET (numeric-safe)
df = df[["Date", "Carbon_Price_USD_per_t"]]

# ✅ SAFE MONTHLY RESAMPLING (Pandas 2.x compatible)
df = (
    df.set_index("Date")
      .resample("M")
      .mean(numeric_only=True)
      .reset_index()
)

y = df["Carbon_Price_USD_per_t"]

# ---------------- Train-Test Split ----------------
split = int(len(df) * 0.8)
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ---------------- Metrics ----------------
def metrics(y_true, y_pred):
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        mean_squared_error(y_true, y_pred)
    )

# ---------------- Train ----------------
with mlflow.start_run(run_name="SARIMAX"):

    model = SARIMAX(
        y_train,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    y_pred = model.forecast(steps=len(y_test))
    rmse, mae, mse = metrics(y_test, y_pred)

    # ---- Params ----
    params = {
        "p": 2, "d": 1, "q": 2,
        "P": 1, "D": 1, "Q": 1,
        "m": 12
    }
    mlflow.log_params(params)
    experiment.log_parameters(params)

    # ---- Metrics ----
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

    # ---- Resource Monitoring ----
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

    model_path = os.path.join(MODEL_DIR, "sarimax.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, "model")

# ---------------- Plot ----------------
plt.figure(figsize=(12, 6))
plt.plot(df["Date"].iloc[:split], y_train, label="Train")
plt.plot(df["Date"].iloc[split:], y_test, label="Actual")
plt.plot(df["Date"].iloc[split:], y_pred, label="Forecast")
plt.legend()
plt.title("SARIMAX Forecast")
plt.tight_layout()

os.makedirs("../outputs", exist_ok=True)
plot_path = "../outputs/sarimax_plot.png"
plt.savefig(plot_path)

experiment.log_figure(figure=plt)
wandb.log({"Forecast_Plot": wandb.Image(plot_path)})
plt.close()

print("✅ SARIMAX TRAINING COMPLETE")
