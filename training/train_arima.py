# train_arima.py this is having ARIMA + MLflow + Comet + WandB (FULL VERSION) 


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import matplotlib.pyplot as plt
import json
import psutil
import time

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# this is comet part

from comet_ml import Experiment

with open("../comet-config/comet.json", "r") as f:
    comet_cfg = json.load(f)

experiment = Experiment(
    api_key=comet_cfg["api_key"],
    project_name=comet_cfg["project_name"],
    workspace=comet_cfg["workspace"]
)

# this one is for wandb-

import wandb
wandb.init(
    project="carbon-price-forecasting",
    name="ARIMA_Run",
    config={"model": "ARIMA"}
)

# ml flow 
import mlflow
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("Carbon_Forecasting_ARIMA")

file_path = "../data/carbon_trading_dataset.csv"
df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[["Date"] + list(numeric_cols)]

df = df.set_index("Date").resample("M")[numeric_cols].mean().reset_index()

target = "Carbon_Price_USD_per_t"
y = df[target]

split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]
y_train, y_test = train[target], test[target]

# for metrics... 
def calculate_metrics(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    return rmse, mae, mse

# for training
p, d, q = 2, 1, 2

with mlflow.start_run(run_name="ARIMA"):

    # Train Model
    model = ARIMA(y_train, order=(p, d, q)).fit()

    #  Forecast 
    y_pred = model.forecast(len(test))
    rmse, mae, mse = calculate_metrics(y_test, y_pred)

    # Log Parameters
    mlflow.log_params({"p": p, "d": d, "q": q})
    experiment.log_parameters({"p": p, "d": d, "q": q})
    wandb.config.update({"p": p, "d": d, "q": q})

    # Log Final Metrics
    mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "MSE": mse})
    experiment.log_metrics({"RMSE": rmse, "MAE": mae, "MSE": mse})
    wandb.log({
        "Final_RMSE": rmse,
        "Final_MAE": mae,
        "Final_MSE": mse
    })

    # this is STEP-WISE METRICS OVER TIME (GRAPHS) 
    for i in range(1, len(y_pred) + 1):
        rmse_i, mae_i, mse_i = calculate_metrics(
            y_test.iloc[:i],
            y_pred.iloc[:i]
        )

        wandb.log({
            "RMSE_over_time": rmse_i,
            "MAE_over_time": mae_i,
            "MSE_over_time": mse_i
        }, step=i)

        experiment.log_metric("RMSE_over_time", rmse_i, step=i)
        experiment.log_metric("MAE_over_time", mae_i, step=i)
        experiment.log_metric("MSE_over_time", mse_i, step=i)

    #  SYSTEM RESOURCE UTILIZATION is this
    for step in range(20):
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent

        wandb.log({
            "CPU_Usage_%": cpu,
            "RAM_Usage_%": ram
        }, step=step)

        experiment.log_metric("CPU_Usage_%", cpu, step=step)
        experiment.log_metric("RAM_Usage_%", ram, step=step)

        time.sleep(0.4)


    MODEL_DIR = r"C:\Users\prath\Downloads\MLOps_Backend\models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "arima.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

# PLOT RESULT for this model...
plt.figure(figsize=(12, 6))
plt.plot(train["Date"], y_train, label="Train", color="black")
plt.plot(test["Date"], y_test, label="Actual", color="gray")
plt.plot(test["Date"], y_pred, label="Forecast", color="red")
plt.legend()
plt.title("ARIMA Forecast")
plt.tight_layout()

os.makedirs("../outputs", exist_ok=True)
plot_path = "../outputs/arima_plot.png"
plt.savefig(plot_path)

experiment.log_figure(figure=plt)
wandb.log({"Forecast_Plot": wandb.Image(plot_path)})

plt.close()

print("\n============================")
print("ARIMA TRAINING COMPLETE")
print("============================")
