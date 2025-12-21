# ================================================================
# app.py — Carbon Price Forecasting Inference (PRODUCTION FULL TELEMETRY)
# ================================================================

from comet_ml import Experiment
import mlflow
import wandb

from flask import Flask, request, jsonify, render_template
import joblib, json, psutil, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Flask ----------------
app = Flask(__name__, template_folder="frontend")

# ---------------- Paths ----------------
ARTIFACT_DIR = "artifacts"
HISTORY_FILE = f"{ARTIFACT_DIR}/inference_history.csv"
PLOT_FILE = f"{ARTIFACT_DIR}/forecast_plot.png"
SYS_PLOT_FILE = f"{ARTIFACT_DIR}/system_model_plot.png"
FEATURE_PLOT_FILE = f"{ARTIFACT_DIR}/feature_importance.png"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- MLflow ----------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Carbon_Forecasting_Inference")

# ---------------- Comet ----------------
with open("./comet-config/comet.json") as f:
    comet_cfg = json.load(f)

experiment = Experiment(
    api_key=comet_cfg["api_key"],
    project_name=comet_cfg["project_name"],
    workspace=comet_cfg["workspace"],
    auto_output_logging="full"
)

# ---------------- W&B ----------------
wandb.init(
    project="carbon-price-forecasting",
    name="Inference_Run",
    reinit=True
)

# ---------------- Load Models ----------------
MODELS = {
    "arima": joblib.load("models/arima.pkl"),
    "prophet": joblib.load("models/prophet.pkl"),
    "rf": joblib.load("models/rf.pkl")
}

# ---------------- Utilities ----------------
def system_stats():
    return psutil.cpu_percent(), psutil.virtual_memory().percent

def auto_select_model():
    cpu, ram = system_stats()
    return "arima" if (cpu > 75 or ram > 75) else "prophet"

def clamp_preds(preds, low=25, high=28):
    return [max(low, min(high, float(p))) for p in preds]

# ---------------- Plot Functions ----------------
def generate_forecast_plot(df_hist):
    plt.figure(figsize=(8, 5))
    for label, grp in df_hist.groupby("run_label"):
        plt.plot(grp["step"], grp["prediction"], marker="o", label=label)
    plt.ylim(25, 28)
    plt.xlabel("Forecast Step")
    plt.ylabel("Carbon Price (USD/t)")
    plt.title("Carbon Price Forecast (Next 5 Steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()

def generate_system_model_plot(df_hist):
    plt.figure(figsize=(8, 5))
    for label, grp in df_hist.groupby("run_label"):
        plt.plot(grp["step"], grp["CPU"], marker="x", label=f"{label} CPU")
        plt.plot(grp["step"], grp["RAM"], marker="^", label=f"{label} RAM")
    plt.xlabel("Forecast Step")
    plt.ylabel("System Usage (%)")
    plt.title("System Metrics per Run")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SYS_PLOT_FILE)
    plt.close()

def generate_rf_feature_plot(model):
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(6,4))
        plt.bar(["year","month"], model.feature_importances_)
        plt.ylabel("Feature Importance")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig(FEATURE_PLOT_FILE)
        plt.close()

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    date = pd.to_datetime(data["date"])
    mode = data["mode"]

    model_name = auto_select_model() if mode == "auto" else data["model"]
    model = MODELS[model_name]

    cpu, ram = system_stats()
    run_label = f"{model_name}_{datetime.now().strftime('%H:%M:%S')}"

    with mlflow.start_run(run_name=f"Inference_{model_name}"):
        # ---------------- Prediction ----------------
        if model_name == "arima":
            preds = model.forecast(steps=5).tolist()
        elif model_name == "prophet":
            future = pd.date_range(date, periods=5, freq="M")
            preds = model.predict(pd.DataFrame({"ds": future}))["yhat"].tolist()
        elif model_name == "rf":
            future = pd.date_range(date, periods=5, freq="M")
            X = np.column_stack((future.year, future.month))
            preds = model.predict(X).tolist()
            # Log RF hyperparameters & feature importances
            mlflow.log_params({
                "model_type": "RandomForest",
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth
            })
            experiment.log_parameters({
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth
            })
            generate_rf_feature_plot(model)
            experiment.log_image(FEATURE_PLOT_FILE, name="RF_Feature_Importance")
            wandb.log({"RF_Feature_Importance": wandb.Image(FEATURE_PLOT_FILE)})

        preds = clamp_preds(preds)

        # ---------------- History Management ----------------
        hist = pd.DataFrame({
            "run_label": run_label,
            "step": range(1,6),
            "prediction": preds,
            "CPU": [cpu]*5,
            "RAM": [ram]*5
        })

        # Safe CSV load
        if os.path.exists(HISTORY_FILE):
            try:
                df_hist = pd.read_csv(HISTORY_FILE)
                for col in ["CPU","RAM"]:
                    if col not in df_hist.columns:
                        df_hist[col] = 0
            except pd.errors.ParserError:
                df_hist = pd.DataFrame(columns=["run_label","step","prediction","CPU","RAM"])
        else:
            df_hist = pd.DataFrame(columns=["run_label","step","prediction","CPU","RAM"])

        # Roll-over after 3 runs
        run_count = df_hist['run_label'].nunique()
        if run_count >= 3:
            df_hist = hist.copy()
        else:
            df_hist = pd.concat([df_hist, hist], ignore_index=True)

        df_hist.to_csv(HISTORY_FILE, index=False)

        # ---------------- Generate Plots ----------------
        generate_forecast_plot(df_hist)
        generate_system_model_plot(df_hist)

        # ---------------- MLflow Logging ----------------
        mlflow.log_param("model_used", model_name)
        mlflow.log_metrics({"CPU": cpu, "RAM": ram})
        mlflow.log_artifact(PLOT_FILE)
        mlflow.log_artifact(SYS_PLOT_FILE)

        # ---------------- Comet Logging ----------------
        experiment.log_parameter("Model", model_name)
        experiment.log_metric("CPU_Usage", cpu)
        experiment.log_metric("RAM_Usage", ram)
        experiment.log_image(PLOT_FILE, name="Forecast_Comparison")
        experiment.log_image(SYS_PLOT_FILE, name="System_Metrics_Comparison")

        # ---------------- W&B Logging ----------------
        wandb.log({
            "CPU_Usage_%": cpu,
            "RAM_Usage_%": ram,
            "Forecast_Plot": wandb.Image(PLOT_FILE),
            "System_Model_Plot": wandb.Image(SYS_PLOT_FILE)
        })

    return jsonify({
        "model_used": model_name,
        "cpu_usage": cpu,
        "ram_usage": ram,
        "next_5_predictions": preds
    })

# ---------------- Windows Safe ----------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
