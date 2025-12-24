from flask import Flask, request, jsonify, render_template
import joblib, json, os, psutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
from datetime import datetime
from comet_ml import Experiment
import mlflow
import wandb

# Flask app
app = Flask(__name__, template_folder="frontend", static_folder="frontend")

# Paths
ARTIFACT_DIR = "artifacts"
HISTORY_FILE = f"{ARTIFACT_DIR}/inference_history.csv"
PLOT_FILE = f"{ARTIFACT_DIR}/forecast_plot.png"
SYS_PLOT_FILE = f"{ARTIFACT_DIR}/system_model_plot.png"
FEATURE_PLOT_FILE = f"{ARTIFACT_DIR}/feature_importance.png"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Carbon_Forecasting_Inference")

# Comet
with open("./comet-config/comet.json") as f:
    comet_cfg = json.load(f)
experiment = Experiment(
    api_key=comet_cfg["api_key"],
    project_name=comet_cfg["project_name"],
    workspace=comet_cfg["workspace"],
    auto_output_logging="full"
)

# W&B
wandb.init(project="carbon-price-forecasting", name="Inference_Run", reinit=True)

# Load models
MODELS = {
    "arima": joblib.load("models/arima.pkl"),
    "prophet": joblib.load("models/prophet.pkl"),
    "rf": joblib.load("models/rf.pkl")
}

# Utilities
def system_stats():
    return psutil.cpu_percent(), psutil.virtual_memory().percent

def auto_select_model():
    cpu, ram = system_stats()
    return "arima" if (cpu > 75 or ram > 75) else "prophet"

def clamp_preds(preds, low=25, high=28):
    return [max(low, min(high, float(p))) for p in preds]

def generate_forecast_plot(df_hist, run_label):
    plt.figure(figsize=(8,5))
    for label, grp in df_hist.groupby("run_label"):
        plt.plot(grp["step"], grp["prediction"], marker="o", label=label)
    plt.ylim(25,28)
    plt.xlabel("Forecast Step")
    plt.ylabel("Carbon Price (USD/t)")
    plt.title("Carbon Price Forecast (Next 5 Steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()
    # Log to W&B
    wandb.log({"forecast_plot": wandb.Image(PLOT_FILE, caption=run_label)})
    
def generate_system_model_plot(df_hist, run_label):
    plt.figure(figsize=(8,5))
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

    wandb.log({"system_metrics": wandb.Image(SYS_PLOT_FILE, caption=run_label)})

def generate_rf_feature_plot(model, run_label):
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(6,4))
        plt.bar(["year","month"], model.feature_importances_)
        plt.ylabel("Feature Importance")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig(FEATURE_PLOT_FILE)
        plt.close()

        wandb.log({"rf_feature_importance": wandb.Image(FEATURE_PLOT_FILE, caption=run_label)})


# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Validate date
    if "date" not in data or not data["date"]:
        return jsonify({"error": "Date is required"}), 400

    try:
        date = pd.to_datetime(data["date"])
    except Exception:
        return jsonify({"error": "Invalid date format"}), 400

    mode = data.get("mode", "manual")
    model_name = auto_select_model() if mode=="auto" else data.get("model", "arima")

    if model_name not in MODELS:
        return jsonify({"error": f"Model {model_name} not found"}), 400

    model = MODELS[model_name]
    cpu, ram = system_stats()
    run_label = f"{model_name}_{datetime.now().strftime('%H:%M:%S')}"

    try:
        if model_name=="arima":
            preds = model.forecast(steps=5).tolist()
        else:
            future = pd.date_range(date, periods=5, freq="M")
            if model_name=="prophet":
                preds = model.predict(pd.DataFrame({"ds": future}))["yhat"].tolist()
            elif model_name=="rf":
                X = np.column_stack((future.year, future.month))
                preds = model.predict(X).tolist()
                generate_rf_feature_plot(model,run_label)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    preds = clamp_preds(preds)

    # Save history
    hist = pd.DataFrame({
        "run_label": run_label,
        "step": range(1,6),
        "prediction": preds,
        "CPU":[cpu]*5,
        "RAM":[ram]*5
    })
    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
    else:
        df_hist = pd.DataFrame(columns=hist.columns)

    run_count = df_hist['run_label'].nunique()
    if run_count >= 3:
        df_hist = hist.copy()
    else:
        df_hist = pd.concat([df_hist, hist], ignore_index=True)

    df_hist.to_csv(HISTORY_FILE, index=False)

    # Generate plots
    generate_forecast_plot(df_hist,run_label)
    generate_system_model_plot(df_hist,run_label)

    return jsonify({
        "model_used": model_name,
        "cpu_usage": cpu,
        "ram_usage": ram,
        "next_5_predictions": preds
    })

if __name__=="__main__":
    app.run(debug=True, use_reloader=False)
