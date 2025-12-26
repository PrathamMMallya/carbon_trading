import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
data = pd.read_csv("data/carbon_trading_dataset.csv")


# --------------------------------------------------
# 2. Split into reference and current data
# --------------------------------------------------
split_index = int(len(data) * 0.7)

reference_data = data.iloc[:split_index]
current_data = data.iloc[split_index:]


# --------------------------------------------------
# 3. Run Evidently Data Drift Report
# --------------------------------------------------
report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

report.save_html("reports/data_drift_report.html")


# --------------------------------------------------
# 4. Extract drift results
# --------------------------------------------------
result = report.as_dict()

dataset_drift = result["metrics"][0]["result"]["dataset_drift"]
drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

print("Dataset Drift Detected:", dataset_drift)
print("Drifted Feature Share:", drift_share)


# --------------------------------------------------
# 5. Drift decision & retraining trigger
# --------------------------------------------------
DRIFT_THRESHOLD = 0.3

if dataset_drift or drift_share > DRIFT_THRESHOLD:
    print("⚠️ Drift detected → Retraining ARIMA model")

    # Trigger existing training pipeline
    os.system("python training/train_arima.py")

else:
    print("✅ No significant drift → No retraining required")
