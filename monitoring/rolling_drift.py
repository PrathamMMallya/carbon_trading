import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import matplotlib.pyplot as plt


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
data = pd.read_csv("data/carbon_trading_dataset.csv")

# Use index as time/order
data["row_index"] = range(len(data))


# --------------------------------------------------
# Define reference data
# --------------------------------------------------
ref_size = int(len(data) * 0.3)
reference_data = data.iloc[:ref_size]


# --------------------------------------------------
# Rolling drift computation
# --------------------------------------------------
WINDOW_SIZE = 100
drift_share_list = []
window_end_points = []


for end in range(ref_size + WINDOW_SIZE, len(data), WINDOW_SIZE):
    current_window = data.iloc[end - WINDOW_SIZE:end]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_window)

    result = report.as_dict()
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

    drift_share_list.append(drift_share)
    window_end_points.append(end)


# --------------------------------------------------
# Plot drift evolution
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(window_end_points, drift_share_list, marker="o")
plt.axhline(y=0.3, color="red", linestyle="--", label="Drift Threshold")
plt.xlabel("Datapoint Index")
plt.ylabel("Share of Drifted Features")
plt.title("Data Drift Progression Over Datapoints")
plt.legend()
plt.tight_layout()
plt.show()
