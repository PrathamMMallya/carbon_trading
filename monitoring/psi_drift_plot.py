import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_psi(expected, actual, bins=10):
    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)

    psi = np.sum(
        (expected_perc - actual_perc) *
        np.log((expected_perc + 1e-6) / (actual_perc + 1e-6))
    )
    return psi


data = pd.read_csv("data/carbon_trading_dataset.csv")

split = int(len(data) * 0.7)
reference = data.iloc[:split]
current = data.iloc[split:]

numeric_cols = data.select_dtypes(include="number").columns

psi_scores = {}

for col in numeric_cols:
    psi_scores[col] = calculate_psi(reference[col], current[col])

plt.figure(figsize=(10, 5))
plt.bar(psi_scores.keys(), psi_scores.values())
plt.axhline(0.2, color="orange", linestyle="--", label="Moderate Drift")
plt.axhline(0.3, color="red", linestyle="--", label="High Drift")
plt.xticks(rotation=45)
plt.title("PSI-Based Data Drift per Feature")
plt.legend()
plt.tight_layout()
plt.show()
