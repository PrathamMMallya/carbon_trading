import pandas as pd
import numpy as np

def load_and_prepare(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[["Date"] + numeric_cols]

    df = df.set_index("Date").resample("M").mean().reset_index()

    exogs = [
        "Energy_Demand_MWh",
        "Emission_Produced_tCO2",
        "Emission_Allowance_tCO2",
        "Compliance_Cost_USD",
        "Carbon_Cost_Savings_USD"
    ]

    df[exogs] = df[exogs].ffill().bfill()

    return df
