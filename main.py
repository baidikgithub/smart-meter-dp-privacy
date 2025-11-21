#!/usr/bin/env python3
"""
Main Script: Smart Meter DP Workflow (grouped by 12 datapoints and controlled noise added)
"""

import os
import random
import pandas as pd
import numpy as np


# ---------- Utility: Preprocess by grouping every 12 datapoints ----------
def preprocess_grouped_sum(raw_csv_path: str, out_csv_path: str, group_size: int = 12) -> str:
    """
    Reads raw smart meter CSV, combines every 'group_size' datapoints
    by summing t_kWh (energy) and averaging other sensor readings.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"❌ File not found: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)
    df.columns = [c.strip() for c in df.columns]

    ts_candidates = ["x_Timestamp", "timestamp", "Timestamp", "time"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if not ts_col:
        raise ValueError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(by=ts_col).reset_index(drop=True)

    # Group every 12 datapoints
    df["group_id"] = df.index // group_size

    grouped = (
        df.groupby("group_id")
        .agg({
            ts_col: "first",
            "t_kWh": "sum",
            "z_Avg Voltage (Volt)": "mean",
            "z_Avg Current (Amp)": "mean",
            "y_Freq (Hz)": "mean",
        })
        .reset_index(drop=True)
    )

    grouped = grouped.rename(columns={ts_col: "timestamp"})

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    grouped.to_csv(out_csv_path, index=False)
    print(f"✅ Grouped every {group_size} datapoints → {out_csv_path} (rows: {len(grouped)})")

    return out_csv_path


# ---------- Controlled Noise Addition ----------
def add_grouped_noise(df, feature_cols, noise_multiplier):
    """
    Add controlled noise such that mean(original) ≈ mean(noisy),
    and record how much noise was added for each data point.
    """
    noisy_df = df.copy()

    for col in feature_cols:
        np.random.seed(42)

        # Scale noise relative to each original value (±noise_multiplier%)
        noise = np.random.normal(
            loc=0,
            scale=noise_multiplier * df[col].mean() * 0.05,  # controlled low variation
            size=len(df)
        )

        # Apply noise
        noisy_df[f"{col}_noise_added"] = noise
        noisy_df[f"{col}_noisy"] = df[col] + noise

        # Shift noisy data to match mean of original
        mean_diff = noisy_df[f"{col}_noisy"].mean() - df[col].mean()
        noisy_df[f"{col}_noisy"] = noisy_df[f"{col}_noisy"] - mean_diff

    return noisy_df


def generate_noise_experiment_data(train_path, feature_cols, noise_multipliers, output_csv):
    """Generate noisy data for different noise multipliers."""
    df = pd.read_csv(train_path)
    results = []

    for noise in noise_multipliers:
        noisy_df = add_grouped_noise(df, feature_cols, noise)

        for col in feature_cols:
            abs_error = np.abs(df[col] - noisy_df[f"{col}_noisy"])
            result = pd.DataFrame({
                "noise_multiplier": noise,
                "original": df[col],
                "noisy": noisy_df[f"{col}_noisy"],
                "noise_added": noisy_df[f"{col}_noise_added"],
                "abs_error": abs_error
            })
            results.append(result)

    result_df = pd.concat(results, ignore_index=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Noise experiment data saved to: {output_csv} ({len(result_df)} rows)")
    return output_csv


# ---------- Main workflow ----------
def main():
    print("========== Smart Meter DP Workflow Started ==========")

    raw_path = "data/raw/smart_meter_bareilly_2021.csv"
    processed_path = "data/processed/processed_grouped.csv"
    result_csv = "results/noise_experiments_grouped.csv"

    # --- Step 1️⃣ Preprocess raw CSV ---
    processed_csv = preprocess_grouped_sum(raw_path, processed_path, group_size=12)

    # --- Step 2️⃣ Generate random noise multipliers ---
    print("[1/2] Generating random noise multipliers...")
    random.seed(42)
    noise_multipliers = sorted([round(random.uniform(0.3, 2.0), 2) for _ in range(10)])
    print(f"Noise multipliers: {noise_multipliers}")

    # --- Step 3️⃣ Run controlled noise experiments ---
    print("[2/2] Running grouped noise experiments (controlled mean-preserving noise)...")
    experiment_csv = generate_noise_experiment_data(
        train_path=processed_csv,
        feature_cols=["t_kWh"],
        noise_multipliers=noise_multipliers,
        output_csv=result_csv
    )

    print("\n✅ All experiments complete!")
    print(f"Results saved at: {experiment_csv}")
    print("========== Workflow Complete ==========")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    main()
