"""
Data preprocessing for Smart Meter DP project.
"""

import os
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_NUMERIC_COLS = [
    't_kwh',
    'z_Avg Voltage (Volt)',
    'z_Avg Current (Amp)',
    'y_Freq (Hz)'
]

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    ts_candidates = ['x_Timestamp', 'timestamp', 'Timestamp', 'time', 'date']
    for c in ts_candidates:
        if c in df.columns:
            df.rename(columns={c: 'timestamp'}, inplace=True)
            break

    meter_candidates = ['meter', 'meter_id', 'meterId']
    for c in meter_candidates:
        if c in df.columns:
            df.rename(columns={c: 'meter'}, inplace=True)
            break
    return df


def preprocess(df: pd.DataFrame, dropna=True, resample_rule=None, numeric_cols=None) -> Tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp').set_index('timestamp')

    if resample_rule:
        df = df.resample(resample_rule).mean()

    if numeric_cols is None:
        numeric_cols = [c for c in DEFAULT_NUMERIC_COLS if c in df.columns]

    if dropna:
        df = df.dropna()
    else:
        df = df.fillna(method='ffill').dropna()

    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['dayofweek'] = df.index.dayofweek

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if not numeric_cols:
        raise ValueError("No numeric columns found!")

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def save_processed(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)


def split_and_save(df: pd.DataFrame, output_dir: str, test_size=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(df.index, pd.DatetimeIndex):
        split_idx = int((1 - test_size) * len(df))
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    return train_path, test_path
