"""
Evaluate privacy-utility tradeoff
"""
import pandas as pd
import os

def evaluate_privacy_utility(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['utility'] = 1 / (1 + df['mse'])
    df['privacy'] = 1 - df['utility']

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Privacy-utility report saved at {output_csv}")
