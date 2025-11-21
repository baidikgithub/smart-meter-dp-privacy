import pandas as pd
import numpy as np
import os

def add_grouped_noise(df, feature_cols, group_size, noise):
    """Add noise to groups of data and return a dataframe with noisy values."""
    noisy_df = df.copy()

    for col in feature_cols:
        # Ensure column is numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Group every 'group_size' rows
        groups = [df[col].iloc[i:i + group_size] for i in range(0, len(df), group_size)]
        noisy_values = []
        group_ids = []

        for gid, group in enumerate(groups):
            avg_value = group.mean()
            noisy_avg = avg_value + np.random.normal(0, noise)
            noisy_values.extend([noisy_avg] * len(group))
            group_ids.extend([gid] * len(group))

        noisy_df[col] = noisy_values[:len(df)]
        noisy_df[f'{col}_group_id'] = group_ids[:len(df)]

    return noisy_df


def generate_noise_experiment_data(train_path, feature_cols, noise_multipliers, group_size, output_csv):
    """Generate noisy data for different noise multipliers."""
    df = pd.read_csv(train_path)
    results = []

    for noise in noise_multipliers:
        noisy_df = add_grouped_noise(df, feature_cols, group_size, noise)

        for col in feature_cols:
            abs_error = np.abs(df[col] - noisy_df[col])

            result = pd.DataFrame({
                'noise_multiplier': noise,
                'original': df[col],
                'noisy': noisy_df[col],
                'abs_error': abs_error,
                'group_id': noisy_df[f'{col}_group_id']
            })

            results.append(result)

    result_df = pd.concat(results, ignore_index=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Noise experiment data saved to: {output_csv} ({len(result_df)} rows)")
    return output_csv
