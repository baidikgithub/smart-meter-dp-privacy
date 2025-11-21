# import pandas as pd
# import numpy as np

# def calculate_rmse(original, noisy):
#     """
#     Computes Root Mean Squared Error between original and noisy values.
#     """
#     original = np.array(original)
#     noisy = np.array(noisy)

#     rmse = np.sqrt(np.mean((original - noisy) ** 2))
#     return rmse


# # ===== Load your experiment CSV =====
# csv_path = "results/noise_experiments_grouped.csv"
# df = pd.read_csv(csv_path)

# # Select one noise multiplier (e.g., the first one)
# noise_level = df["noise_multiplier"].unique()[0]
# df_subset = df[df["noise_multiplier"] == noise_level]

# # Take first 10 datapoints
# df_10 = df_subset.head(10)

# # Calculate RMSE
# rmse_value = calculate_rmse(df_10["original"], df_10["noisy"])

# print(f"RMSE for first 10 points (noise σ = {noise_level}): {rmse_value:.6f}")
# import pandas as pd
# import numpy as np

# def calculate_mae(original, noisy):
#     """
#     Computes Mean Absolute Error between original and noisy values.
#     """
#     original = np.array(original)
#     noisy = np.array(noisy)

#     mae = np.mean(np.abs(original - noisy))
#     return mae


# # ===== Load your experiment CSV =====
# csv_path = "results/noise_experiments_grouped.csv"
# df = pd.read_csv(csv_path)

# # Select one noise multiplier (e.g., the first available)
# noise_level = df["noise_multiplier"].unique()[0]
# df_subset = df[df["noise_multiplier"] == noise_level]

# # Take first 10 datapoints
# df_10 = df_subset.head(10)

# # Calculate MAE
# mae_value = calculate_mae(df_10["original"], df_10["noisy"])

# print(f"MAE for first 10 points (noise σ = {noise_level}): {mae_value:.6f}")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # ===== Load your experiment CSV =====
# csv_path = "results/noise_experiments_grouped.csv"
# df = pd.read_csv(csv_path)

# # Choose one noise multiplier (first available)
# noise_level = df["noise_multiplier"].unique()[0]
# df_subset = df[df["noise_multiplier"] == noise_level]

# # Take first 10 datapoints
# df_10 = df_subset.head(10)

# # Compute absolute error
# abs_error = np.abs(df_10["original"] - df_10["noisy"])

# # ====== Plot ======
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, 11), abs_error, marker='o', linewidth=2)
# plt.title(f"Absolute Error for First 10 Data Points (noise σ = {noise_level})")
# plt.xlabel("Data Point Index")
# plt.ylabel("Absolute Error")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import pearsonr

# def plot_pearson_graph(csv_path):
#     # Load results
#     df = pd.read_csv(csv_path)

#     original = df["original"]
#     noisy = df["noisy"]

#     # Compute Pearson Correlation
#     corr, p_value = pearsonr(original, noisy)

#     # Scatter Plot
#     plt.figure(figsize=(7, 5))
#     plt.scatter(original, noisy, alpha=0.5, label='Data Points')

#     # Best-fit line
#     m, b = np.polyfit(original, noisy, 1)
#     line = m * np.array(original) + b
#     plt.plot(original, line, label=f'Best Fit Line', linewidth=2)

#     # Labels and title
#     plt.title(f"Pearson Correlation Graph\nCorrelation = {corr:.4f} | p-value = {p_value:.4e}")
#     plt.xlabel("Original t_kWh")
#     plt.ylabel("Noisy t_kWh")
#     plt.legend()
#     plt.grid(True)

#     plt.show()


# # -------- Example Usage ----------
# plot_pearson_graph("results/noise_experiments_grouped.csv")


