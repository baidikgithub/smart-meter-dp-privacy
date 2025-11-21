import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Load your noise experiment CSV
df = pd.read_csv("results/noise_experiments_grouped.csv")

# Plot for first 10 data points
# plt.figure(figsize=(10, 5))
# plt.plot(df['original'][:10], label='Original', marker='o', color='blue', alpha=0.7)
# plt.plot(df['noisy'][:10], label='Noisy', marker='x', color='red', alpha=0.7)
# plt.title("Original vs Noisy (First 10 Data Points)")
# plt.xlabel("Data Index")
# plt.ylabel("t_kWh")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



plt.figure(figsize=(8, 5))
sns.histplot(df['abs_error'], bins=30, kde=True, color='teal')
plt.title("Distribution of Absolute Error")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

