import pandas as pd
import matplotlib.pyplot as plt

def plot_privacy_utility(csv_path="results/noise_experiments.csv"):
    df = pd.read_csv(csv_path)
    avg_mse = df.groupby("noise_multiplier")["mse"].mean().reset_index()

    plt.plot(avg_mse["noise_multiplier"], avg_mse["mse"], marker="o")
    plt.xlabel("Noise Multiplier")
    plt.ylabel("Mean Squared Error (Utility Loss)")
    plt.title("Privacy–Utility Tradeoff")
    plt.grid(True)
    plt.savefig("results/plots/privacy_utility.png")
    plt.show()

if __name__ == "__main__":
    plot_privacy_utility()
