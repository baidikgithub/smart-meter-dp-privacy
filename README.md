Protecting Smart Meter Data Using Differential Privacy

Overview
This project applies Differential Privacy techniques using AI to protect sensitive details in smart meter electricity consumption data, especially for industrial settings. By adding controlled noise in the machine learning training process, the solution balances privacy protection with preserving meaningful energy usage trends.

Features
1) Utilizes Differentially Private Stochastic Gradient Descent (DP-SGD) for training.
2) Neural network autoencoder model for smart meter data.
3) Tracks original and noisy consumption values with audited noise injections.
4) Maintains overall accuracy while protecting individual operational details.
5) Suitable for deployment in factories and industrial environments.

3.	IMPLEMENTATION DETAILS

1. Data Collection & Processing
●	Load datasets (Kaggle, CEEW Bareilly 2020 Smart Meter Data).
●	Imported core libraries (os, pandas, numpy) for structured data processing.
●	Sorted timestamp fields to maintain proper temporal order.

2.  Privacy-Preserving Model
●	Develop a Neural Network Autoencoder integrated with Differential Privacy (DP-SGD).
●	For each training batch:
    ○ Compute gradients.
    ○ Clip gradients to limit the influence of individual readings.
    ○ Add Gaussian noise to preserve privacy.
●	Added two AI-aligned noise tracking features:
    ○ t_kWh_noisy → the DP consumption value.
    ○ t_kWh_noise_added → the exact injected noise for transparency and auditability.
●	The system outputs a rich CSV containing:
    ● original t_kWh
    ● noisy t_kWh
    ● added noise
    ● absolute error
    ● noise multiplier

3.  Testing and Evaluation
●	Final Comparison Test: Compare anomaly detection and accuracy with the baseline.
●	Key metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE).
●	Plot privacy-utility tradeoff graphs.


Algorithm Steps:

START

1. Read the raw smart meter dataset D.

2. Preprocess the dataset:
   a. Normalize column names.
   b. Convert timestamp field to datetime format.
   c. Sort readings in chronological order.
   d. Group the dataset into blocks of k datapoints each.
      For each group:
         t_kWh_original = SUM(t_kWh of all k datapoints)

3. Prepare input features for training the autoencoder (AE).

4. Train the Autoencoder using Differentially Private SGD (DP-SGD):
   a. Set gradient clipping norm C.
   b. For each training batch:
      i.   Compute per-sample gradients g_i.
      ii.  Clip gradients: g_i = g_i / max(1, ||g_i|| / C)
      iii. Add Gaussian noise: g_noisy = g_i + N(0, (σ * C)^2)
      iv.  Update AE model parameters using g_noisy.
   c. Use privacy accountant to compute privacy budget ε.

5. For each noise multiplier σ in σ_list:
   a. Set noise sensitivity S.
   b. For each grouped datapoint:
      i. Generate Gaussian noise: noise_value = N(0, (σ * S)^2)
      ii. Compute noisy energy value: t_kWh_noisy = t_kWh_original + noise_value
      iii. Compute absolute deviation: abs_error = |t_kWh_original - t_kWh_noisy|

6. Store results:
   - t_kWh_original
   - t_kWh_noisy
   - noise_value
   - abs_error
   - σ

STOP

![Evaluation using Mean Absolute Error(MAE)
](images/flowchart.png)

Result and Discussion:

| Original | Noisy  | Noise Added | Abs Error | Noise Multiplier |
| -------- | ------ | ----------- | --------- | ---------------- |
| 0.126    | 0.1279 | 0.00196     | 0.0019    | 0.34             |
| 0.08     | 0.0794 | -0.00054    | 0.0005    | 0.34             |
| 0.079    | 0.0815 | 0.0025      | 0.0025    | 0.34             |
| 0.1600   | 0.1660 | 0.0060      | 0.0060    | 0.34             |
| 0.038    | 0.0370 | -0.0009     | 0.0009    | 0.34             |

Table: Sample of the experiment output file capturing all noise variations


For 5 data points, sum comparison:
Original = 0.483
Noisy = 0.4918

![Plotted a comparison graph showing original vs. noise-added t_kWh values
](images/Comparison.png)

![Higher noise multipliers introduce larger errors, confirming the trade-off between privacy strength and data accuracy.
](images/mse.png)

![Evaluation using Mean Absolute Error(MAE)
](images/mae.png)


