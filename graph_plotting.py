import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

"""
This code will generate a line graph of the median with Q1 and Q3.
The median is calculated based on the index in all lines.
Data must by in the following format:
 - Each measurement in separate lines (one line = multiple epochs)
 - Each epoche in line separeted by semicolon (;)
 - Lines do NOT have to have the same length
"""

# Step 1: Load data with variable line lengths
data_by_epoch_QL = defaultdict(list)
data_by_epoch_SARSA = defaultdict(list)

steps = 250000
trainer = "FrozenLake"

# Read the CSV file line by line
with open(f".\\results\\test\\{trainer}\\QL_avg_reward_{steps//1000}k.csv", 'r') as file:
    for line in file:
        # Split line into values and convert to float
        values = [float(x) for x in line.strip().split(';') if x]
        # Append each value to the corresponding epoch in data_by_epoch
        for i, value in enumerate(values):
            data_by_epoch_QL[i].append(value)

with open(f".\\results\\test\\{trainer}\\SARSA_avg_reward_{steps//1000}k.csv", 'r') as file:
    for line in file:
        # Split line into values and convert to float
        values = [float(x) for x in line.strip().split(';') if x]
        # Append each value to the corresponding epoch in data_by_epoch
        for i, value in enumerate(values):
            data_by_epoch_SARSA[i].append(value)


# Step 2: Calculate median, Q1, and Q3 for each index
epochs = sorted(data_by_epoch_QL.keys())
medians = []
q1_values = []
q3_values = []
std = []


epochs_SARSA = sorted(data_by_epoch_SARSA.keys())
medians_SARSA = []
q1_values_SARSA = []
q3_values_SARSA = []
std_SARSA = []

for epoch in epochs:
    values_at_index = data_by_epoch_QL[epoch]
    medians.append(np.median(values_at_index))
    q1_values.append(np.percentile(values_at_index, 25))
    q3_values.append(np.percentile(values_at_index, 75))
    std.append(np.std(values_at_index))

for epoch in epochs_SARSA:
    values_at_index = data_by_epoch_SARSA[epoch]
    medians_SARSA.append(np.median(values_at_index))
    q1_values_SARSA.append(np.percentile(values_at_index, 25))
    q3_values_SARSA.append(np.percentile(values_at_index, 75))
    std_SARSA.append(np.std(values_at_index))

# Step 3: Plot the Median with IQR (Q1-Q3) as a shaded area
plt.figure(figsize=(10, 6))

# Median line
plt.plot(epochs, medians, label="QL - Median", color="red", linewidth=2)
plt.plot(epochs_SARSA, medians_SARSA, label="SARSA - Median", color="blue", linewidth=2)

medians = np.array(medians)
std = np.array(std)
medians_SARSA = np.array(medians_SARSA)
SARSA_std = np.array(std_SARSA)

# Shaded area for IQR (Q1 to Q3) or STD
plt.fill_between(epochs, (medians - std), (medians + std), color="orange", alpha=0.4, label="QL - Std Dev")
plt.fill_between(epochs_SARSA, medians_SARSA - SARSA_std, medians_SARSA + SARSA_std, color="lightblue", alpha=0.4, label="SARSA - Std Dev")

# plt.fill_between(epochs, q1_values, q3_values, color="orange", alpha=0.4, label="QL - IQR (Q1 - Q3)")
# plt.fill_between(epochs_SARSA, q1_values_SARSA, q3_values_SARSA, color="lightblue", alpha=0.4, label="SARSA - IQR (Q1 - Q3)")

# Customize the plot
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Accumulated reward")
# This need to be reworked, right now it is manual
plt.title(f"{trainer}\nAverage accumulated reward\nalpha=0.1 epsilon=0.1 gama=0.99 steps={steps}")
plt.legend()
plt.show()
