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
data_by_epoch = defaultdict(list)

# Read the CSV file line by line
with open('PATH_TO_CSV_FILE', 'r') as file:
    for line in file:
        # Split line into values and convert to float
        values = [float(x) for x in line.strip().split(';') if x]
        
        # Append each value to the corresponding epoch in data_by_epoch
        for i, value in enumerate(values):
            data_by_epoch[i].append(value)

# Step 2: Calculate median, Q1, and Q3 for each index
epochs = sorted(data_by_epoch.keys())
medians = []
q1_values = []
q3_values = []

for epoch in epochs:
    values_at_index = data_by_epoch[epoch]
    medians.append(np.median(values_at_index))
    q1_values.append(np.percentile(values_at_index, 25))
    q3_values.append(np.percentile(values_at_index, 75))

# Step 3: Plot the Median with IQR (Q1-Q3) as a shaded area
plt.figure(figsize=(10, 6))

# Median line
plt.plot(epochs, medians, label="Median", color="red", linewidth=2)

# Shaded area for IQR (Q1 to Q3)
plt.fill_between(epochs, q1_values, q3_values, color="orange", alpha=0.4, label="IQR (Q1 - Q3)")

# Customize the plot
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Accumulated reward")
# This need to be reworked, right now it is manual
plt.title("Q-Learning average accumulated reward\nalpha=0.1 epsilon=0.1 gama=0.99, steps=100,000")
plt.legend()
plt.show()
