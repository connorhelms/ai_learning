import numpy as np
import pandas as pd
from scipy import stats # Often used for statistical functions

data = np.array([10, 15, 15, 18, 20, 22, 25, 25, 25, 30, 32, 35, 40, 45, 50])
data_series = pd.Series(data)

# --- Descriptive Statistics ---
print("--- Descriptive Statistics ---")
#Measure of Central Tendency
mean_val = np.mean(data)
median_val = np.median(data)
mode_val_scipy = stats.mode(data, keepdims=False)
# Pandas Series also has a mode method which can return multiple modes
mode_val_pandas = data_series.mode()

print(f"Data: {data}")
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Mode (Scipy): {mode_val_scipy}, (Count: {mode_val_scipy.count}")
print(f"Mode (Pandas): {mode_val_pandas}")
print("---"*30)

# Measures of Dispersion/Variability
variance_val = np.var(data)
std_dev_val = np.std(data)
# For sample variance/std dev (ddof=1):
# variance_sample = np.var(data, ddof=1)
# std_dev_sample = np.std(data, ddof=1)

range_val = np.max(data) - np.min(data)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

print(f"Variance: {variance_val:.2f}")
print(f"Standard Deviation: {std_dev_val:.2f}")
print(f"Range: {range_val}")
print(f"25th Percentile (Q1): {q1}")
print(f"75th Percentile (Q3): {q3}")
print(f"Interquartile Range (IQR): {iqr}")
print("---"*30)

# Pandas describe() is very handy
print("\n--- Pandas Describe ---")
print(data_series.describe())
print("---"*30)

# --- Basic Probability (Conceptual) ---
# Example: Rolling a fair six-sided die
sample_space_size = 6
event_A_outcomes = 1
prob_A = event_A_outcomes / sample_space_size
print(f"\n--- Basic Probability Example ---")
print(f"Probability of rolling a 3 on a fair die: {prob_A:.2f}")

# Event B: Rolling an even number (2, 4, 6)
even_B_outcomes = 3
prob_B = even_B_outcomes / sample_space_size
print(f"Probability of rollin an even number: {prob_B:.2f}")

# --- Normal Distribution (Conceptual) ---
# We often work with data that is assumed to be normally distributed.
# Generate some normally distributed data for visualization later
np.random.seed(42) #for reproducibility
normal_data = np.random.normal(loc=0, scale=1, size=1000)
print(f"First 10 samples from a generated normal distrubition: {normal_data[:10]}")


