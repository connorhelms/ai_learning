# Take a list of 10 exam scores (e.g., [60, 75, 80, 82, 85, 88, 90, 92, 95, 100]).
# Calculate the mean, median, and mode of these scores using Python/NumPy/SciPy.
# Calculate the variance and standard deviation.
# Find the 25th, 50th (median), and 75th percentiles.
# Create a Matplotlib line plot of the function y=x^3−2x^2+5 for x values from -5 to 5. Add a title and labels.
# Generate 500 random numbers from a uniform distribution between 0 and 1 (use np.random.rand(500)). Create a Matplotlib histogram to visualize their distribution. Experiment with the number of bins.
# Using the tips dataset from Seaborn (sns.load_dataset('tips')):
# Create a Seaborn scatter plot showing the relationship between total_bill and size (number of people in the party).
# Create a Seaborn histplot for the tip amount. Add a KDE overlay.
# Create a Seaborn boxplot to compare the distribution of total_bill for smokers vs. non-smokers.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Calculate the mean, median, and mode of these scores

scores = [60, 75, 80, 82, 85, 88, 90, 92, 95, 100]

mean = np.mean(scores)
median = np.median(scores)
mode = stats.mode(scores).mode

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")

# 2. Calculate the variance and standard deviation

variance = np.var(scores)
std_dev = np.std(scores)

print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")

# 3. Find the 25th, 50th (median), and 75th percentiles

percentiles = np.percentile(scores, [25, 50, 75])
print(f"25th Percentile: {percentiles[0]}")
print(f"50th Percentile (Median): {percentiles[1]}")
print(f"75th Percentile: {percentiles[2]}")

# 4. Create a Matplotlib line plot of the function y=x^3−2x^2+5 for x values from -5 to 5

x = np.linspace(-5, 5, 400)
y = x**3 - 2*x**2 + 5

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='y = x^3 - 2x^2 + 5', color='blue')
plt.title('Graph of y = x^3 - 2x^2 + 5')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 5. Generate 500 random numbers from a uniform distribution between 0 and 1

random_numbers = np.random.rand(500)

plt.figure(figsize=(10, 6))
plt.hist(random_numbers, bins=20, density=True, alpha=0.7, color='green')
plt.title('Histogram of Uniform Distribution')
plt.xlabel('Random Numbers')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# 6. Using the tips dataset from Seaborn (sns.load_dataset('tips')):

tips = sns.load_dataset('tips')

# Create a Seaborn scatter plot showing the relationship between total_bill and size

plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='size', data=tips, hue='sex', style='smoker', size='tip')
plt.title('Scatter Plot of Total Bill vs. Size')
plt.xlabel('Total Bill')
plt.ylabel('Size')
plt.legend()
plt.grid(True)
plt.show()

# Create a Seaborn histplot for the tip amount

plt.figure(figsize=(10, 6))
sns.histplot(tips['tip'], bins=20, kde=True, color='purple')
plt.title('Histogram of Tip Amounts')
plt.xlabel('Tip Amount')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Create a Seaborn boxplot to compare the distribution of total_bill for smokers vs. non-smokers

plt.figure(figsize=(10, 6))
sns.boxplot(x='smoker', y='total_bill', data=tips, palette='Set2')
plt.title('Boxplot of Total Bill by Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Total Bill')
plt.legend()
plt.grid(True)
plt.show()
