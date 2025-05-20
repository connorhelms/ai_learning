import matplotlib.pyplot as plt
import numpy as np

# Sample data for plotting
x = np.linspace(0, 10, 100) # 100 points from 0 to 10
y_line = np.sin(x)
y_scatter = np.cos(x) + np.random.randn(100) * 0.2
hist_data = np.random.randn(1000)
bar_labels = ['A', 'B', 'C', 'D']
bar_values = [10, 25, 15, 30]

#Line Plot
plt.figure(figsize=(8, 6))
plt.plot(x, y_line, color='blue', linestyle='--', marker='o', markersize=3, label='sin(x)')
plt.title('Simple Line Plot')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.legend()
plt.grid(True)
plt.show()

#Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y_scatter, color='red', marker='x', label='cos(x) + noise')
plt.title("Simple Scatter Plot")
plt.xlabel('X-value')
plt.ylabel('Y-value')
plt.legend()
plt.grid(True)
plt.show()

#Histogram
plt.figure(figsize=(8, 6))
plt.hist(hist_data, bins=30, color='green', edgecolor='black', alpha=0.7)
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

#Bar Chart
plt.figure(figsize=(8, 6))
plt.bar(bar_labels, bar_values, color=['cyan', 'magenta', 'yellow', 'black'])
plt.title('Simple Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()

#Subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))# 2 rows, 2 columns of plots
fig.suptitle('Multiple Plots (Subplots)', fontsize=16)

axs[0, 0].plot(x, y_line, 'tab:blue')
axs[0, 0].set_title('Line Plot')

axs[0, 1].scatter(x, y_scatter, color='tab:orange')
axs[0, 1].set_title('Scatter Plot')

axs[1, 0].hist(hist_data, bins=20, color='tab:green')
axs[1, 0].set_title('Histogram')

axs[1, 1].bar(bar_labels, bar_values, color='tab:red')
axs[1, 1].set_title('Bar Chart')

for ax in axs.flat:
    ax.set_xlabel('X-data', fontsize=10)
    ax.set_ylabel('Y-data', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


