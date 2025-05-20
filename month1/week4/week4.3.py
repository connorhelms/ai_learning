import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data
np.random.seed(0)
normal_data_for_seaborn = np.random.normal(loc=5, scale=2, size=200)
df_tips = sns.load_dataset('tips') # Seaborn comes with some sample datasets

print(f"---Seaborn Visualization---")
print(f"Tips dataset head:\n{df_tips.head()}") # Explore the tips dataset
print("---"*30)

#--- Histogram with KDE (Kernel Density Estimate) using histplot ---
plt.figure(figsize=(8, 6))
sns.histplot(normal_data_for_seaborn, kde=True, color='purple', bins=20)
plt.title('Seaborn Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Frequency / Density')
plt.show()

#--- KDE Plot (Density Plot) ---
plt.figure(figsize=(8, 6))
sns.kdeplot(normal_data_for_seaborn, fill=True, color='skyblue', linewidth=2)
plt.title('Seaborn KDE Plot')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

#--- Box Plot (using tips dataset) ---
# Shows distribution, median, quartiles, outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x='day', y='total_bill', data=df_tips, palette='pastel')
plt.title('Box Plot of Total Bill by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill ($)')
plt.show()

#--- Violin Plot (combines box plot with KDE) ---
plt.figure(figsize=(8, 6))
sns.violinplot(x='day', y='total_bill', data=df_tips, palette='husl')
plt.title('Violin Plot of Total Bill by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill ($)')

#--- Scatter Plot with Regression Line (using lmplot or regplot) ---
plt.figure(figsize=(8, 6))
sns.regplot(x='total_bill', y='tip', data=df_tips, scatter_kws={'s':20, 'alpha':0.6}, line_kws={'color':'red'})
plt.title('Scatter Plot of Tip vs Total Bill with Regression Line')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.show()

