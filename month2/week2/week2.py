import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics # For evaluation


# --- 1. Generate or Load Sample Data ---
# Let's create some simple data: hours_studied vs. exam_score
np.random.seed(0) # for reproducibility
hours_studied = np.array([1, 2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 7, 8, 8.5, 9, 10]).reshape(-1, 1)
#score = 50 + 5*hours + noise
exam_score = 50 + 5 * hours_studied + np.random.normal(0, 5, size=hours_studied.shape)
exam_score = exam_score.reshape(-1, 1)

#Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(hours_studied, exam_score, color='blue', label='Actual Data Points')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs Exam Score')
plt.legend()
plt.show()

