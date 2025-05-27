import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # For feature scaling, good practice
from sklearn.datasets import load_iris
from sklearn import metrics # We'll dive deeper into metrics next week

# Load the iris dataset
iris = load_iris()
x = iris.data
y = iris.target
# For a binary classification example, you could do:
# X_binary = X[y != 2] # Select only first two classes
# y_binary = y[y != 2]

print(f"Iris dataset features: {iris.feature_names}")
print(f"Iris dataset classes: {iris.target_names} (0, 1, 2)")
print(f"X shape: {x.shape}, Y shape: {y.shape}")
print("---"*30)

#Prepare data and split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

#Feature Scaling (Important for many algorithms, including sometimes Logistic Regression
scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)

#Create and fit logistic regression model
# 'solver' and 'multi_class' can be important. 'liblinear' is good for smaller datasets.
# 'ovr' (One-vs-Rest) is common for multi_class. 'auto' often picks a good default.
log_reg_model = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=42)
log_reg_model.fit(x_train_scaled, y_train)

#Make prediction
y_pred_log_reg = log_reg_model.predict(x_test_scaled)
y_pred_proba_log_reg = log_reg_model.predict_proba(x_test_scaled)

print("\n--- Linear Regression Results ---")
print(f"Sample Test Labels:    {y_test[:10]}")
print(f"Predicted Labels:     {y_pred_log_reg[:10]}")
print(f"Predicted Probabilities (first 5 samples for each class):\n{y_pred_proba_log_reg[:5]}")
print("---"*30)
# For each sample, predict_proba returns an array of probabilities for each class.
# e.g., [[P(class_0), P(class_1), P(class_2)], ...]

# --- 6. Basic Evaluation (more next week) ---
accuracy_log_reg = metrics.accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {accuracy_log_reg:.2f}")
print("---"*30)