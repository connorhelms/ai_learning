#Find a simple regression dataset online (e.g., look for "boston housing dataset csv", "salary data simple csv" 
# or use Scikit-learn's built-in datasets like load_diabetes()).

from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

print("="*60)
print("DIABETES DATASET ANALYSIS")
print("="*60)

diabetes = load_diabetes()
X_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_diabetes = pd.Series(diabetes.target)

print("Dataset Overview:")
print(f"Features shape: {X_diabetes.shape}")
print(f"Target shape: {y_diabetes.shape}")
print(f"\nFeature names: {list(diabetes.feature_names)}")
print(f"\nFirst 5 rows of features:")
print(X_diabetes.head())
print("---"*30)
print("First 5 target values:")
print(y_diabetes.head())
print("---"*30)

print("="*60)
print("SIMPLE LINEAR REGRESSION")
print("="*60)

# Perform Simple Linear Regression:
# Choose one feature from your chosen dataset as the independent variable (X) and the target variable (y).
# Split your data into training and testing sets.
# Train a LinearRegression model.
# Print the intercept and coefficient. Interpret the coefficient.
# Make predictions on the test set.
# Calculate and print MAE, MSE, RMSE, and R-squared.
# Plot the test data points and the regression line.

# Choose one feature from your chosen dataset as the independent variable (X) and the target variable (y).
X = X_diabetes[['bmi']]  # Double brackets to keep it as DataFrame (2D)
y = y_diabetes

# Split your data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LinearRegression model.
model = LinearRegression()
model.fit(X_train, y_train)

# Print the intercept and coefficient. Interpret the coefficient.
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")

# Make predictions on the test set.
y_pred = model.predict(X_test)

# Calculate and print MAE, MSE, RMSE, and R-squared.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

# Plot the test data points and the regression line.
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)

# Sort the test data for proper line plotting
sorted_indices = np.argsort(X_test.values.flatten())
X_test_sorted = X_test.iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Regression Line', linewidth=2)
plt.xlabel('BMI')
plt.ylabel('Diabetes Progression')
plt.title('Simple Linear Regression: BMI vs Diabetes Progression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Add interpretation
print(f"\nInterpretation:")
print(f"For every 1 unit increase in BMI, diabetes progression increases by {model.coef_[0]:.2f} units.")
print(f"The model explains {r2:.1%} of the variance in diabetes progression.")

print("="*60)
print("MULTIPLE LINEAR REGRESSION")
print("="*60)

# (Optional Bonus) Perform Multiple Linear Regression:
# Choose two or more features from your chosen dataset as independent variables (X).
# Repeat the steps for training, prediction, and evaluation. How do the evaluation metrics change compared to your simple linear regression model? What does model.coef_ look like now?

# Choose two or more features from your chosen dataset as independent variables (X).
X = X_diabetes[['bmi', 's5']]
y = y_diabetes

# Split your data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LinearRegression model.
model = LinearRegression()
model.fit(X_train, y_train)

#Print the intercept and coefficient. Interpret the coefficient.
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")

#Make predictions on the test set.
y_pred = model.predict(X_test)

#Calculate and print MAE, MSE, RMSE, and R-squared.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

#Plot the test data points and the regression line.
plt.figure(figsize=(15, 5))

# Subplot 1: BMI vs Target
plt.subplot(1, 3, 1)
plt.scatter(X_test['bmi'], y_test, color='blue', alpha=0.6)
plt.xlabel('BMI')
plt.ylabel('Diabetes Progression')
plt.title('BMI vs Target')
plt.grid(True, alpha=0.3)

# Subplot 2: S5 vs Target  
plt.subplot(1, 3, 2)
plt.scatter(X_test['s5'], y_test, color='green', alpha=0.6)
plt.xlabel('S5 (Blood Sugar)')
plt.ylabel('Diabetes Progression')
plt.title('S5 vs Target')
plt.grid(True, alpha=0.3)

# Subplot 3: Actual vs Predicted
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Add interpretation for multiple regression
print(f"\nMultiple Regression Interpretation:")
print(f"BMI coefficient: {model.coef_[0]:.2f} - For every 1 unit increase in BMI, diabetes progression increases by {model.coef_[0]:.2f} units (holding S5 constant)")
print(f"S5 coefficient: {model.coef_[1]:.2f} - For every 1 unit increase in S5, diabetes progression changes by {model.coef_[1]:.2f} units (holding BMI constant)")
print(f"This model explains {r2:.1%} of the variance in diabetes progression.")

# Compare with simple regression
print(f"\nModel Comparison:")
print(f"Simple regression R²: {0.033:.1%} (BMI only)")
print(f"Multiple regression R²: {r2:.1%} (BMI + S5)")
print(f"Improvement: {r2-0.033:.1%}")
