#Find a simple regression dataset online (e.g., look for "boston housing dataset csv", "salary data simple csv" 
# or use Scikit-learn's built-in datasets like load_diabetes()).

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

diabetes = load_diabetes()
X_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_diabetes = pd.Series(diabetes.target)

print(X_diabetes.head())
print("---"*30)
print(y_diabetes.head())
print("---"*30)


# Perform Simple Linear Regression:
# Choose one feature from your chosen dataset as the independent variable (X) and the target variable (y).
# Split your data into training and testing sets.
# Train a LinearRegression model.
# Print the intercept and coefficient. Interpret the coefficient.
# Make predictions on the test set.
# Calculate and print MAE, MSE, RMSE, and R-squared.
# Plot the test data points and the regression line.

# Choose one feature from your chosen dataset as the independent variable (X) and the target variable (y).
X = X_diabetes['bmi']
y = y_diabetes

# Split your data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LinearRegression model.

