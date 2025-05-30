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

# --- 2. Prepare Data (X: features, y: target) ---
X = hours_studied
y = exam_score

# --- 3. Split Data into Training and Testing Sets ---
# test_size=0.2 means 20% of data for testing, 80% for training
# random_state ensures the same split every time for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}.")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_train.shape}")
print("---"*30)

# --- 4. Create an Instance of the Linear Regression Model ---
model = LinearRegression()

# --- 5. Fit the Model to the Training Data ---
model.fit(X_train, y_train)

# --- 6. Access Model Parameters ---
print(f"Model Intercept (beta_0): {model.intercept_[0]:.2f}")
print(f"Model coefficient (beta_1 for hours_studied): {model.coef_[0][0]:.2f}")
print("---"*30)

# Interpretation: For each additional hour studied, the score is expected to increase by approx. model.coef_[0] points,
# starting from a base of model.intercept_ if hours_studied were 0 (which might not be meaningful here).

# --- 7. Make Predictions on the Test Data ---
y_pred = model.predict(X_test)

# --- 8. Visualize the Results (Line of Best Fit) ---
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Test Data')
plt.plot(X_test, y_test, color='red', linewidth=2, label='Predicted Line on Best Fit')
plt.scatter(X_train, y_train, color='green', label='Training Data', alpha=0.3)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

# --- 9. Evaluate the Model ---
print("--- Model Evaluation ---")
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")
print("---"*30)
# Interpretation of R-squared: Approx. {r2*100:.0f}% of the variance in exam scores
# can be explained by the hours studied, according to this model on this test set.




