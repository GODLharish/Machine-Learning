import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Example: Using a simple dataset
# For demonstration, we'll generate a simple dataset using numpy
# Replace this with loading your own data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 data points, one feature
y = 2 * X + 1 + np.random.randn(100, 1)  # Linear relationship with noise

# Step 1: Load the data (using generated data in this case)
data = pd.DataFrame({'Feature': X.flatten(), 'Target': y.flatten()})

# Step 2: Statistical Analysis
print("Basic Statistical Summary:")
print(data.describe())  # Basic statistics

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Fit a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Coefficients
print("\nModel Coefficients:")
print("Slope (Coefficient):", model.coef_)
print("Intercept:", model.intercept_)

# Step 6: Predict on Test Set
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Performance Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Optional: Plotting the data and regression line
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Regression line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.legend()
plt.show()

