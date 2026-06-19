import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("Housing.csv")

# Select features and target
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate model
print("\n----- Training Performance -----")
print("R2 Score :", round(r2_score(y_train, y_train_pred), 4))
print("MAE      :", round(mean_absolute_error(y_train, y_train_pred), 2))
print("RMSE     :", round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2))

print("\n----- Testing Performance -----")
print("R2 Score :", round(r2_score(y_test, y_test_pred), 4))
print("MAE      :", round(mean_absolute_error(y_test, y_test_pred), 2))
print("RMSE     :", round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2))

# Display intercept and slopes
print("\n----- Model Parameters -----")
print("Intercept:", round(model.intercept_, 2))

print("\nSlopes (Coefficients):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {round(coef, 2)}")

# Regression Equation
print("\nRegression Equation:")
print(
    f"Price = {model.intercept_:.2f} "
    f"+ ({model.coef_[0]:.2f} * Area) "
    f"+ ({model.coef_[1]:.2f} * Bedrooms) "
    f"+ ({model.coef_[2]:.2f} * Bathrooms)"
)

# Predict a sample house price
# Predict a sample house price
sample_house = pd.DataFrame({
    "area": [5000],
    "bedrooms": [3],
    "bathrooms": [2]
})

predicted_price = model.predict(sample_house)

print("\nSample Prediction")
print("House Details: Area=5000, Bedrooms=3, Bathrooms=2")
print("Predicted Price:", round(predicted_price[0], 2))

# Actual vs Predicted Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_test_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Residual Plot
residuals = y_test - y_test_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_test_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
