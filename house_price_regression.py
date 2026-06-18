"""
Linear Regression: Predicting House Prices
Features: square footage (area), number of bedrooms, number of bathrooms
Dataset: Housing.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv("Housing.csv")

features = ["area", "bedrooms", "bathrooms"]
target = "price"

X = df[features]
y = df[target]

print("Dataset shape:", df.shape)
print("\nFeature summary:")
print(X.describe())

# ----------------------------------------------------------------------
# 2. Train / test split
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------------
# 3. Scale features (helps interpret coefficients on a comparable scale)
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 4. Fit linear regression (on raw features, for real-unit coefficients)
# ----------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ----------------------------------------------------------------------
# 5. Evaluate
# ----------------------------------------------------------------------
def report(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{label} performance:")
    print(f"  R^2   : {r2:.4f}")
    print(f"  MAE   : {mae:,.0f}")
    print(f"  RMSE  : {rmse:,.0f}")
    return r2, mae, rmse

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
report(y_train, y_pred_train, "Train")
report(y_test, y_pred_test, "Test")

# ----------------------------------------------------------------------
# 6. Inspect coefficients
# ----------------------------------------------------------------------
print("\n" + "=" * 50)
print("MODEL COEFFICIENTS")
print("=" * 50)
print(f"Intercept: {model.intercept_:,.2f}")
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:10s}: {coef:,.2f}  (price change per +1 unit, holding others fixed)")

# Standardized coefficients to compare relative importance
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
print("\nStandardized coefficients (relative importance, larger |value| = stronger effect):")
for feat, coef in sorted(
    zip(features, model_scaled.coef_), key=lambda t: -abs(t[1])
):
    print(f"  {feat:10s}: {coef:,.2f}")

# ----------------------------------------------------------------------
# 7. Example prediction
# ----------------------------------------------------------------------
example = pd.DataFrame({"area": [3000], "bedrooms": [3], "bathrooms": [2]})
example_pred = model.predict(example)[0]
print("\n" + "=" * 50)
print("EXAMPLE PREDICTION")
print("=" * 50)
print(f"House with {example.iloc[0]['area']} sqft, "
      f"{example.iloc[0]['bedrooms']} bedrooms, "
      f"{example.iloc[0]['bathrooms']} bathrooms")
print(f"Predicted price: {example_pred:,.0f}")

# ----------------------------------------------------------------------
# 8. Plot: actual vs predicted prices (test set)
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolor="k")
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
print("Saved plot: actual_vs_predicted.png")

# ----------------------------------------------------------------------
# 9. Plot: feature relationships with price (with regression trend lines)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Area vs price (continuous -> scatter + trend line)
axes[0].scatter(df["area"], df["price"], alpha=0.5, edgecolor="k", color="#4C72B0")
z = np.polyfit(df["area"], df["price"], 1)
x_line = np.linspace(df["area"].min(), df["area"].max(), 100)
axes[0].plot(x_line, np.poly1d(z)(x_line), "r--", linewidth=2)
axes[0].set_xlabel("Area (sqft)")
axes[0].set_ylabel("Price")
axes[0].set_title("Price vs Area")

# Bedrooms vs price (discrete -> boxplot)
df.boxplot(column="price", by="bedrooms", ax=axes[1])
axes[1].set_xlabel("Bedrooms")
axes[1].set_ylabel("Price")
axes[1].set_title("Price vs Bedrooms")
fig.suptitle("")  # remove default boxplot supertitle

# Bathrooms vs price (discrete -> boxplot)
df.boxplot(column="price", by="bathrooms", ax=axes[2])
axes[2].set_xlabel("Bathrooms")
axes[2].set_ylabel("Price")
axes[2].set_title("Price vs Bathrooms")

plt.tight_layout()
plt.savefig("feature_relationships.png", dpi=150)
print("Saved plot: feature_relationships.png")

# ----------------------------------------------------------------------
# 10. Residual plot (errors vs predicted price)
# ----------------------------------------------------------------------
residuals = y_test - y_pred_test
plt.figure(figsize=(7, 6))
plt.scatter(y_pred_test, residuals, alpha=0.6, edgecolor="k", color="#55A868")
plt.axhline(0, color="r", linestyle="--", linewidth=2)
plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot (Test Set)")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
print("Saved plot: residual_plot.png")
