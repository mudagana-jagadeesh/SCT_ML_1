import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("Housing.csv")
Y = df['price']
X = df[['area', 'bedrooms', 'bathrooms']]  # square footage, bedrooms, bathrooms
X = X.to_numpy()
Y = Y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
Y_pred = regr.predict(X_test)

# With 3 input features we can't plot a single regression line like before,
# so instead we plot predicted price vs actual price (a perfect model would
# fall on the dashed diagonal line).
plt.scatter(Y_test, Y_pred, color='black')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], linewidth=2, color='red', linestyle='--')
plt.title('Predicted vs Actual Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.savefig("regression_plot.png")
print("Plot saved as regression_plot.png")

print("R^2 on test set:", regr.score(X_test, Y_test))
print("Coefficients (area, bedrooms, bathrooms):", regr.coef_)
print("Intercept:", regr.intercept_)
plt.show()
