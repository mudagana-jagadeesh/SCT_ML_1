# House Price Prediction — Linear Regression

A simple linear regression model that predicts house prices from **square footage (area)**, **number of bedrooms**, and **number of bathrooms**.

## Dataset

`Housing.csv` — 545 house listings with price, area, bedrooms, bathrooms, and additional categorical features (mainroad, guestroom, basement, airconditioning, parking, prefarea, furnishing status). This model uses only the three numeric features named above; the categorical columns are left for future extension.

## Results

| Metric | Train | Test |
|---|---|---|
| R² | 0.49 | 0.46 |
| MAE | 932,090 | 1,265,276 |
| RMSE | 1,249,294 | 1,658,325 |

**Coefficients** (price change per +1 unit, holding other features fixed):

| Feature | Coefficient |
|---|---|
| Area (sqft) | +345 |
| Bedrooms | +360,198 |
| Bathrooms | +1,422,320 |

Standardized coefficients show `area` is the strongest overall predictor, followed by `bathrooms`, then `bedrooms`.

> Note: R² ≈ 0.46 means these three features alone explain less than half of the price variation — the dataset's other columns (air conditioning, parking, preferred area, etc.) likely matter too. See "Next steps" below.

## Visualizations

- `actual_vs_predicted.png` — predicted vs. actual prices on the test set
- `feature_relationships.png` — price vs. area / bedrooms / bathrooms
- `residual_plot.png` — residuals vs. predicted price, to check for bias/heteroscedasticity

## Usage

```bash
pip install -r requirements.txt
python house_price_regression.py
```

## Example Prediction

A 3,000 sqft house with 3 bedrooms and 2 bathrooms is predicted at **₹5,021,117**.

## Next steps

- Include categorical features (one-hot encode `mainroad`, `airconditioning`, etc.) for higher accuracy
- Try regularized models (Ridge/Lasso) or non-linear models (Random Forest, Gradient Boosting) for comparison
- Add cross-validation instead of a single train/test split

## License

Dataset sourced from a public Kaggle housing prices dataset; code in this repo is provided under the MIT License.
