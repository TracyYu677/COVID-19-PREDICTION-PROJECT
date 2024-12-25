#5740 xg over a peroid 

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data
full_data = pd.read_csv('clean_covid_case.csv')

# Define the full feature set
full_features = [
    "lat",
    "lng",
    "population",
    "days_since_zero",
    "cases_last_week",
    "deaths_last_week",
    "cases_per_100k",
    "deaths_per_100k"
]

# Prepare the data
full_data = full_data.sort_values(by=["fips", "days_since_zero"])

# Augment the dataset for cumulative next-month cases
# Create a rolling sum for cases for the next 30 days
full_data["cases_next_month_sum"] = (
    full_data.groupby("fips")["cases_last_week"]
    .rolling(window=30, min_periods=1)
    .sum()
    .shift(-30)
    .reset_index(level=0, drop=True)
)

# Fill missing values caused by the shift
full_data = full_data.fillna(0)

# Define the new target variable
target_variable = "cases_next_month_sum"

# Train-test split based on temporal order
split_day = full_data["days_since_zero"].quantile(0.8)  # 80% for training
train_data = full_data[full_data["days_since_zero"] <= split_day]
test_data = full_data[full_data["days_since_zero"] > split_day]

# Prepare features and targets
X_train = train_data[full_features]
X_test = test_data[full_features]
y_train = train_data[target_variable]
y_test = test_data[target_variable]

# Initialize and train the XGBoost model
xgb_model = XGBRegressor(
    tree_method="hist",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Evaluate the model
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Display evaluation metrics
print(f"RMSE Train: {rmse_train}")
print(f"RMSE Test: {rmse_test}")
print(f"R2 Train: {r2_train}")
print(f"R2 Test: {r2_test}")

# Feature importance
feature_importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': full_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)
