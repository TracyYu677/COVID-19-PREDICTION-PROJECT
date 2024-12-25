import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the datasets
full_data = pd.read_csv('covid_data_augmented.csv')
sarimax_features = pd.read_csv('sarimax_features.csv')  # Ensure this file is saved locally
sarimax_features['date'] = pd.to_datetime(sarimax_features['date'])
full_data['date'] = pd.to_datetime(full_data['date'])

# Merge SARIMAX features into the full dataset
full_data = full_data.merge(sarimax_features, on='date', how='left')

# Fill any missing values after merging
full_data[['trend', 'residuals']] = full_data[['trend', 'residuals']].fillna(0)

# Define the feature set (now including trend and residuals)
full_features = [
    "days_since_zero",
    "cases_per_100k",
    "trend",
    "residuals"  # Added residuals
]

# Prepare the data
full_data = full_data.sort_values(by=["fips", "days_since_zero"])

# Update target variables to include residuals
full_data["case_tomorrow"] = full_data.groupby("fips")["cases_last_week"].shift(-1) + full_data.groupby("fips")["residuals"].shift(-1)
full_data["case_next_week"] = full_data.groupby("fips")["cases_last_week"].shift(-7) + full_data.groupby("fips")["residuals"].shift(-7)
full_data["case_next_month"] = full_data.groupby("fips")["cases_last_week"].shift(-30) + full_data.groupby("fips")["residuals"].shift(-30)
full_data["case_next_3month"] = full_data.groupby("fips")["cases_last_week"].shift(-90) + full_data.groupby("fips")["residuals"].shift(-90)
full_data["case_next_6month"] = full_data.groupby("fips")["cases_last_week"].shift(-180) + full_data.groupby("fips")["residuals"].shift(-180)

# Fill missing target values and propagate data forward
full_data = full_data.groupby("fips").apply(lambda group: group.fillna(method="ffill")).reset_index(drop=True)
full_data = full_data.dropna()

# Define target variables
target_variables = ["case_tomorrow", "case_next_week", "case_next_month", "case_next_3month", "case_next_6month"]

# Split into train and test sets
split_day = full_data["days_since_zero"].quantile(0.8)
train_data = full_data[full_data["days_since_zero"] <= split_day]
test_data = full_data[full_data["days_since_zero"] > split_day]

X_train = train_data[full_features]
X_test = test_data[full_features]

# XGBoost Training Loop
results = {}

for target in target_variables:
    y_train = train_data[target]
    y_test = test_data[target]

    # Initialize the XGBoost model
    xgb_model = XGBRegressor(
        tree_method="hist",  # Use 'gpu_hist' if running on a GPU
        device = "cuda",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    # Evaluate the model
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Store RMSE and R2 results
    results[target] = {
        "RMSE Train": rmse_train,
        "RMSE Test": rmse_test,
        "R2 Train": r2_train,
        "R2 Test": r2_test
    }

    # Feature Importance
    feature_importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': full_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(f"Feature importances for {target}:")
    print(feature_importance_df)

# Output RMSE and R2 results
for target, metrics in results.items():
    print(f"Metrics for {target}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")




'''
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the datasets
full_data = pd.read_csv('clean_covid_case1.csv')
sarimax_features = pd.read_csv('sarimax_features.csv')  # Ensure this file is saved locally
sarimax_features['date'] = pd.to_datetime(sarimax_features['date'])
full_data['date'] = pd.to_datetime(full_data['date'])

# Merge SARIMAX features into the full dataset
full_data = full_data.merge(sarimax_features, on='date', how='left')

# Fill any missing values after merging
full_data[['trend', 'residuals']] = full_data[['trend', 'residuals']].fillna(0)

# Define the feature set (now including trend and residuals)
full_features = [
    "cases",
    "population",
    "days_since_zero",
    "cases_last_week",
    "cases_per_100k",
    "avg_7",
    "residuals"  # Added residuals
]

# Prepare the data
full_data = full_data.sort_values(by=["fips", "days_since_zero"])
full_data["case_tomorrow"] = full_data.groupby("fips")["cases_last_week"].shift(-1)
full_data["case_next_week"] = full_data.groupby("fips")["cases_last_week"].shift(-7)
full_data["case_next_month"] = full_data.groupby("fips")["cases_last_week"].shift(-30)
full_data["case_next_3month"] = full_data.groupby("fips")["cases_last_week"].shift(-90)
full_data["case_next_6month"] = full_data.groupby("fips")["cases_last_week"].shift(-180)

# Fill missing target values and propagate data forward
full_data = full_data.groupby("fips").apply(lambda group: group.fillna(method="ffill")).reset_index(drop=True)
full_data = full_data.dropna()

# Define target variables
target_variables = ["case_tomorrow", "case_next_week", "case_next_month", "case_next_3month", "case_next_6month"]

# Split into train and test sets
split_day = full_data["days_since_zero"].quantile(0.8)
train_data = full_data[full_data["days_since_zero"] <= split_day]
test_data = full_data[full_data["days_since_zero"] > split_day]

X_train = train_data[full_features]
X_test = test_data[full_features]

# XGBoost Training Loop
results = {}




''''''
for target in target_variables:
    y_train = train_data[target]
    y_test = test_data[target]

    # Initialize the XGBoost model
    xgb_model = XGBRegressor(
        tree_method="hist",  # Use 'gpu_hist' if running on a GPU
        device = "cuda",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    # Evaluate the model
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Store RMSE and R2 results
    results[target] = {
        "RMSE Train": rmse_train,
        "RMSE Test": rmse_test,
        "R2 Train": r2_train,
        "R2 Test": r2_test
    }

    # Feature Importance
    feature_importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': full_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(f"Feature importances for {target}:")
    print(feature_importance_df)

# Output RMSE and R2 results
for target, metrics in results.items():
    print(f"Metrics for {target}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the datasets
full_data = pd.read_csv('clean_covid_case1.csv')
sarimax_features = pd.read_csv('sarimax_features.csv')  # Ensure this file is saved locally
sarimax_features['date'] = pd.to_datetime(sarimax_features['date'])
full_data['date'] = pd.to_datetime(full_data['date'])

# Merge SARIMAX features into the full dataset
full_data = full_data.merge(sarimax_features, on='date', how='left')

# Fill any missing values after merging
full_data[['trend', 'residuals']] = full_data[['trend', 'residuals']].fillna(0)

# Define the feature set (now including trend and residuals)
full_features = [
    "days_since_zero",
    "cases_last_week",
    "cases_per_100k",
    "avg_7",
]

# Prepare the data
full_data = full_data.sort_values(by=["fips", "days_since_zero"])
full_data["case_tomorrow"] = full_data.groupby("fips")["cases_last_week"].shift(-1)
full_data["case_next_week"] = full_data.groupby("fips")["cases_last_week"].shift(-7)
full_data["case_next_month"] = full_data.groupby("fips")["cases_last_week"].shift(-30)
full_data["case_next_3month"] = full_data.groupby("fips")["cases_last_week"].shift(-90)
full_data["case_next_6month"] = full_data.groupby("fips")["cases_last_week"].shift(-180)

# Fill missing target values and propagate data forward
full_data = full_data.groupby("fips").apply(lambda group: group.fillna(method="ffill")).reset_index(drop=True)
full_data = full_data.dropna()

# Define target variables (use residuals as targets instead of cases)
target_variables = ["case_tomorrow", "case_next_week", "case_next_month", "case_next_3month", "case_next_6month"]

# Split into train and test sets
split_day = full_data["days_since_zero"].quantile(0.8)
train_data = full_data[full_data["days_since_zero"] <= split_day]
test_data = full_data[full_data["days_since_zero"] > split_day]

X_train = train_data[full_features]
X_test = test_data[full_features]

results = {}

for target in target_variables:
    # Calculate the residuals as the target
    train_data = train_data.copy()
    test_data = test_data.copy()

    train_data.loc[:, "target_residuals"] = train_data[target] - train_data["trend"]
    test_data.loc[:, "target_residuals"] = test_data[target] - test_data["trend"]


    y_train = train_data["target_residuals"]
    y_test = test_data["target_residuals"]

    # Initialize the XGBoost model
    xgb_model = XGBRegressor(
        tree_method="hist",  # Use 'gpu_hist' if running on a GPU
        n_estimators=100,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8
    )

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train_residual = xgb_model.predict(X_train)
    y_pred_test_residual = xgb_model.predict(X_test)

    # Convert residual predictions back to full predictions
    y_pred_train = y_pred_train_residual + train_data["trend"]
    y_pred_test = y_pred_test_residual + test_data["trend"]

    # Evaluate the model
    rmse_train = mean_squared_error(y_train + train_data["trend"], y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test + test_data["trend"], y_pred_test, squared=False)
    r2_train = r2_score(y_train + train_data["trend"], y_pred_train)
    r2_test = r2_score(y_test + test_data["trend"], y_pred_test)

    # Store RMSE and R2 results
    results[target] = {
        "RMSE Train": rmse_train,
        "RMSE Test": rmse_test,
        "R2 Train": r2_train,
        "R2 Test": r2_test
    }

    # Feature Importance
    feature_importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': full_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(f"Feature importances for {target}:")
    print(feature_importance_df)

# Output RMSE and R2 results
for target, metrics in results.items():
    print(f"Metrics for {target}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")


        


'''