import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import optuna

# Load dataset
full_data = pd.read_csv('covid_data_augmented.csv')

# Define the full feature set
full_features = [
    "population",
    "days_since_zero",
    "cases_last_week",
    "cases_per_100k",
    "neighbor_population_sum",
    "cases_past_7_days"
]

# Prepare the data
# Group data by fips to preserve time-series structure
full_data = full_data.sort_values(by=["fips", "days_since_zero"])

# Calculate past 7 days total cases
full_data["cases_past_7_days"] = (
    full_data.groupby("fips")["cases_last_week"]
    .apply(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
    .reset_index(level=0, drop=True)  # Align the index with full_data
)

# Fill NaN values in "cases_past_7_days" with current day data
full_data["cases_past_7_days"] = full_data["cases_past_7_days"].fillna(full_data["cases_last_week"])

# Shift columns to create target for case_tomorrow
full_data["case_tomorrow"] = full_data.groupby("fips")["cases_last_week"].shift(-1)

# Fill NaN values with the last valid observation for each fips group
full_data = full_data.groupby("fips").apply(lambda group: group.fillna(method="ffill")).reset_index(drop=True)

# Drop any remaining NaN values at the start of each group (if any)
full_data = full_data.dropna()

# Define target variable
target_variable = "case_tomorrow"

# Define a custom time series split function with limited train and test set size
def limited_time_series_split(data_length, n_splits, train_size, test_size):
    """
    Custom time series split with limited train and test set size.
    
    Parameters:
        data_length: int, total length of the dataset
        n_splits: int, number of splits
        train_size: int, number of samples in the training set
        test_size: int, number of samples in the test set

    Yields:
        train_indices, test_indices: indices for the train and test sets
    """
    step = (data_length - (train_size + test_size)) // (n_splits + 1)
    for i in range(n_splits):
        train_start = step * i
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size

        train_indices = np.arange(train_start, train_end)
        test_indices = np.arange(test_start, test_end)

        yield train_indices, test_indices

# Define parameters for cross-validation
n_splits = 3
train_size = 90
test_size = 30

# Initialize previous parameters and R2 threshold
previous_params = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "reg_lambda": 1,
    "reg_alpha": 0
}
previous_r2 = 0.5

# Create a DataFrame to store results
results_df = pd.DataFrame(columns=["Fold", "R2 Before", "R2 After", "Predicted", "Actual", "Day", "FIPS"])

# Optuna objective function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
    }

    X = full_data[full_features]
    y = full_data[target_variable]
    data_length = len(X)
    fold_rmse = []

    for train_index, test_index in limited_time_series_split(data_length, n_splits, train_size, test_size):
        # Split the data into training and testing sets for this fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize the XGBoost model
        xgb_model = XGBRegressor(tree_method="hist", **params, random_state=42)

        # Train the model on this fold's training set
        xgb_model.fit(X_train, y_train)

        # Predict on the testing set
        y_pred_test = xgb_model.predict(X_test)

        # Evaluate RMSE for this fold
        rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
        fold_rmse.append(rmse_test)

    return np.mean(fold_rmse)

# Initialize a DataFrame to store all results
all_results_df = pd.DataFrame(columns=["FIPS", "Fold", "R2 Before", "R2 After", "Predicted", "Actual", "Day"])

# Loop through each unique FIPS
for fips in full_data["fips"].unique():

    if fips> 1025:
        break

    print(f"Processing FIPS: {fips}")
    fips_data = full_data[full_data["fips"] == fips].reset_index(drop=True)

    for fold, (train_index, test_index) in enumerate(limited_time_series_split(len(fips_data), n_splits, train_size, test_size)):
        X_train, X_test = fips_data[full_features].iloc[train_index], fips_data[full_features].iloc[test_index]
        y_train, y_test = fips_data[target_variable].iloc[train_index], fips_data[target_variable].iloc[test_index]

        # Assign weights to training data (recent data gets full weight, older data gets less weight)
        sample_weights = np.linspace(0.5, 1.0, len(y_train))

        print(f"Running fold {fold + 1} for FIPS: {fips}...")
        r2_before = previous_r2

        if previous_r2 < 0.7:
            print("R2 below threshold, running Optuna...")
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=10)
            best_params = study.best_params
            previous_params = best_params
        else:
            print("Using previous parameters...")
            best_params = previous_params

        # Train model with the best parameters and weighted training data
        xgb_model = XGBRegressor(tree_method="hist", **best_params, random_state=42)
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

        # Make predictions
        y_pred_test = xgb_model.predict(X_test)
        fold_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        fold_r2 = r2_score(y_test, y_pred_test)

        print(f"Fold RMSE: {fold_rmse}, Fold R2: {fold_r2}")

        # Store results
        all_results_df = pd.concat([
            all_results_df,
            pd.DataFrame({
                "FIPS": [fips] * len(y_test),
                "Fold": [fold + 1] * len(y_test),
                "R2 Before": [r2_before] * len(y_test),
                "R2 After": [fold_r2] * len(y_test),
                "Predicted": y_pred_test,
                "Actual": y_test.values,
                "Day": fips_data["days_since_zero"].iloc[test_index].values
            })
        ], ignore_index=True)

        previous_r2 = fold_r2

# Save aggregated results to CSV
all_results_df.to_csv("all_fips_model_results.csv", index=False)

# Final model evaluation using average R2
final_r2 = all_results_df["R2 After"].mean()
print(f"Final Average R2 across all FIPS: {final_r2}")
