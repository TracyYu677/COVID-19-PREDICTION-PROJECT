import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

full_data = pd.read_csv('covid_data_augmented.csv')


# Define the full feature set
full_features = [
    "population",
    "days_since_zero",
    "cases_last_week",
    "cases_per_100k",
    "neighbor_population_sum",
]


'''
old one after analysis descided to drops some 


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



'''

# Prepare the data
# Group data by fips to preserve time-series structure
full_data = full_data.sort_values(by=["fips", "days_since_zero"])
counties = full_data["fips"].unique()

# Shift columns to create targets for case_tomorrow, case_next_week, case_next_month
full_data["case_tomorrow"] = full_data.groupby("fips")["cases_last_week"].shift(-1)
full_data["case_next_week"] = full_data.groupby("fips")["cases_last_week"].shift(-7)
full_data["case_next_month"] = full_data.groupby("fips")["cases_last_week"].shift(-30)
full_data["case_next_3month"] = full_data.groupby("fips")["cases_last_week"].shift(-90)
full_data["case_next_6month"] = full_data.groupby("fips")["cases_last_week"].shift(-180)


# Drop rows with NaN values due to shifting , but drop too many data for long term prediction model 
#full_data = full_data.fillna(0)


# Fill NaN values with the last valid observation for each fips group
full_data = full_data.groupby("fips").apply(lambda group: group.fillna(method="ffill")).reset_index(drop=True)

# Drop any remaining NaN values at the start of each group (if any)
full_data = full_data.dropna()

# Define target variables
target_variables = ["case_tomorrow", "case_next_week", "case_next_month","case_next_3month","case_next_6month"]

split_day = full_data["days_since_zero"].quantile(0.8)



train_data = full_data[full_data["days_since_zero"] <= split_day]
test_data = full_data[full_data["days_since_zero"] > split_day]


X_train = train_data[full_features]
X_test = test_data[full_features]
'''


# this might not works. due to random split data 
X = full_data[full_features]
X_train, X_test, train_indices, test_indices = train_test_split(X, full_data.index, test_size=0.2, random_state=42)

# Add county_x names as labels
X_train_with_labels = X_train.copy(),
X_test_with_labels = X_test.copy()
X_train_with_labels['county_x'] = full_data.loc[train_indices, 'county_x'].values
X_test_with_labels['county_x'] = full_data.loc[test_indices, 'county_x'].values

# Save feature sets with labels to CSV
X_train_with_labels.to_csv("X_train_with_labels.csv", index=False)
X_test_with_labels.to_csv("X_test_with_labels.csv", index=False)

'''




results = {}
comparison_data = []





import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import numpy as np

# Define the number of splits for time-series cross-validation
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Storage for hyperparameter results
all_best_params = []

# Objective function for Optuna
def objective(trial, X_train, y_train):
    # Define the search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "lambda": trial.suggest_float("lambda", 1, 10),
        "alpha": trial.suggest_float("alpha", 0, 10),
    }

    # Perform time-series cross-validation
    rmse_scores = []
    for train_index, test_index in tscv.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Initialize and train the model
        model = XGBRegressor(tree_method="hist", **params)
        model.fit(X_train_fold, y_train_fold)

        # Predict and calculate RMSE for this fold
        y_pred_test_fold = model.predict(X_test_fold)
        rmse = mean_squared_error(y_test_fold, y_pred_test_fold, squared=False)
        rmse_scores.append(rmse)

    # Return the average RMSE across folds
    return np.mean(rmse_scores)

'''
# Optimize hyperparameters for "case_next_6month"
target = "case_next_6month"
print(f"Optimizing hyperparameters for target: {target}")
y = full_data[target]

# Define the optimization function
def target_objective(trial):
    return objective(trial, X_train, y)

# Create an Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(target_objective, n_trials=20)

# Store the best parameters for "case_next_6month"
optimized_params = study.best_params
print(f"Best hyperparameters for {target}: {optimized_params}")

# Train final models for all targets using the optimized hyperparameters
final_results = {}
for target in target_variables:
    print(f"\nTraining final model for target: {target} using optimized hyperparameters")
    y = full_data[target]

    # Perform time-series cross-validation with optimized hyperparameters
    rmse_scores = []
    r2_scores = []
    for train_index, test_index in tscv.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Train the model using optimized hyperparameters
        model = XGBRegressor(tree_method="gpu_hist")
        model.fit(X_train_fold, y_train_fold)

        # Predict and calculate metrics for this fold
        y_pred_test_fold = model.predict(X_test_fold)
        rmse = mean_squared_error(y_test_fold, y_pred_test_fold, squared=False)
        r2 = r2_score(y_test_fold, y_pred_test_fold)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    # Store average metrics for this target
    final_results[target] = {
        "Average RMSE": np.mean(rmse_scores),
        "Average R2": np.mean(r2_scores),
    }

# Output final results
print("\nFinal Model Results Using Optimized Hyperparameters:")
for target, metrics in final_results.items():
    print(f"Target: {target}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")






'''

# Optimize hyperparameters for each target
for target in target_variables:
    print(f"Optimizing hyperparameters for target: {target}")
    y = full_data[target]

    # Define the optimization function
    def target_objective(trial):
        return objective(trial, X_train, y)

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(target_objective, n_trials=20)

    # Store the best parameters for this target
    print(f"Best hyperparameters for {target}: {study.best_params}")
    all_best_params.append(study.best_params)

# Calculate average hyperparameters across all models
avg_params = {}
for key in all_best_params[0]:  # Iterate over all hyperparameter keys
    avg_params[key] = np.mean([params[key] for params in all_best_params])

print("\nAverage Hyperparameters Across All Models:")
for param, value in avg_params.items():
    print(f"  {param}: {value}")

# Train final models using the average hyperparameters
final_results = {}
for target in target_variables[4]:
    print(f"\nTraining final model for target: {target} using average hyperparameters")
    y = full_data[target]

    # Perform time-series cross-validation with average hyperparameters
    rmse_scores = []
    r2_scores = []
    for train_index, test_index in tscv.split(X_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Train the model using average hyperparameters
        model = XGBRegressor(tree_method="hist", **avg_params)
        model.fit(X_train_fold, y_train_fold)

        # Predict and calculate metrics for this fold
        y_pred_test_fold = model.predict(X_test_fold)
        rmse = mean_squared_error(y_test_fold, y_pred_test_fold, squared=False)
        r2 = r2_score(y_test_fold, y_pred_test_fold)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

    # Store average metrics for this target
    final_results[target] = {
        "Average RMSE": np.mean(rmse_scores),
        "Average R2": np.mean(r2_scores),
    }

# Output final results
print("\nFinal Model Results Using Average Hyperparameters:")
for target, metrics in final_results.items():
    print(f"Target: {target}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")











        