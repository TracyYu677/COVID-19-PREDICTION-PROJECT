#xgcross
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

for target in target_variables:
    from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define the number of splits for cross-validation
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

results = {}

# Loop through each target variable
for target in target_variables:
    print(f"Performing cross-validation for target: {target}")
    
    # Prepare storage for fold metrics
    fold_metrics = {
        "Fold": [],
        "RMSE Train": [],
        "RMSE Test": [],
        "R2 Train": [],
        "R2 Test": []
    }

    y = full_data[target]  # Use the current target column
    for fold, (train_index, test_index) in enumerate(tscv.split(X_train)):
        # Split the data into training and testing sets for this fold
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Initialize the XGBoost model
        xgb_model = XGBRegressor(
        tree_method="hist",  
         device = "cuda",
        n_estimators=155,  # Updated based on best hyperparameters
        max_depth=5,  # Updated based on best hyperparameters
        learning_rate=0.013966296391930243,  # Updated based on best hyperparameters
        subsample=0.8505513505059591,  # Updated based on best hyperparameters
        colsample_bytree=0.6544180140723905,  # Updated based on best hyperparameters
        gamma=5.906779319329253,  # Updated based on best hyperparameters
        reg_lambda=8.076599789131794,  # Updated based on best hyperparameters
        reg_alpha=2.4128045031178256  # Updated based on best hyperparameters
    )

        # Train the model on this fold's training set
        xgb_model.fit(X_train_fold, y_train_fold)

        # Predict on both training and testing sets
        y_pred_train_fold = xgb_model.predict(X_train_fold)
        y_pred_test_fold = xgb_model.predict(X_test_fold)

        # Evaluate metrics for this fold
        rmse_train = mean_squared_error(y_train_fold, y_pred_train_fold, squared=False)
        rmse_test = mean_squared_error(y_test_fold, y_pred_test_fold, squared=False)
        r2_train = r2_score(y_train_fold, y_pred_train_fold)
        r2_test = r2_score(y_test_fold, y_pred_test_fold)

        # Store metrics for this fold
        fold_metrics["Fold"].append(fold + 1)
        fold_metrics["RMSE Train"].append(rmse_train)
        fold_metrics["RMSE Test"].append(rmse_test)
        fold_metrics["R2 Train"].append(r2_train)
        fold_metrics["R2 Test"].append(r2_test)

    # Store average metrics across all folds for this target
    avg_metrics = {metric: sum(values) / len(values) for metric, values in fold_metrics.items() if metric != "Fold"}
    results[target] = avg_metrics

    # Output fold-wise metrics for this target
    print(f"Cross-validation results for {target}:")
    for fold, rmse_train, rmse_test, r2_train, r2_test in zip(
        fold_metrics["Fold"],
        fold_metrics["RMSE Train"],
        fold_metrics["RMSE Test"],
        fold_metrics["R2 Train"],
        fold_metrics["R2 Test"]
    ):
        print(f"  Fold {fold}: RMSE Train = {rmse_train}, RMSE Test = {rmse_test}, R2 Train = {r2_train}, R2 Test = {r2_test}")

    print(f"Average metrics for {target}:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value}")

# Output average metrics across all targets
print("\nFinal Cross-Validation Results:")
for target, metrics in results.items():
    print(f"Target: {target}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    