import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
full_data = pd.read_csv('covid_data_augmented.csv')

# Define feature set
full_features = [
    "population",
    "days_since_zero",
    "cases_last_week",
    "cases_per_100k",
    "neighbor_population_sum",
]

# Sort data to preserve time-series structure
full_data = full_data.sort_values(by=["fips", "days_since_zero"])

# Create target variables for future predictions of cases_per_100k
full_data["cases_per_100k_tomorrow"] = full_data.groupby("fips")["cases_per_100k"].shift(-1)
full_data["cases_per_100k_next_week"] = full_data.groupby("fips")["cases_per_100k"].shift(-7)
full_data["cases_per_100k_next_month"] = full_data.groupby("fips")["cases_per_100k"].shift(-30)
full_data["cases_per_100k_next_3month"] = full_data.groupby("fips")["cases_per_100k"].shift(-90)
full_data["cases_per_100k_next_6month"] = full_data.groupby("fips")["cases_per_100k"].shift(-180)

# Drop NaN values caused by shifting
full_data = full_data.groupby("fips").apply(lambda group: group.fillna(method="ffill")).reset_index(drop=True)
full_data = full_data.dropna()

# Define target variables
target_variables = [
    "cases_per_100k_tomorrow",
    "cases_per_100k_next_week",
    "cases_per_100k_next_month",
    "cases_per_100k_next_3month",
    "cases_per_100k_next_6month"
]

# Split the dataset into training and testing sets
split_day = full_data["days_since_zero"].quantile(0.8)
train_data = full_data[full_data["days_since_zero"] <= split_day]
test_data = full_data[full_data["days_since_zero"] > split_day]

X_train = train_data[full_features]
X_test = test_data[full_features]

# Initialize a dictionary to store results
results = {}
final_combined_output = test_data[["days_since_zero", "fips"]].copy()
train_combined_output = train_data[["days_since_zero", "fips"]].copy()

# Loop through each target variable
for target in target_variables:
    y_train = train_data[target]
    y_test = test_data[target]

    # Initialize the XGBoost model
    xgb_model = XGBRegressor(
        tree_method="hist",
        device="cuda",  # Use GPU acceleration if available
        n_estimators=500,
        max_depth=6,
        learning_rate=0.0125,
        subsample=0.786,
        colsample_bytree=0.753,
        gamma=5.59,
        reg_lambda=5.68,
        reg_alpha=4.36,
        random_state=42
    )


    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    # Save predictions to the final combined output
    if target == "cases_per_100k_tomorrow":
        final_combined_output["y_test"] = y_test.values
        train_combined_output["y_train"] = y_train.values
    final_combined_output[f"y_pred_{target}"] = y_pred_test
    train_combined_output[f"y_pred_{target}"] = y_pred_train

    # Evaluate the model
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Store results for each target
    results[target] = {
        "RMSE Train": rmse_train,
        "RMSE Test": rmse_test,
        "R^2 Train": r2_train,
        "R^2 Test": r2_test
    }

    print(f"Processed predictions for target: {target}")

# Display evaluation metrics
for target, metrics in results.items():
    print(f"Results for {target}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print()

# Save the final combined output to a CSV file
final_combined_output.to_csv("final_combined_output_cases_per_100k.csv", index=False)
train_combined_output.to_csv("train_combined_output_cases_per_100k.csv", index=False)
print("All outputs saved to 'final_combined_output_cases_per_100k.csv'")
'''
# Visualize predictions vs actual values for each target
for target in target_variables:
    plt.figure(figsize=(8, 6))
    plt.scatter(test_data[target], final_combined_output[f"y_pred_{target}"], alpha=0.5)
    plt.plot([test_data[target].min(), test_data[target].max()], 
             [test_data[target].min(), test_data[target].max()], 'r--', lw=2)
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"Actual vs Predicted for {target}")
    plt.grid(True)
    plt.show()
'''