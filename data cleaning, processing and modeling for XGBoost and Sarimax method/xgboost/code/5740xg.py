import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
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
final_combined_output = test_data[["days_since_zero", "fips"]].copy()
train_combined_output = train_data[["days_since_zero", "fips"]].copy()


for target in target_variables:
    
    y_train = train_data[target]
    y_test = test_data[target]
    

    # Initialize the XGBoost model


# Update XGBoost model with new hyperparameters
    xgb_model = XGBRegressor(
    tree_method="hist",  # Use 'gpu_hist' if running on a GPU
    device="cuda",  # For GPU acceleration
    n_estimators=81,  # Average hyperparameter across all models
    max_depth=4,  # Average hyperparameter across all models
    learning_rate=0.022515959962743554,  # Average hyperparameter across all models
    subsample=0.786175684217626,  # Average hyperparameter across all models
    colsample_bytree=0.7532617251938488,  # Average hyperparameter across all models
    gamma=5.590621106342558,  # Average hyperparameter across all models
    reg_lambda=5.677927787435517,  # Regularization parameter (lambda)
    reg_alpha=4.356785098886055,  # Regularization parameter (alpha)
    random_state=42  # For reproducibility
)

   


    ''' 
    
    xgb_model = XGBRegressor(
        tree_method="hist",  # Use 'gpu_hist' if running on a GPU
        device = "cuda",
        n_estimators=100,  # You can adjust this for deeper training
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,  # Use 80% of the data for each tree
        colsample_bytree=0.8,  # Use 80% of features for each tree
        random_state=42 
    )
       

    
    '''

    # Train the model
    xgb_model.fit(X_train, y_train)

   # Make predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    if target=="case_tomorrow":
        final_combined_output[f"y_test"] = y_test.values
        train_combined_output["y_train"] = y_train.values
    train_combined_output[f"y_pred_{target}"] = y_pred_train
    final_combined_output[f"y_pred_{target}"] = y_pred_test
    print(f"Processed predictions for target: {target}")


    '''
    # Feature Importance
    feature_importances = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': full_features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(f"Feature importances for {target}:")
    print(feature_importance_df)
    
    '''




    # Evaluate the model
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Threshold for classification (set based on domain knowledge or experimentation)
    threshold = 0.5

    # Binarize predictions for classification metrics
    y_pred_train_class = (y_pred_train >= threshold).astype(int)
    y_pred_test_class = (y_pred_test >= threshold).astype(int)
    y_train_class = (y_train >= threshold).astype(int)
    y_test_class = (y_test >= threshold).astype(int)

    # Compute additional metrics
    accuracy_train = accuracy_score(y_train_class, y_pred_train_class)
    accuracy_test = accuracy_score(y_test_class, y_pred_test_class)
    precision_train = precision_score(y_train_class, y_pred_train_class)
    precision_test = precision_score(y_test_class, y_pred_test_class)
    recall_train = recall_score(y_train_class, y_pred_train_class)
    recall_test = recall_score(y_test_class, y_pred_test_class)
    f1_train = f1_score(y_train_class, y_pred_train_class)
    f1_test = f1_score(y_test_class, y_pred_test_class)

    # Store all results
    results[target] = {
        "RMSE Train": rmse_train,
        "RMSE Test": rmse_test,
        "R2 Train": r2_train,
        "R2 Test": r2_test,
        "Accuracy Train": accuracy_train,
        "Accuracy Test": accuracy_test,
        "Precision Train": precision_train,
        "Precision Test": precision_test,
        "Recall Train": recall_train,
        "Recall Test": recall_test,
        "F1 Score Train": f1_train,
        "F1 Score Test": f1_test
    }

for target, metrics in results.items():
    print(f"Results for {target}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    print()


# Add population column to final_combined_output
#final_combined_output["population"] = test_data["population"].values
#train_combined_output["population"] = train_data["population"].values




# Save the final combined output to a single CSV file
train_combined_output.to_csv("train_combined_output.csv", index=False)
final_combined_output.to_csv("final_combined_output.csv", index=False)
print("All outputs saved to 'final_combined_output.csv'")



'''

import matplotlib.pyplot as plt

def compare_predictions(y_test, y_pred, target_variable):
    """
    Compare predictions with actual data and visualize the results.

    Parameters:
    y_test (pd.Series): Actual target values.
    y_pred (np.ndarray): Predicted target values.
    target_variable (str): Name of the target variable.
    """
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })
    
    # Save the comparison to CSV
    comparison_df.to_csv(f"comparison_{target_variable}.csv", index=False)
    print(f"Comparison data for {target_variable} saved to 'comparison_{target_variable}.csv'")
    
    # Plot the actual vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Line y=x
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted for {target_variable}")
    plt.grid(True)
    plt.show()


# Example usage inside the loop
for target in target_variables:
    y_train = train_data[target]
    y_test = test_data[target]
    y_pred_test = xgb_model.predict(X_test)
    
    compare_predictions(y_test, y_pred_test, target)
'''

'''
# Combine data for output
for target in target_variables:

    ...
    # Get corresponding y_test and predictions for the target
    y_test = test_data[target]
    y_pred = xgb_model.predict(X_test)
    
    # Create a DataFrame with relevant columns
    combined_output = pd.DataFrame({
        "days_since_zero": test_data["days_since_zero"].values,
        "fips": test_data["fips"].values,
        "y_test": y_test.values,
        "y_pred": y_pred
    })
    
    # Save to CSV
    combined_output.to_csv(f"combined_output_{target}.csv", index=False)
    print(f"Combined output for {target} saved to 'combined_output_{target}.csv'")
    

'''


'''
# Compute adjusted predictions per 100k
for target in target_variables:
    # Predicted values
    y_pred = final_combined_output[f"y_pred_{target}"]

    # Compute mean and variance of predicted values
    y_pred_mean = y_pred.mean()
    y_pred_var = y_pred.var()

    # Adjusted per 100k calculation using Taylor expansion
    final_combined_output[f"y_pred_{target}_per_100k"] = (
        (y_pred * 100000 / final_combined_output["population"])
        + (y_pred_var / (2 * (final_combined_output["population"] ** 2) * y_pred_mean))
    )

# Ensure `cases_per_100k` is included in the output
final_combined_output["cases_per_100k"] = test_data["cases_per_100k"].values
# Add training results to the final output
train_combined_output = train_data[["days_since_zero", "fips"]].copy()
train_combined_output["population"] = train_data["population"].values
for target in target_variables:
    # Include y_train and y_pred_train
    train_combined_output[f"y_train_{target}"] = train_data[target].values
    train_combined_output[f"y_pred_train_{target}"] = xgb_model.predict(X_train)

    # Adjusted per 100k calculation for training data
    y_pred_train = train_combined_output[f"y_pred_train_{target}"]
    y_pred_train_mean = y_pred_train.mean()
    y_pred_train_var = y_pred_train.var()

    train_combined_output[f"y_pred_train_{target}_per_100k"] = (
        (y_pred_train * 100000 / train_combined_output["population"])
        + (y_pred_train_var / (2 * (train_combined_output["population"] ** 2) * y_pred_train_mean))
    )

'''