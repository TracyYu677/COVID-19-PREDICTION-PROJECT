#autocorrelation study

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

# Load your data (replace 'your_file_path.xlsx' with your file path)
data = pd.read_csv('processed_covid_data.csv')

# Step 1: Encode 'risk_level' to numeric values
# Map risk levels (categorical) to numeric values
risk_level_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
data['risk_level'] = data['risk_level'].map(risk_level_mapping)


'''
# Step 2: Align counties by the start of the pandemic
aligned_autocorrelations = []
max_lag = 30  # Define the maximum lag to compute

for county, county_data in data.groupby('county_x'):  # Replace 'county_x' with your county column
    # Filter and sort by date for this county
    county_data = county_data.sort_values('date')
    
    # Define the start of the pandemic as the first non-zero risk level
    county_data = county_data[county_data['risk_level'].notna()]
    
    # Align the timeline
    county_data = county_data.reset_index(drop=True)
    risk_levels = county_data['risk_level']
    
    # Compute autocorrelations for this county
    autocorrelations = [risk_levels.autocorr(lag) for lag in range(1, max_lag + 1)]
    aligned_autocorrelations.append(autocorrelations)

# Step 3: Compute the average autocorrelation across all counties
average_autocorrelations = np.nanmean(aligned_autocorrelations, axis=0)


# Step 4: Plot the averaged autocorrelation
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_lag + 1), average_autocorrelations, marker='o', label='Average Autocorrelation')
plt.axhline(0.8, color='red', linestyle='--', label='0.8 Threshold')
plt.title('Average Autocorrelation of Risk Levels Across Counties')
plt.xlabel('Lag (days)')
plt.ylabel('Autocorrelation')
plt.legend()
plt.show()
'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Step 5: Prepare the data
# Ensure the data is sorted and filtered by county and date
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['county_x', 'date'])

# Create lagged features (3 previous days as inputs)
for lag in range(1, 4):
    data[f'risk_level_lag_{lag}'] = data.groupby('county_x')['risk_level'].shift(lag)

for lag in range(1, 4):
    data[f'risk_level_lag_{lag}'] = data[f'risk_level_lag_{lag}'].fillna(data['risk_level'])

# Split into training and testing datasets
train_data = data[data['date'] <= '2020-05-01']
test_data = data[data['date'] > '2020-05-01']

# Define feature columns and target
feature_cols = [f'risk_level_lag_{lag}' for lag in range(1, 4)]
X_train, y_train = train_data[feature_cols], train_data['risk_level']
X_test, y_test = test_data[feature_cols], test_data['risk_level']
'''
#file for check
train_data.to_csv('log_train_data.csv', index=False)
test_data.to_csv('log_test_data.csv', index=False)
'''



# Step 6: Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Predict the test data
test_data['predicted_risk_level'] = model.predict(X_test)

# Step 8: Evaluate the model
def calculate_metrics(predictions, actual, time_period):
    rss = np.sum((predictions - actual) ** 2)
    r2 = r2_score(actual, predictions)
    return rss, r2

# Evaluate in different time periods
time_frames = {
    "First week": (test_data['date'] <= '2020-05-08'),
    "First month": (test_data['date'] <= '2020-06-01'),
    "First 3 months": (test_data['date'] <= '2020-08-01'),
    "Total": slice(None)
}

results = {}
for period, mask in time_frames.items():
    filtered_data = test_data[mask]
    rss, r2 = calculate_metrics(filtered_data['predicted_risk_level'], filtered_data['risk_level'], period)
    results[period] = {'RSS': rss, 'R^2': r2}

# Show the results
import ace_tools_open as tools; tools.display_dataframe_to_user(name="Evaluation Metrics for Prediction Performance", dataframe=pd.DataFrame(results).T)

# Step 9: Plot actual vs predicted cases
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], test_data['risk_level'], label='Actual Risk Level', alpha=0.7)
plt.plot(test_data['date'], test_data['predicted_risk_level'], label='Predicted Risk Level', alpha=0.7)
plt.title('Actual vs Predicted Risk Levels')
plt.xlabel('Date')
plt.ylabel('Risk Level')
plt.legend()
plt.grid(True)
plt.show()

