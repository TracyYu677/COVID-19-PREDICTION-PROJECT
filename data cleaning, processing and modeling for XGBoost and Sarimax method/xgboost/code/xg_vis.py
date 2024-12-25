#xg visulization

import pandas as pd

# Load the uploaded file
file_path = 'final_combined_output.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the data
data.head()
import matplotlib.pyplot as plt

# Filter data for fips=1001
data_1001 = data[data['fips'] == 1001]

# Calculate the average of predictions across the models
prediction_columns = [
    'y_pred_case_tomorrow', 
    'y_pred_case_next_week', 
    'y_pred_case_next_month', 
    'y_pred_case_next_3month', 
    'y_pred_case_next_6month'
]
data_1001['y_pred_case_average'] = data_1001[prediction_columns].mean(axis=1)

# Plot actual vs predictions
plt.figure(figsize=(12, 6))

# Plot actual values
plt.plot(data_1001['days_since_zero'], data_1001['y_test'], label='Actual', marker='o')

# Plot predictions for each model
for col in prediction_columns:
    plt.plot(data_1001['days_since_zero'], data_1001[col], label=col.replace('_', ' ').title(), linestyle='--')

# Plot average prediction
plt.plot(data_1001['days_since_zero'], data_1001['y_pred_case_average'], label='Average Prediction', linestyle='-', linewidth=2)

# Adding labels and legend
plt.title('Actual vs Predicted Values for FIPS=1001')
plt.xlabel('Days Since Zero')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()


# Filter data for fips=18057
data_18057 = data[data['fips'] == 18057]

# Calculate the average of predictions across the models
data_18057['y_pred_case_average'] = data_18057[prediction_columns].mean(axis=1)

# Plot actual vs predictions
plt.figure(figsize=(12, 6))

# Plot actual values
plt.plot(data_18057['days_since_zero'], data_18057['y_test'], label='Actual', marker='o')

# Plot predictions for each model
for col in prediction_columns:
    plt.plot(data_18057['days_since_zero'], data_18057[col], label=col.replace('_', ' ').title(), linestyle='--')

# Plot average prediction
plt.plot(data_18057['days_since_zero'], data_18057['y_pred_case_average'], label='Average Prediction', linestyle='-', linewidth=2)

# Adding labels and legend
plt.title('Actual vs Predicted Values for FIPS=18057')
plt.xlabel('Days Since Zero')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()

# Calculate rolling averages for actual and predicted values (window size = 7 days)
rolling_window = 7
data_18057_rolling = data_18057.copy()
data_18057_rolling['y_test_rolling'] = data_18057['y_test'].rolling(rolling_window).mean()

for col in prediction_columns:
    data_18057_rolling[f'{col}_rolling'] = data_18057[col].rolling(rolling_window).mean()

# Plot rolling averages
plt.figure(figsize=(12, 6))

# Plot rolling average of actual values
plt.plot(data_18057_rolling['days_since_zero'], data_18057_rolling['y_test_rolling'], 
         label='Actual (Rolling Avg)', marker='o')

# Plot rolling averages for each model
for col in prediction_columns:
    plt.plot(data_18057_rolling['days_since_zero'], data_18057_rolling[f'{col}_rolling'], 
             label=col.replace('_', ' ').title() + ' (Rolling Avg)', linestyle='--')

# Adding labels and legend
plt.title('Rolling Averages of Actual vs Predicted Values for FIPS=18057')
plt.xlabel('Days Since Zero')
plt.ylabel('Rolling Average Values')
plt.legend()
plt.grid()
plt.show()
