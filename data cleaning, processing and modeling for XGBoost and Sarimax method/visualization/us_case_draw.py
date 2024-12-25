import pandas as pd

# Load the uploaded file
file_path = 'us.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand its structure
data.head()


import matplotlib.pyplot as plt

# Convert the date column to datetime format for plotting
data['date'] = pd.to_datetime(data['date'])

# Calculate daily change in cases
data['daily_change_cases'] = data['cases'].diff().fillna(0)

# Plot the daily change in cases
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['daily_change_cases'], label='Daily Change in Cases', alpha=0.7)
plt.title('Daily Change in COVID-19 Cases Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Change in Cases', fontsize=12)
plt.grid(True, alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()



# Calculate daily change in deaths
data['daily_change_deaths'] = data['deaths'].diff().fillna(0)

# Plot the daily change in deaths
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['daily_change_deaths'], label='Daily Change in Deaths', color='red', alpha=0.7)
plt.title('Daily Change in COVID-19 Deaths Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Change in Deaths', fontsize=12)
plt.grid(True, alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
