# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score







'''



'''









# Load data
file_path = 'us.csv'  # Replace with your file path
total_us_data = pd.read_csv(file_path)

# Preprocess data
total_us_data['date'] = pd.to_datetime(total_us_data['date'])
total_us_data.set_index('date', inplace=True)
total_us_data['daily_cases'] = total_us_data['cases'].diff().fillna(0)

# Split data into training and testing sets
split_date = '2022-09-01'
train_data = total_us_data.loc[:split_date, 'daily_cases']
test_data = total_us_data.loc[split_date:, 'daily_cases']



'''
from statsmodels.tsa.stattools import adfuller

adf_results = {
    'Test Statistic': adf_test[0],
    'p-value': adf_test[1],
    'Lags Used': adf_test[2],
    'Number of Observations': adf_test[3],
    'Critical Values': adf_test[4]
}

print(adf_results)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF for daily cases
plt.figure(figsize=(12, 6))

# ACF plot
plt.subplot(1, 2, 1)
plot_acf(total_us_data['daily_cases'], lags=40, ax=plt.gca())
plt.title('ACF of Daily Cases')

# PACF plot
plt.subplot(1, 2, 2)
plot_pacf(total_us_data['daily_cases'], lags=40, ax=plt.gca(), method='ywm')
plt.title('PACF of Daily Cases')

plt.tight_layout()
plt.show()


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Tentative SARIMA parameters based on ACF/PACF analysis
# p, d, q = 2, 0, 2 (regular components)
# P, D, Q, s = 1, 1, 1, 7 (seasonal components with weekly seasonality)
sarima_model = SARIMAX(
    total_us_data['daily_cases'],
    order=(2, 0, 2),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

# Fit the model
sarima_fit = sarima_model.fit(disp=False)

# Display model summary
sarima_fit.summary()


 SARIMAX Results                                      
===========================================================================================
Dep. Variable:                         daily_cases   No. Observations:                 1158
Model:             SARIMAX(2, 0, 2)x(1, 1, [1], 7)   Log Likelihood              -13855.860
Date:                             Thu, 05 Dec 2024   AIC                          27725.720
Time:                                     20:24:12   BIC                          27760.997
Sample:                                 01-21-2020   HQIC                         27739.042
                                      - 03-23-2023                                         
Covariance Type:                               opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          1.7103      0.014    121.582      0.000       1.683       1.738
ar.L2         -0.7333      0.014    -52.699      0.000      -0.761      -0.706
ma.L1         -1.6542      0.015   -109.797      0.000      -1.684      -1.625
ma.L2          0.8122      0.012     69.205      0.000       0.789       0.835
ar.S.L7       -0.3737      0.031    -12.027      0.000      -0.435      -0.313
ma.S.L7       -0.0993      0.033     -2.974      0.003      -0.165      -0.034
sigma2      2.484e+09   8.77e-12   2.83e+20      0.000    2.48e+09    2.48e+09
===================================================================================
Ljung-Box (L1) (Q):                   1.58   Jarque-Bera (JB):            113309.78
Prob(Q):                              0.21   Prob(JB):                         0.00
Heteroskedasticity (H):               1.88   Skew:                            -1.98
Prob(H) (two-sided):                  0.00   Kurtosis:                        51.66
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
[2] Covariance matrix is singular or near-singular, with condition number 3.56e+35. Standard errors may be unstable.
"""
The SARIMA model was successfully fitted to the daily COVID-19 cases data with the following parameters:

Order (p, d, q): (2, 0, 2)
Seasonal Order (P, D, Q, s): (1, 1, 1, 7)


'''





# Train SARIMA model
sarima_model = SARIMAX(
    train_data,
    order=(2, 0, 2),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_fit = sarima_model.fit(disp=False)

# Forecast for the testing period
forecast_steps_test = len(test_data)
forecast_test = sarima_fit.get_forecast(steps=forecast_steps_test)
forecast_test_series = pd.Series(forecast_test.predicted_mean, index=test_data.index)

# Adjust predictions
adjusted_forecast = forecast_test_series - 1e4

# Evaluate metrics on the testing period
aligned_test_data = test_data.dropna()

aligned_forecast_data = adjusted_forecast.loc[aligned_test_data.index]

aligned_test_data = aligned_test_data.dropna()
aligned_forecast_data = aligned_forecast_data.dropna()

#alligh length
common_index = aligned_test_data.index.intersection(aligned_forecast_data.index)
aligned_test_data = aligned_test_data.loc[common_index]
aligned_forecast_data = aligned_forecast_data.loc[common_index]




# Check model fit on the training data
train_predictions = sarima_fit.get_prediction(start=train_data.index[0], end=train_data.index[-1])
train_pred_mean = train_predictions.predicted_mean

# Evaluate metrics on training data
mae_train = mean_absolute_error(train_data, train_pred_mean)
rmse_train = mean_squared_error(train_data, train_pred_mean, squared=False)
mape_train = (abs(train_data - train_pred_mean) / train_data).mean() * 100
r2_train = r2_score(train_data, train_pred_mean)

# Metrics for training data
train_metrics = {
    "Mean Absolute Error (MAE)": mae_train,
    "Root Mean Squared Error (RMSE)": rmse_train,
    "Mean Absolute Percentage Error (MAPE)": mape_train,
    "R-Squared (R2)": r2_train,
}
print("Metrics on Training Data:")
print(train_metrics)

# Visualize training predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Observed (Train Data)', color='blue', linewidth=1.5)
plt.plot(train_pred_mean, label='Predicted (Train Data)', color='orange', linestyle='--', linewidth=1.5)
plt.title('Observed vs Predicted Daily Cases (Training Data)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Cases', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



















mae = mean_absolute_error(aligned_test_data, aligned_forecast_data)




rmse = mean_squared_error(aligned_test_data, aligned_forecast_data, squared=False)
mape = (abs(aligned_test_data - aligned_forecast_data) / aligned_test_data).mean() * 100
r2 = r2_score(aligned_test_data, aligned_forecast_data)

metrics = {
    "Mean Absolute Error (MAE)": mae,
    "Root Mean Squared Error (RMSE)": rmse,
    "Mean Absolute Percentage Error (MAPE)": mape,
    "R-Squared (R2)": r2,
}
print("Metrics on Full Testing Data:", metrics)

# Evaluate metrics for September to October 2022
detailed_start = '2022-09-01'
detailed_end = '2022-10-31'
detailed_test_data = aligned_test_data.loc[detailed_start:detailed_end]
detailed_forecast = aligned_forecast_data.loc[detailed_start:detailed_end]

mae_detailed = mean_absolute_error(detailed_test_data, detailed_forecast)
rmse_detailed = mean_squared_error(detailed_test_data, detailed_forecast, squared=False)
mape_detailed = (abs(detailed_test_data - detailed_forecast) / detailed_test_data).mean() * 100
r2_detailed = r2_score(detailed_test_data, detailed_forecast)

detailed_metrics = {
    "Mean Absolute Error (MAE)": mae_detailed,
    "Root Mean Squared Error (RMSE)": rmse_detailed,
    "Mean Absolute Percentage Error (MAPE)": mape_detailed,
    "R-Squared (R2)": r2_detailed,
}
print("Metrics for Sep-Oct 2022:", detailed_metrics)












# Plot observed vs adjusted forecasted values for full testing period
plt.figure(figsize=(12, 6))
plt.plot(aligned_test_data, label='Observed (Test Data)', color='blue')
plt.plot(aligned_forecast_data, label='Adjusted Forecasted', color='orange')
plt.title('Observed vs Adjusted Forecasted Daily Cases (Test Period)')
plt.xlabel('Date')
plt.ylabel('Daily Cases (x 1e5)')
plt.ylim([-1e5, 5e5])
plt.legend()
plt.tight_layout()
plt.show()

# Plot observed vs adjusted forecasted values for September to October 2022
plt.figure(figsize=(12, 6))
plt.plot(detailed_test_data, label='Observed (Sep-Oct 2022)', color='blue')
plt.plot(detailed_forecast, label='Adjusted Forecasted (Sep-Oct 2022)', color='orange')
plt.title('Observed vs Adjusted Forecasted Daily Cases (Sep-Oct 2022)')
plt.xlabel('Date')
plt.ylabel('Daily Cases (x 1e5)')
plt.ylim([-1e5, 5e5])
plt.legend()
plt.tight_layout()
plt.show()

'''
Metrics on Training Data:
{'Mean Absolute Error (MAE)': np.float64(19875.309261785642), 'Root Mean Squared Error (RMSE)': np.float64(47960.804360673785), 'Mean Absolute Percentage Error (MAPE)': np.float64(inf), 'R-Squared (R2)': 0.8788884586216319}
C:\Users\haoyu\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
  warnings.warn(
Metrics on Full Testing Data: {'Mean Absolute Error (MAE)': np.float64(35593.31126893882), 'Root Mean Squared Error (RMSE)': np.float64(43057.81498247143), 'Mean Absolute Percentage Error (MAPE)': np.float64(233.75662788604618), 'R-Squared (R2)': 0.006364513379461423}
C:\Users\haoyu\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_regression.py:492: FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6. To calculate the root mean squared error, use the function'root_mean_squared_error'.
  warnings.warn(
Metrics for Sep-Oct 2022: {'Mean Absolute Error (MAE)': np.float64(32396.04193394592), 'Root Mean Squared Error (RMSE)': np.float64(39008.01575352829), 'Mean Absolute Percentage Error (MAPE)': np.float64(151.7767898369961), 'R-Squared (R2)': -0.14081369416093104}

C:\ML>
'''