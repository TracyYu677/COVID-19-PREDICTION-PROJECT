### README for "Machine Learning-Based Prediction on COVID-19 Case Numbers"

---

# **Machine Learning-Based Prediction on COVID-19 Case Numbers**  

**Authors:** Joyce Gan, Zimeng Gu, Hao Yu, Xieqing Yu  
**Purpose:** Predicting COVID-19 case numbers in the U.S. using machine learning models to enhance public health planning and resource allocation.  

---

## **Project Overview**  

Coronavirus disease (COVID-19) has profoundly impacted global public health and economies. This project focuses on predicting daily case trends and counts in U.S. counties using demographic, temporal, and epidemiological data. Our models aim to assist in:  
- **Short-term outbreak detection:** Enabling targeted government interventions.  
- **Long-term resource planning:** Providing insights into seasonal and spatial patterns for healthcare management.

---

## **Key Features**  

1. **Data Cleaning and Preparation:**
   - Data from *The New York Times* COVID-19 repository.
   - Normalization per 100,000 population for fair comparisons.
   - Incorporation of additional features like population density, healthcare resources, and mobility indices.

2. **Models Used:**
   - **XGBoost:** Handles nonlinear relationships and missing values; tuned for short-term and long-term predictions.
   - **Spatial-Temporal Gaussian Process (ST-GP):** Combines spatial and temporal components for county-level predictions, capturing neighborhood influences and seasonality.

3. **Interactive Visualization:**
   - R Shiny app for exploring spatiotemporal dynamics, viewing trends, and comparing model predictions.

4. **Evaluation Metrics:**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)
   - Coefficient of Determination (\(R^2\))

---

## **Models and Methodologies**

### **XGBoost**  
- **Purpose:**  Quickly predict trends of daily case changes per 100,000 population.  
- **Features:** Incorporates epidemiological patterns and uses rolling windows for model retraining.  
- **Performance:** Achieved \(R^2 \geq 0.85\) for short-term predictions, declining to ~0.5 for long-term forecasts.  
- **Validation:** Used Optuna for hyperparameter tuning and time-series cross-validation.  

### **Spatial-Temporal Gaussian Process**  
- **Purpose:**  Accurately predict daily change in cases.    
- **Kernel Design:** Combines spatial (RBF kernel) and temporal (autoregressive components) to model dependencies.  
- **Performance:** High accuracy for 30-day forecasts with metrics such as MAE = 0.048, RMSE = 0.0034, and \(R^2\) ≈ 0.999999.  

---

## **Dataset Details**  

- **Data Source:** COVID-19 daily case counts and deaths by U.S. counties (The New York Times).  [here](https://www.kaggle.com/datasets/paultimothymooney/nytimes-covid19-data?select=README.md)
- **Time Period:** January 2020 - May 2022.  
- **Variables:**  
  - Temporal: `date`, `days_since_zero`, `cases_last_week`, etc.  
  - Spatial: `latitude`, `longitude`, `neighbor_population_sum`.  
  - Epidemiological: `daily_change_per_100k`, `mobility_index`, `population_density`, `total_facility_bed`.  

Refer to the [detailed variable descriptions](#variable-descriptions) for more information.

---
### **Installation Instructions**

This project involves two components: **Python-based XGBoost modeling** and **R-based spatial-temporal Gaussian Process (ST-GP) modeling with a Shiny app**. Follow the steps below to set up your environment.

---

### **Python Environment Setup**

#### **Dependencies**
The Python code for this project requires the following libraries:
- `pandas`
- `scikit-learn`
- `xgboost`
- `statsmodels`
- `matplotlib`
- `openpyxl`

#### **Installation Steps**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/covid19-prediction.git
   cd covid19-prediction
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas scikit-learn xgboost statsmodels matplotlib openpyxl
   ```

4. **Verify installation:**
   Test the Python setup by running:
   ```bash
   python xgboost_model.py
   ```

---

### **R Environment Setup**

#### **Dependencies**
The R code for this project requires the following packages:
- `shiny`
- `leaflet`
- `sf`
- `dplyr`
- `lubridate`
- `ggplot2`
- `plotly`

#### **Installation Steps**
1. **Install R and RStudio:**
   - Download R from [CRAN](https://cran.r-project.org/).
   - Download RStudio from [RStudio](https://www.rstudio.com/).

2. **Install required R packages:**
   Open RStudio and run:
   ```R
   install.packages(c("shiny", "leaflet", "sf", "dplyr", "lubridate", "ggplot2", "plotly"))
   ```

3. **Run the Shiny app:**
   Navigate to the `rshiny_app/` directory and open the `app.R` file in RStudio. Then click **Run App** to launch the interactive visualization.

---

### **Notes**
- Ensure your Python version is 3.7+ and R version is 4.0+ for compatibility.
- Install additional packages or libraries if prompted during execution.
- If using a cloud platform for hosting the R Shiny app or Python scripts, configure the deployment environment accordingly.  

---


## **Visualization**  

### Interactive R Shiny App  
Access the app [here](https://jxygan.shinyapps.io/final_plot_app/) to:  
- Explore trends dynamically by entering specific FIPS codes.  
- View heatmaps and county-specific time series comparisons.  

### Sample Outputs:  
- **XGBoost Predictions:**  
![Visualization of COVID-19 Predictions](https://github.com/HaoYu2024/COVID-19-PREDICTION-PROJECT/blob/main/data%20cleaning%2C%20processing%20and%20modeling%20for%20XGBoost%20and%20Sarimax%20method/xgboost/visualization/Autauga.png)


- **ST-GP Predictions:**  
![ST-GP Plot](https://github.com/HaoYu2024/COVID-19-PREDICTION-PROJECT/blob/main/data%20cleaning%2C%20processing%20and%20modeling%20for%20Spatial-temporal%20Gaussian%20Process/plumas.png)


---
### **Evaluation and Results**

#### **XGBoost Results**  
The XGBoost model was deployed to predict daily COVID-19 case changes per 100,000 population across U.S. counties. Key outcomes include:  
- **General Trends and Tactical Predictions:**
  - The model excelled in predicting general trends and case changes, particularly for short-term intervals (1-30 days).
  - Robustness was achieved using a rolling window retraining approach, where the most recent data was weighted higher, adapting to emerging patterns.  
- **Performance Metrics:**
  - Achieved high accuracy with \(R^2 \geq 0.85\) for short-term predictions and RMSE under 200 for daily changes per 100,000 people.
  - For long-term forecasts (up to 6 months), the model’s performance declined as expected but still maintained reasonable accuracy (\(R^2 \approx 0.5\)).
- **Validation Observations:**
  - Time-series cross-validation revealed interesting seasonal and temporal trends. For instance, summer test periods showed stronger performance compared to others.
  - Removing seasonal trends with SARIMAX did not significantly improve the overall \(R^2\).

#### **Spatial-Temporal Gaussian Process (ST-GP) Results**  
The Spatial-Temporal Gaussian Process (ST-GP) model was applied to California counties, leveraging both spatial and temporal features. Key outcomes include:  
- **Short-Term Predictions:**  
  - The model demonstrated exceptional accuracy for predicting daily case changes 30 days ahead.
  - Integration of external covariates such as population density, healthcare resource allocation, and mobility indices significantly improved predictions.  
- **Performance Metrics:**  
  - **MAE:** 0.048  
  - **RMSE:** 0.0034  
  - **MAPE:** 0.5765%  
  - \(R^2\): ~0.999999, indicating the model explained nearly all the variability in daily case changes.  
- **Spatial and Temporal Analysis:**  
  - Visual comparison of actual vs. predicted cases across counties (e.g., Plumas and Mariposa) showed high alignment for short-term predictions.  
  - The inclusion of mobility indices and vaccination changes allowed the model to capture dynamic local trends effectively.  

---

### **Key Findings**
- **XGBoost Strengths:**  
  - Best suited for general trends and tactical decision-making at county or regional levels.  
  - Highly adaptable to emerging patterns with rolling window retraining, making it a reliable tool for dynamic scenarios.  
- **ST-GP Strengths:**  
  - Provided nuanced spatial and temporal insights, especially useful for localized, short-term planning.  
  - Leveraged external covariates to refine predictions, demonstrating the importance of contextual data in spatial-temporal modeling.  
- **Applications:**  
  - These models can support public health officials by identifying high-risk counties, optimizing resource allocation, and informing targeted interventions.  

By combining these methods, this study demonstrates the power of machine learning for pandemic forecasting and highlights avenues for further research, such as expanding ST-GP to nationwide datasets and enhancing long-term prediction accuracy.n of external covariates (e.g., mobility, vaccination rates) enhanced precision.

---

## **Conclusion**  

Machine learning provides robust tools for COVID-19 forecasting. Key takeaways:  
- **XGBoost:** Efficient for trend analysis and resource allocation.  
- **ST-GP:** Advanced spatiotemporal modeling enhances prediction accuracy at local scales.  

Future directions include extending to localized, block-level models for quarantine planning and exploring scalable Gaussian Process methods for broader applications.  

---

## **Acknowledgments**  

Data sources include:  
- The New York Times COVID-19 Data Repository  
- World Health Organization  
- California State Open Data Portal  

---

### **Variable Descriptions**

| **Variable**                  | **Description**                                                                                  |
|-------------------------------|--------------------------------------------------------------------------------------------------|
| `date`                        | Specific date of data collection.                                                                |
| `county_x`                    | County name.                                                                                     |
| `fips`                        | County-specific Federal Information Processing Standards code.                                   |
| `cases`                       | Total confirmed or probable COVID-19 cases.                                                      |
| `deaths`                      | Total confirmed or probable COVID-19 deaths.                                                     |
| `population`                  | Total county population.                                                                         |
| `days_since_zero`             | Days since the first recorded case in the dataset.                                               |
| `cases_per_100k`              | Weekly cases per 100,000 population.                                                             |
| `neighbor_population_sum`     | Sum of populations of the four nearest counties.                                                 |
| `population_density`          | County population density.                                                                       |
| `total_facility_bed`          | Total hospital beds authorized by the California Health Department.                              |
| `mobility_index`              | Aggregate mobility percentage changes across categories (e.g., work, recreation).                |

---
