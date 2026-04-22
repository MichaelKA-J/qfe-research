# Technical Documentation: Econ 525 Final Project
**Project:** ECON 525 Final Project   
**Authors:** Michael Jaberi, Henry Rutsch, Charles Harris  
**Environment:** Python 3.x

## 1. Overview
This script tests the GARCH(1,1) model against the GJR-GARCH model for forecasting the volatility of Brent Crude Oil (LCOc1) daily returns. It includes:
*   **Returns Series Summary Statistics:** Descriptive analysis (Mean, Volatility, Skewness, Kurtosis).
*   **In-Sample Estimation:** Comparison between a baseline GARCH(1,1) and an asymmetric GJR-GARCH(1,1) with Student’s t-distribution.
*   **Backtesting of Both Models:** A 250-day Out-of-Sample (OOS) rolling window forecast.
*   **Model Performance Metrics:** MFE, RMSFE, and OOS $R^2$.
*   **Clark-West Test:** to statistically verify if the GJR-GARCH model improves upon the nested GARCH baseline.

## 2. Prerequisites
The script requires Python 3 and the following libraries. You can install them via terminal using:

```bash
pip install numpy pandas matplotlib scipy arch
```

*   **`arch`**: Required for the `arch_model` functionality.
*   **`pandas`**: Used for data handling and remote CSV retrieval.

## 3. Data Source and Auxiliary Files
The code is designed for reproducibility and automatically pulls the cleaned dataset directly from a remote GitHub repository. 

**Note to Grader:** 
*   While a spreadsheet containing the raw price data is included in the submission zip file for your reference, the code does **not** require it to run, as it fetches the processed returns automatically. 
*   Additionally, we have included a file named `data_cleaning.py`. This script contains the code used to pull raw price series from LSEG via API and transform it into the clean returns series used in the main analysis. This is provided for transparency to show our data processing methodology; you do **not** need to run this file.

Please ensure you have an active internet connection when running the main script.
