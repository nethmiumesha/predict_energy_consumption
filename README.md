# Predict Energy Consumption Using Machine Learning & Time Factors

This project predicts regional hourly energy consumption (in Megawatts) using historical time-series data from PJM Interconnection and machine learning regression models.

## 📊 Dataset Overview
The dataset contains hourly energy consumption data from PJM Interconnection (an East Coast US regional transmission organization):
- **Primary Region**: `PJME_hourly.csv` (PJM East Region, 2002–2018, ~145,000+ hourly records).
- **Other Regions**: AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJMW.

## 🚀 Key Features & Approach
- **Time Factors**: Hour of day, day of week, month, day of year, weekend flag, quarter.
- **Cyclical Encoding**: Sine and Cosine transformations for hour and month to preserve circular temporal continuity.
- **Autoregressive Lags**: Energy consumption at $t-1$ (1 hour ago), $t-24$ (yesterday same hour), and $t-168$ (last week same hour).
- **Rolling Aggregations**: 24-hour moving average and volatility.
- **Validation**: Strict chronological split (training on past data, evaluating on future unseen data) to prevent data leakage.
- **Models**: Linear/Ridge Regression, Random Forest, Gradient Boosting, XGBoost / LightGBM.

## 🛠️ Project Structure
```text
Predict_Energy_Consumption/
├── PJM_Hourly_Energy_Cosumption_Dataset/  # Raw hourly CSV datasets
├── .gitignore
└── README.md
```

## 📈 Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **$R^2$ Score**
