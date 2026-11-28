import os
import pandas as pd
import numpy as np

def build_features(input_path, output_path):
    print("=" * 80)
    print("STEP 04: FEATURE ENGINEERING FOR ENERGY CONSUMPTION PREDICTION")
    print("=" * 80)
    
    # 1. Load Data
    print(f"\n[1/5] Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    print(f"       Raw rows loaded: {len(df):,}")
    
    # 2. Data Cleaning: Deduplication & Continuous Hourly Index
    print("\n[2/5] Cleaning data: Deduplicating and filling missing timestamps...")
    # Average duplicate hours (Daylight Saving Time duplicates)
    df = df.groupby('Datetime', as_index=False)['PJME_MW'].mean()
    
    # Create complete hourly continuous range
    min_time = df['Datetime'].min()
    max_time = df['Datetime'].max()
    full_index = pd.date_range(start=min_time, end=max_time, freq='h')
    
    df = df.set_index('Datetime').reindex(full_index)
    df.index.name = 'Datetime'
    
    # Interpolate the ~30 missing hours so lag steps are strictly continuous
    missing_count = df['PJME_MW'].isna().sum()
    print(f"       Interpolating {missing_count} missing hours for strict hourly continuity...")
    df['PJME_MW'] = df['PJME_MW'].interpolate(method='linear')
    df = df.reset_index()
    
    # 3. Calendar & Cyclical Features
    print("\n[3/5] Extracting Calendar & Cyclical (Sine/Cosine) features...")
    df['hour'] = df['Datetime'].dt.hour
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['month'] = df['Datetime'].dt.month
    df['dayofyear'] = df['Datetime'].dt.dayofyear
    df['quarter'] = df['Datetime'].dt.quarter
    df['year'] = df['Datetime'].dt.year
    
    # Cyclical Sine/Cosine
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 4. Lag Features & Rolling Window Statistics
    print("\n[4/5] Computing Autoregressive Lag Features & Rolling Window Stats...")
    # Lags: 1 hour, 2 hours, 24 hours (yesterday), 48 hours, 168 hours (last week)
    df['lag_1'] = df['PJME_MW'].shift(1)
    df['lag_2'] = df['PJME_MW'].shift(2)
    df['lag_24'] = df['PJME_MW'].shift(24)
    df['lag_48'] = df['PJME_MW'].shift(48)
    df['lag_168'] = df['PJME_MW'].shift(168)
    
    # Rolling stats (shift(1) strictly avoids lookahead leakage)
    df['rolling_mean_24h'] = df['PJME_MW'].shift(1).rolling(window=24).mean()
    df['rolling_std_24h'] = df['PJME_MW'].shift(1).rolling(window=24).std()
    df['rolling_mean_7d'] = df['PJME_MW'].shift(1).rolling(window=168).mean()
    
    # Drop rows containing NaN due to lag_168
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)
    print(f"       Dropped {dropped} initial rows (due to 168h lag window). Final rows: {len(df):,}")
    
    # 5. Summary & Save
    print(f"\n[5/5] Saving engineered dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE! SUMMARY OF GENERATED FEATURES:")
    print("=" * 80)
    print(f"Total Rows:     {len(df):,}")
    print(f"Total Columns:  {len(df.columns)}")
    print(f"Features list:  {list(df.columns)}")
    
    print("\n--- Correlation of Top Features with PJME_MW (Target) ---")
    corr = df.drop(columns=['Datetime']).corr()['PJME_MW'].sort_values(ascending=False)
    print(corr.to_string())
    
    print("\n--- Sample Preview (First 3 Rows) ---")
    print(df[['Datetime', 'PJME_MW', 'hour', 'month', 'lag_1', 'lag_24', 'rolling_mean_24h']].head(3).to_string())
    print("=" * 80)
    
    return df

if __name__ == "__main__":
    input_file = "Predict_Energy_Consumption/PJM_Hourly_Energy_Cosumption_Dataset/PJME_hourly.csv"
    output_file = "Predict_Energy_Consumption/pjme_features.csv"
    build_features(input_file, output_file)
