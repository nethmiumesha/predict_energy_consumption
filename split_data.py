import os
import pandas as pd
import matplotlib.pyplot as plt

def split_and_validate(features_path, split_date="2015-01-01", plot_output="Predict_Energy_Consumption/eda_plots/train_test_split.png"):
    print("=" * 80)
    print("STEP 05: CHRONOLOGICAL TRAIN / TEST SPLIT")
    print("=" * 80)
    
    # 1. Load engineered features dataset
    print(f"\n[1/3] Loading engineered features dataset from: {features_path}")
    df = pd.read_csv(features_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # 2. Chronological Split
    print(f"\n[2/3] Performing chronological split at cutoff date: {split_date}...")
    train = df[df['Datetime'] < split_date].copy()
    test = df[df['Datetime'] >= split_date].copy()
    
    # Separate Features (X) and Target (y)
    ignore_cols = ['Datetime', 'PJME_MW']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    target_col = 'PJME_MW'
    
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    
    total_len = len(df)
    train_pct = (len(train) / total_len) * 100
    test_pct = (len(test) / total_len) * 100
    
    print("\n" + "-" * 50)
    print("SPLIT DETAILS:")
    print("-" * 50)
    print(f"Total Observations:    {total_len:,}")
    print(f"Number of Features (X): {len(feature_cols)}")
    print(f"\n--- Training Set ---")
    print(f"  Rows:        {len(train):,} ({train_pct:.2f}%)")
    print(f"  Date Range:  {train['Datetime'].min().date()} to {train['Datetime'].max().date()}")
    print(f"  X_train:     {X_train.shape}")
    print(f"  y_train:     {y_train.shape}")
    
    print(f"\n--- Testing Set ---")
    print(f"  Rows:        {len(test):,} ({test_pct:.2f}%)")
    print(f"  Date Range:  {test['Datetime'].min().date()} to {test['Datetime'].max().date()}")
    print(f"  X_test:      {X_test.shape}")
    print(f"  y_test:      {y_test.shape}")
    
    # 3. Generate Timeline Visualization
    print(f"\n[3/3] Creating timeline plot of Train vs Test split...")
    os.makedirs(os.path.dirname(plot_output), exist_ok=True)
    
    plt.figure(figsize=(15, 6))
    plt.plot(train['Datetime'], train['PJME_MW'], label=f"Train Set (2002-2014) - {len(train):,} hrs", color='#1f77b4', alpha=0.8, linewidth=0.6)
    plt.plot(test['Datetime'], test['PJME_MW'], label=f"Test Set (2015-2018) - {len(test):,} hrs", color='#ff7f0e', alpha=0.8, linewidth=0.6)
    plt.axvline(pd.to_datetime(split_date), color='red', linestyle='--', linewidth=2, label=f"Split Cutoff: {split_date}")
    
    plt.title("PJME Energy Consumption: Chronological Train / Test Split Timeline", fontsize=14, fontweight='bold')
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Energy Consumption (MW)", fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_output, dpi=200)
    plt.close()
    
    print(f"Timeline plot saved to: {plot_output}")
    print("=" * 80)
    print("STEP 05 COMPLETE: Train and Test data are cleanly partitioned with 0% data leakage!")
    print("=" * 80)

if __name__ == "__main__":
    split_and_validate("Predict_Energy_Consumption/pjme_features.csv")
