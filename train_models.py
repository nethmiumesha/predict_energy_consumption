import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

def train_and_evaluate(features_path, split_date="2015-01-01", output_dir="Predict_Energy_Consumption/eda_plots"):
    print("=" * 85)
    print("STEP 06: MACHINE LEARNING MODEL TRAINING & BENCHMARKING")
    print("=" * 85)
    
    # 1. Load data & Split
    print(f"\n[1/5] Loading data and preparing Train / Test sets...")
    df = pd.read_csv(features_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    train = df[df['Datetime'] < split_date].copy()
    test = df[df['Datetime'] >= split_date].copy()
    
    feature_cols = [c for c in df.columns if c not in ['Datetime', 'PJME_MW']]
    target_col = 'PJME_MW'
    
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]
    
    print(f"       Features used ({len(feature_cols)}): {feature_cols}")
    print(f"       Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 2. Define Models to Benchmark
    models = {
        "Ridge Regression (Baseline)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=16, max_samples=0.8, n_jobs=-1, random_state=42
        ),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42
        )
    }
    
    results = []
    predictions = {}
    
    # 3. Train & Evaluate each model
    print(f"\n[2/5] Training and evaluating {len(models)} Regression models on Test Set (2015-2018)...")
    for name, model in models.items():
        print(f"       Training: {name}...")
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        predictions[name] = preds
        
        mae, rmse, mape, r2 = evaluate_predictions(y_test, preds)
        results.append({
            "Model": name,
            "MAE (MW)": round(mae, 2),
            "RMSE (MW)": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R2 Score": round(r2, 4)
        })
    
    results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
    
    print("\n" + "=" * 85)
    print("MODEL BENCHMARK RESULTS (TEST SET: 2015 - 2018)")
    print("=" * 85)
    print(results_df.to_string(index=False))
    print("=" * 85)
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    best_preds = predictions[best_model_name]
    print(f"\n🏆 Best Performing Model: {best_model_name}")
    print(f"   Test R² Score: {results_df.iloc[0]['R2 Score']:.4f} | Test MAPE: {results_df.iloc[0]['MAPE (%)']:.2f}% | Test MAE: {results_df.iloc[0]['MAE (MW)']:.2f} MW")
    
    # 4. Save Best Model
    model_save_path = "Predict_Energy_Consumption/best_energy_model.joblib"
    print(f"\n[3/5] Saving best model to: {model_save_path}")
    joblib.dump(best_model, model_save_path)
    
    # 5. Visualizations
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[4/5] Generating Actual vs Predicted visualization plot...")
    
    # Plot a 2-week zoom-in on the Test Set (e.g. July 2017 Summer Peak week)
    test_vis = test.copy()
    test_vis['Best_Prediction'] = best_preds
    
    zoom_start = "2017-07-15"
    zoom_end = "2017-07-29"
    zoom_df = test_vis[(test_vis['Datetime'] >= zoom_start) & (test_vis['Datetime'] <= zoom_end)]
    
    plt.figure(figsize=(15, 6))
    plt.plot(zoom_df['Datetime'], zoom_df['PJME_MW'], label='Actual Energy Consumption (MW)', color='black', linewidth=1.8)
    plt.plot(zoom_df['Datetime'], zoom_df['Best_Prediction'], label=f'Predicted ({best_model_name})', color='#e74c3c', linestyle='--', linewidth=1.8)
    
    plt.title(f"Energy Consumption Forecast vs Actual (Zoom: {zoom_start} to {zoom_end})", fontsize=14, fontweight='bold')
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Electricity Demand (MW)", fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_actual_vs_pred = os.path.join(output_dir, "actual_vs_predicted_zoom.png")
    plt.savefig(plot_actual_vs_pred, dpi=200)
    plt.close()
    print(f"       Saved: {plot_actual_vs_pred}")
    
    # Feature Importance (Random Forest)
    print(f"\n[5/5] Extracting & Plotting Feature Importances...")
    rf_model = models["Random Forest Regressor"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 8))
    importances.plot(kind='barh', color='#2980b9')
    plt.title("Feature Importance Ranking (Random Forest Regressor)", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.tight_layout()
    plot_feat_imp = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_feat_imp, dpi=200)
    plt.close()
    print(f"       Saved: {plot_feat_imp}")
    
    print("\n" + "=" * 85)
    print("STEP 06 COMPLETE: All models trained, evaluated, and saved successfully!")
    print("=" * 85)

if __name__ == "__main__":
    train_and_evaluate("Predict_Energy_Consumption/pjme_features.csv")
