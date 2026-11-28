import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
output_dir = "Predict_Energy_Consumption/eda_plots"
os.makedirs(output_dir, exist_ok=True)

data_dir = "Predict_Energy_Consumption/PJM_Hourly_Energy_Cosumption_Dataset"
csv_files = sorted(glob.glob(os.path.join(data_dir, "*_hourly.csv")))

# Collect normalized hourly and monthly profiles for comparison
hourly_profiles = {}
monthly_profiles = {}

for f in csv_files:
    reg_name = os.path.basename(f).replace("_hourly.csv", "")
    df = pd.read_csv(f)
    col = [c for c in df.columns if c != 'Datetime'][0]
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['hour'] = df['Datetime'].dt.hour
    df['month'] = df['Datetime'].dt.month
    
    # Normalized (0 to 1 scale) so all regions can be compared on same graph
    h_mean = df.groupby('hour')[col].mean()
    hourly_profiles[reg_name] = (h_mean - h_mean.min()) / (h_mean.max() - h_mean.min())
    
    m_mean = df.groupby('month')[col].mean()
    monthly_profiles[reg_name] = (m_mean - m_mean.min()) / (m_mean.max() - m_mean.min())

# Plot 1: Hourly Patterns Across All Regions
plt.figure(figsize=(12, 6))
for reg, series in hourly_profiles.items():
    if reg == 'PJME':
        plt.plot(series.index, series.values, label=f"{reg} (Primary)", color='red', linewidth=3)
    else:
        plt.plot(series.index, series.values, label=reg, alpha=0.45)
plt.title("Normalized Hourly Energy Profile Across All 12 PJM Regions", fontsize=14, fontweight='bold')
plt.xlabel("Hour of Day (0-23)", fontsize=12)
plt.ylabel("Normalized Demand (0=Min, 1=Peak)", fontsize=12)
plt.xticks(range(0, 24))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_regions_hourly_profile.png"), dpi=200)
plt.close()

# Plot 2: Monthly Seasonal Patterns Across All Regions
plt.figure(figsize=(12, 6))
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for reg, series in monthly_profiles.items():
    if reg == 'PJME':
        plt.plot(series.index, series.values, label=f"{reg} (Primary)", color='red', linewidth=3)
    else:
        plt.plot(series.index, series.values, label=reg, alpha=0.45)
plt.title("Normalized Monthly Seasonal Demand Profile Across All 12 PJM Regions", fontsize=14, fontweight='bold')
plt.xlabel("Month", fontsize=12)
plt.ylabel("Normalized Demand (0=Min, 1=Peak)", fontsize=12)
plt.xticks(range(1, 13), month_labels)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_regions_monthly_seasonality.png"), dpi=200)
plt.close()

print(f"Visualizations saved successfully in '{output_dir}/'")
