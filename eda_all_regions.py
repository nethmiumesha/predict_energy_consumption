import os
import glob
import pandas as pd
import numpy as np

def audit_dataset(data_dir, output_csv="eda_summary_report.csv"):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*_hourly.csv")))
    
    summary_records = []
    
    print(f"{'='*100}")
    print(f"PJM HOURLY ENERGY CONSUMPTION - COMPREHENSIVE DATASET AUDIT & EDA")
    print(f"{'='*100}")
    print(f"Found {len(csv_files)} regional hourly CSV files.\n")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        
        # Identify energy column (the non-Datetime column)
        energy_col = [c for c in df.columns if c != 'Datetime'][0]
        
        # Parse datetimes
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        total_rows = len(df)
        min_date = df['Datetime'].min()
        max_date = df['Datetime'].max()
        
        # 1. Duplicates check
        num_duplicates = df.duplicated(subset=['Datetime']).sum()
        
        # Deduplicate by averaging duplicate hours for continuity check
        df_unique = df.groupby('Datetime', as_index=False)[energy_col].mean()
        
        # 2. Missing timestamps check
        ideal_range = pd.date_range(start=min_date, end=max_date, freq='h')
        missing_count = len(ideal_range.difference(df_unique['Datetime']))
        completeness_pct = (len(df_unique) / len(ideal_range)) * 100
        
        # 3. Consumption stats
        min_mw = df[energy_col].min()
        max_mw = df[energy_col].max()
        mean_mw = df[energy_col].mean()
        median_mw = df[energy_col].median()
        
        # 4. Temporal analysis: Peak Hours & Peak Months
        df['hour'] = df['Datetime'].dt.hour
        df['month'] = df['Datetime'].dt.month
        
        hourly_avg = df.groupby('hour')[energy_col].mean()
        top_hours = hourly_avg.nlargest(3).index.tolist()
        lowest_hours = hourly_avg.nsmallest(3).index.tolist()
        
        monthly_avg = df.groupby('month')[energy_col].mean()
        top_months = monthly_avg.nlargest(3).index.tolist()
        lowest_months = monthly_avg.nsmallest(3).index.tolist()
        
        month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                       7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        
        top_months_named = [month_names[m] for m in top_months]
        lowest_months_named = [month_names[m] for m in lowest_months]
        
        summary_records.append({
            "File": filename,
            "Region": energy_col.replace('_MW', ''),
            "Total_Records": total_rows,
            "Start_Date": str(min_date.date()),
            "End_Date": str(max_date.date()),
            "Duplicate_Timestamps": num_duplicates,
            "Missing_Hours": missing_count,
            "Completeness_%": round(completeness_pct, 2),
            "Mean_MW": round(mean_mw, 1),
            "Min_MW": round(min_mw, 1),
            "Max_MW": round(max_mw, 1),
            "Peak_Hours": ", ".join(f"{h}:00" for h in top_hours),
            "OffPeak_Hours": ", ".join(f"{h}:00" for h in lowest_hours),
            "Peak_Months": ", ".join(top_months_named),
            "Lowest_Months": ", ".join(lowest_months_named)
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_csv, index=False)
    
    # Print formatted output table
    display_cols = ["Region", "Total_Records", "Start_Date", "End_Date", "Duplicate_Timestamps", "Missing_Hours", "Completeness_%", "Mean_MW", "Peak_Hours", "Peak_Months"]
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nSaved full detailed audit report to: {output_csv}")
    return summary_df

if __name__ == "__main__":
    audit_dataset("Predict_Energy_Consumption/PJM_Hourly_Energy_Cosumption_Dataset", "Predict_Energy_Consumption/eda_summary_report.csv")
