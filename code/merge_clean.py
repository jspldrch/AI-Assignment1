import pandas as pd
import numpy as np
import os
import glob

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
SAMPLE_RATE   = 50 
TRIM_SECONDS  = 15 
TRIM_ROWS     = SAMPLE_RATE * TRIM_SECONDS 

RAW_FOLDER        = "data/raw"
PROCESSED_FOLDER  = "data/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ACTIVITIES = ["walking", "running", "sitting", "stairs", "standing"]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def find_file(activity, sensor_name):
    # Absolute path for debugging
    base_path = os.path.join(os.getcwd(), RAW_FOLDER)
    print(f"  [DEBUG] Searching in: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"  [DEBUG] Directory DOES NOT EXIST: {base_path}")
        return None
    
    all_files = os.listdir(base_path)
    for f in all_files:
        if activity.lower() in f.lower() and sensor_name.lower() in f.lower() and f.endswith('.csv'):
            full_path = os.path.join(base_path, f)
            print(f"  [DEBUG] Found file: {f}")
            return full_path
            
    return None

def load_csv(filepath):
    if filepath is None or not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, sep=',')
    if len(df.columns) == 1:
        df = pd.read_csv(filepath, sep='\t')
    print(f"  Loaded: {os.path.basename(filepath)} ({len(df)} rows)")
    return df

def standardize_columns(df, sensor_type):
    df.columns = df.columns.str.strip()
    time_col = [c for c in df.columns if 'time' in c.lower()]
    if time_col:
        df = df.rename(columns={time_col[0]: 'time'})
    
    prefix = 'acc' if 'acc' in sensor_type.lower() else 'gyro'
    for col in df.columns:
        if col == 'time': continue
        if 'x' in col.lower(): df = df.rename(columns={col: f'{prefix}_x'})
        elif 'y' in col.lower(): df = df.rename(columns={col: f'{prefix}_y'})
        elif 'z' in col.lower(): df = df.rename(columns={col: f'{prefix}_z'})
    return df

def trim_recording(df):
    if len(df) <= 2 * TRIM_ROWS:
        print("  [WARNING] File too short to trim!")
        return df
    df = df.iloc[TRIM_ROWS:-TRIM_ROWS].reset_index(drop=True)
    return df

def merge_acc_gyro(acc_df, gyro_df):
    acc_df  = acc_df.sort_values('time').reset_index(drop=True)
    gyro_df = gyro_df.sort_values('time').reset_index(drop=True)
    merged = pd.merge_asof(acc_df, gyro_df, on='time', direction='nearest', tolerance=0.05)
    merged = merged.dropna().reset_index(drop=True)
    print(f"  Merged successfully: {len(merged)} rows")
    return merged

# ─────────────────────────────────────────────
# MAIN PROCESS
# ─────────────────────────────────────────────

def process_activity(activity):
    print(f"\n{'='*50}\n  {activity.upper()}\n{'='*50}")

    # Search for files using flexible patterns
    acc_path  = find_file(activity, "Linear")
    gyro_path = find_file(activity, "Gyroscope")

    if not acc_path or not gyro_path:
        print(f"  [SKIPPING] Missing files for {activity}")
        return None

    acc_df  = load_csv(acc_path)
    gyro_df = load_csv(gyro_path)

    acc_df  = standardize_columns(acc_df, 'acc')
    gyro_df = standardize_columns(gyro_df, 'gyro')

    acc_df  = trim_recording(acc_df)
    gyro_df = trim_recording(gyro_df)

    merged = merge_acc_gyro(acc_df, gyro_df)
    merged['activity'] = activity

    out_path = os.path.join(PROCESSED_FOLDER, f"{activity}_merged.csv")
    merged.to_csv(out_path, index=False)
    return merged

if __name__ == "__main__":
    all_data = []
    for activity in ACTIVITIES:
        result = process_activity(activity)
        if result is not None:
            all_data.append(result)

    if len(all_data) > 0:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(os.path.join(PROCESSED_FOLDER, "all_activities.csv"), index=False)
        print(f"\nDONE! Combined file saved in {PROCESSED_FOLDER}")