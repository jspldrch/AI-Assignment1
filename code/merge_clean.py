"""
Loads raw accelerometer and gyroscope CSV files,
trims the first/last 15 seconds, merges both sensors
into one file per activity, and saves to data/processed/.
"""
 
import pandas as pd
import numpy as np
import os
 
# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
 
SAMPLE_RATE   = 50                        # Hz
TRIM_SECONDS  = 15                        # seconds to cut from start and end
TRIM_ROWS     = SAMPLE_RATE * TRIM_SECONDS  # = 750 rows
 
RAW_FOLDER        = "data/raw"
PROCESSED_FOLDER  = "data/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
 
# Add more activities here as you collect the data
ACTIVITIES = [
    "walking",
    # "running",
    # "sitting",
    # "stairs",
    # "standing",
]
 
 
# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
 
def load_csv(filepath):
    """Load a CSV file — tries comma separator, then tab."""
    if not os.path.exists(filepath):
        print(f"  [WARNING] File not found: {filepath}")
        return None
    df = pd.read_csv(filepath, sep=',')
    if len(df.columns) == 1:          # probably tab-separated
        df = pd.read_csv(filepath, sep='\t')
    print(f"  Loaded: {filepath}  ({len(df)} rows)")
    return df
 
 
def standardize_columns(df, sensor_type):
    """
    Rename columns to a consistent format.
    Accelerometer -> time, acc_x, acc_y, acc_z
    Gyroscope     -> time, gyro_x, gyro_y, gyro_z
    This is necessary because both files have columns
    called 'X', 'Y', 'Z' — after merging we need to
    tell them apart.
    """
    df.columns = df.columns.str.strip()
 
    # Rename time column
    time_col = [c for c in df.columns if 'time' in c.lower()]
    if time_col:
        df = df.rename(columns={time_col[0]: 'time'})
 
    # Rename X, Y, Z based on sensor type
    prefix = 'acc' if sensor_type == 'acc' else 'gyro'
    for col in df.columns:
        if col == 'time':
            continue
        if 'x' in col.lower():
            df = df.rename(columns={col: f'{prefix}_x'})
        elif 'y' in col.lower():
            df = df.rename(columns={col: f'{prefix}_y'})
        elif 'z' in col.lower():
            df = df.rename(columns={col: f'{prefix}_z'})
 
    return df
 
 
def trim_recording(df):
    """Remove the first and last 15 seconds (750 rows at 50 Hz)."""
    df = df.iloc[TRIM_ROWS:-TRIM_ROWS].reset_index(drop=True)
    print(f"  After trimming: {len(df)} rows")
    return df
 
 
def merge_acc_gyro(acc_df, gyro_df):
    """
    Merge accelerometer and gyroscope by nearest timestamp.
    The two sensors don't record at exactly the same moments,
    so for each acc timestamp we find the closest gyro timestamp.
    """
    acc_df  = acc_df.sort_values('time').reset_index(drop=True)
    gyro_df = gyro_df.sort_values('time').reset_index(drop=True)
 
    merged = pd.merge_asof(
        acc_df,
        gyro_df,
        on='time',
        direction='nearest',
        tolerance=0.05      # max 50ms gap allowed
    )
 
    merged = merged.dropna().reset_index(drop=True)
    print(f"  After merging:  {len(merged)} rows")
    return merged
 
 
# ─────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────
 
def process_activity(activity):
    print(f"\n{'='*50}")
    print(f"  {activity.upper()}")
    print(f"{'='*50}")
 
    # Load
    acc_df  = load_csv(os.path.join(RAW_FOLDER, f"{activity}_Accelerometer.csv"))
    gyro_df = load_csv(os.path.join(RAW_FOLDER, f"{activity}_Gyroscope.csv"))
 
    if acc_df is None or gyro_df is None:
        print(f"  Skipping {activity} — file(s) missing.")
        return None
 
    # Standardize column names
    acc_df  = standardize_columns(acc_df,  'acc')
    gyro_df = standardize_columns(gyro_df, 'gyro')
 
    # Trim first and last 15 seconds
    acc_df  = trim_recording(acc_df)
    gyro_df = trim_recording(gyro_df)
 
    # Merge acc + gyro
    merged = merge_acc_gyro(acc_df, gyro_df)
 
    # Add activity label
    merged['activity'] = activity
 
    # Save
    out_path = os.path.join(PROCESSED_FOLDER, f"{activity}_merged.csv")
    merged.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
 
    return merged
 
 
# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    all_data = []
 
    for activity in ACTIVITIES:
        result = process_activity(activity)
        if result is not None:
            all_data.append(result)
 
    # Once you have all activities, this saves one combined file
    if len(all_data) > 1:
        combined = pd.concat(all_data, ignore_index=True)
        out_path = os.path.join(PROCESSED_FOLDER, "all_activities.csv")
        combined.to_csv(out_path, index=False)
        print(f"\nCombined file saved -> {out_path}")
        print(combined['activity'].value_counts())
 
    print("\nDone.")