import pandas as pd
import numpy as np
import os

#settings

WINDOW_SIZE_SEC = 2    
OVERLAP_PCT     = 0.5  
SAMPLING_RATE   = 50   

PROCESSED_FOLDER = "data/processed"
FEATURES_OUTPUT  = os.path.join(PROCESSED_FOLDER, "final_features.csv")

#create features

def extract_features_from_window(window, activity_label):
    """
    Calculates statistical features for a single 2-second window.
    These features represent the 'fingerprint' of the movement.
    """
    features = {}
    #only sensor
    sensor_cols = [c for c in window.columns if c not in ['time', 'activity']]
    
    for col in sensor_cols:
        
        features[f'{col}_mean'] = window[col].mean()
        features[f'{col}_std']  = window[col].std()
        features[f'{col}_max']  = window[col].max()
        features[f'{col}_min']  = window[col].min()
        
        #Signal Intensity (RMS)
        features[f'{col}_rms']  = np.sqrt(np.mean(window[col]**2))
        
        features[f'{col}_median'] = window[col].median()

    features['label'] = activity_label
    return features

def process_all_activities():
    
    all_feature_rows = []
    
    files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith('_merged.csv')]
    

    print(f"Found {len(files)} activity files. Starting windowing...")

    window_step = int(SAMPLING_RATE * WINDOW_SIZE_SEC)
    step_size = int(window_step * (1 - OVERLAP_PCT))

    for file_name in files:
        file_path = os.path.join(PROCESSED_FOLDER, file_name)
        df = pd.read_csv(file_path)
        activity = df['activity'].iloc[0]
        
        print(f"  Processing {activity}...")
        
        count = 0
       
        for start in range(0, len(df) - window_step, step_size):
            window = df.iloc[start : start + window_step]
            
           
            feature_row = extract_features_from_window(window, activity)
            all_feature_rows.append(feature_row)
            count += 1
            
        print(f"    -> Generated {count} windows for {activity}")

    #final dataframe
    final_df = pd.DataFrame(all_feature_rows)
    final_df.to_csv(FEATURES_OUTPUT, index=False)
    
    print("\n" + "="*30)
    print(f"SUCCESS: {FEATURES_OUTPUT} created.")
    print(f"Total dataset size: {final_df.shape}")
    print("Class distribution:")
    print(final_df['label'].value_counts())
    print("="*30)

if __name__ == "__main__":
    process_all_activities()
