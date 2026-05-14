import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(data_dir):
    """
    Step 1: Load all CSV files using Pandas
    """
    print("Loading datasets...")
    traffic_df = pd.read_csv(os.path.join(data_dir, "traffic_data.csv"))
    weather_df = pd.read_csv(os.path.join(data_dir, "weather_data.csv"))
    accident_df = pd.read_csv(os.path.join(data_dir, "accident_data.csv"))
    toll_df = pd.read_csv(os.path.join(data_dir, "toll_data.csv"))
    route_df = pd.read_csv(os.path.join(data_dir, "route_data.csv"))
    return traffic_df, weather_df, accident_df, toll_df, route_df

def merge_datasets(traffic_df, weather_df, accident_df, toll_df, route_df):
    """
    Step 2: Merge datasets properly based on timestamp and route_id
    """
    print("Merging datasets...")
    # Start with the main traffic dataframe
    # Merge traffic with route data on 'route_id'
    df = pd.merge(traffic_df, route_df, on='route_id', how='left')
    
    # Merge with weather data on 'timestamp'
    df = pd.merge(df, weather_df, on='timestamp', how='left')
    
    # Merge with accident data
    # We drop duplicates just in case there are multiple accidents at the exact same hour/route
    accident_summary = accident_df[['timestamp', 'route_id', 'severity', 'vehicles_involved']].drop_duplicates(subset=['timestamp', 'route_id'])
    df = pd.merge(df, accident_summary, on=['timestamp', 'route_id'], how='left')
    
    # Merge with toll data
    toll_summary = toll_df[['timestamp', 'route_id', 'toll_fee_inr', 'surge_pricing_active']].drop_duplicates(subset=['timestamp', 'route_id'])
    df = pd.merge(df, toll_summary, on=['timestamp', 'route_id'], how='left')
    
    return df

def feature_engineering(df):
    """
    Step 6: Create useful ML features
    """
    print("Creating useful ML features...")
    # Convert timestamp to a DateTime object to easily extract hour and day
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek # 0 = Monday, 6 = Sunday
    
    # Create is_peak_hour flag (8-11 AM and 5-8 PM on weekdays)
    df['is_peak_hour'] = ((df['hour'].between(8, 11) | df['hour'].between(17, 20)) & (df['day_of_week'] < 5)).astype(int)
    
    # Map weather conditions to a numeric severity score
    weather_severity_map = {
        'Clear': 0,
        'Partly Cloudy': 1,
        'Overcast': 2,
        'Fog': 3,
        'Light Rain': 3,
        'Heavy Rain': 5
    }
    df['weather_severity'] = df['condition'].map(weather_severity_map).fillna(0)
    
    # Map accident severity to a numerical impact score
    # If severity is NaN (no accident), it gets filled with 0
    accident_impact_map = {
        'Minor': 1,
        'Major': 3,
        'Severe': 5
    }
    df['accident_impact'] = df['severity'].map(accident_impact_map).fillna(0)
    
    return df

def handle_missing_and_encode(df):
    """
    Step 3 & 4: Handle missing values and encode categorical columns
    """
    print("Handling missing values and encoding...")
    
    # Step 3: Handle missing values
    # If there are no accidents, vehicles_involved should be 0
    # For missing toll data, we assume surge pricing is False and toll fee is the median
    df['vehicles_involved'] = df['vehicles_involved'].fillna(0)
    df['surge_pricing_active'] = df['surge_pricing_active'].fillna(False).astype(int)
    
    if df['toll_fee_inr'].isnull().any():
        median_fee = df['toll_fee_inr'].median()
        # If median is NaN because no toll data matched, default to 50
        if pd.isna(median_fee): median_fee = 50.0 
        df['toll_fee_inr'] = df['toll_fee_inr'].fillna(median_fee)
        
    df['condition'] = df['condition'].fillna('Clear')
    
    # Step 4: Encode categorical columns (convert text to numbers for ML)
    categorical_cols = ['route_id', 'route_name', 'start_point', 'end_point', 'condition']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    return df, label_encoders

def process_labels_and_normalize(df):
    """
    Step 5 & 7: Map congestion labels and normalize numeric columns
    """
    print("Processing labels and normalizing...")
    
    # Step 5: Encode congestion labels (Low=0, Medium=1, High=2)
    # This acts as our Target Variable (Y) for the Machine Learning model
    congestion_map = {'Low': 0, 'Medium': 1, 'High': 2}
    if 'congestion_level' in df.columns:
        df['congestion_level_encoded'] = df['congestion_level'].map(congestion_map)
    
    # Step 7: Normalize required numeric columns (scale them to mean=0, std=1)
    numeric_cols = ['traffic_volume', 'average_speed_kmph', 'distance_km', 'base_eta_mins', 
                    'temperature_celsius', 'precipitation_mm', 'visibility_km', 'toll_fee_inr']
    
    scaler = StandardScaler()
    for col in numeric_cols:
        if col in df.columns:
            df[col + '_scaled'] = scaler.fit_transform(df[[col]])
            
    return df, scaler

def main():
    # Automatically locate the datasets folder relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'datasets')
    
    # Execute the pipeline step by step
    traffic_df, weather_df, accident_df, toll_df, route_df = load_data(data_dir)
    
    df_merged = merge_datasets(traffic_df, weather_df, accident_df, toll_df, route_df)
    print(f"\nData Shape after merging: {df_merged.shape}")
    
    df_features = feature_engineering(df_merged)
    df_clean, encoders = handle_missing_and_encode(df_features)
    df_final, scaler = process_labels_and_normalize(df_clean)
    
    # Step 8: Save the final cleaned dataset
    output_path = os.path.join(data_dir, "cleaned_traffic_data.csv")
    df_final.to_csv(output_path, index=False)
    print(f"\nPipeline completed! Cleaned dataset saved to: {output_path}")
    
    # Print sample output for verification
    print("\n--- SAMPLE OF FINAL ML-READY FEATURES ---")
    ml_cols = ['timestamp', 'route_id_encoded', 'hour', 'is_peak_hour', 
               'weather_severity', 'accident_impact', 'traffic_volume_scaled', 
               'congestion_level_encoded']
    print(df_final[ml_cols].head())

if __name__ == "__main__":
    main()
