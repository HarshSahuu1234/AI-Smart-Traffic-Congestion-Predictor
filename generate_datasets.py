import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

out_dir = r'c:\Users\Harsh\.gemini\antigravity\scratch\MARibbonEA\AI-Smart-Traffic-Congestion-Predictor\datasets'
os.makedirs(out_dir, exist_ok=True)

# 1. Route Data
routes = [
    {'route_id': 'R001', 'route_name': 'Delhi-Gurgaon Expressway', 'start_point': 'Dhaula Kuan', 'end_point': 'Rajiv Chowk', 'distance_km': 28.0, 'base_eta_mins': 45},
    {'route_id': 'R002', 'route_name': 'Noida-Greater Noida Expressway', 'start_point': 'Mahamaya Flyover', 'end_point': 'Pari Chowk', 'distance_km': 24.5, 'base_eta_mins': 30},
    {'route_id': 'R003', 'route_name': 'Delhi-Meerut Expressway', 'start_point': 'Nizamuddin Bridge', 'end_point': 'Meerut', 'distance_km': 96.0, 'base_eta_mins': 90},
    {'route_id': 'R004', 'route_name': 'Outer Ring Road (Delhi)', 'start_point': 'Nehru Place', 'end_point': 'Janakpuri', 'distance_km': 22.0, 'base_eta_mins': 50},
    {'route_id': 'R005', 'route_name': 'NH-48 (Delhi-Jaipur)', 'start_point': 'Gurgaon Toll Plaza', 'end_point': 'Manesar', 'distance_km': 20.0, 'base_eta_mins': 25}
]
route_df = pd.DataFrame(routes)
route_df.to_csv(os.path.join(out_dir, 'route_data.csv'), index=False)

# Time Generation for 1 week, hourly intervals
start_time = datetime(2023, 10, 1, 0, 0, 0)
timestamps = [start_time + timedelta(hours=i) for i in range(24 * 7)]  # 168 hours

# 2. Weather Data
weather_conditions = ['Clear', 'Partly Cloudy', 'Overcast', 'Light Rain', 'Heavy Rain', 'Fog']
weather_data = []
for ts in timestamps:
    weather = np.random.choice(weather_conditions, p=[0.5, 0.2, 0.1, 0.05, 0.05, 0.1])
    temp = round(np.random.normal(25, 5), 1)
    if weather == 'Clear': temp += 2
    if weather in ['Light Rain', 'Heavy Rain']: temp -= 3
    weather_data.append({
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'temperature_celsius': temp,
        'precipitation_mm': round(np.random.exponential(5), 1) if 'Rain' in weather else 0.0,
        'visibility_km': round(np.random.uniform(0.5, 2.0), 1) if weather == 'Fog' else round(np.random.uniform(5.0, 10.0), 1),
        'condition': weather
    })
weather_df = pd.DataFrame(weather_data)
weather_df.to_csv(os.path.join(out_dir, 'weather_data.csv'), index=False)

# 3. Traffic Data
traffic_data = []
for ts in timestamps:
    hour = ts.hour
    is_weekend = ts.weekday() >= 5
    for r in routes:
        # Determine peak hour
        is_morning_peak = 8 <= hour <= 11 and not is_weekend
        is_evening_peak = 17 <= hour <= 20 and not is_weekend
        
        # Base logic
        base_vol = np.random.randint(50, 200)
        base_speed = np.random.randint(40, 80)
        
        if is_morning_peak or is_evening_peak:
            volume = int(base_vol * np.random.uniform(2.5, 4.0))
            speed = max(5, int(base_speed * np.random.uniform(0.2, 0.5)))
        else:
            volume = int(base_vol * np.random.uniform(0.8, 1.5))
            speed = max(15, int(base_speed * np.random.uniform(0.8, 1.2)))
            
        # Introduce randomness based on route
        if r['route_id'] == 'R001': # High congestion
            volume = int(volume * 1.2)
            speed = max(5, int(speed * 0.8))
            
        weather_row = weather_df[weather_df['timestamp'] == ts.strftime('%Y-%m-%d %H:%M:%S')].iloc[0]
        if weather_row['condition'] in ['Heavy Rain', 'Fog']:
            speed = max(5, int(speed * 0.6))
            
        traffic_data.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'route_id': r['route_id'],
            'traffic_volume': volume,
            'average_speed_kmph': speed,
            'congestion_level': 'High' if speed < 20 else ('Medium' if speed < 40 else 'Low')
        })
traffic_df = pd.DataFrame(traffic_data)
traffic_df.to_csv(os.path.join(out_dir, 'traffic_data.csv'), index=False)

# 4. Accident Data
accident_data = []
for _ in range(50):
    ts = np.random.choice(timestamps)
    route = np.random.choice(routes)
    severity = np.random.choice(['Minor', 'Major', 'Severe'], p=[0.7, 0.2, 0.1])
    lat_base, lon_base = 28.5, 77.2  # Delhi approx
    
    accident_data.append({
        'accident_id': f'A{np.random.randint(1000, 9999)}',
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'route_id': route['route_id'],
        'latitude': round(lat_base + np.random.uniform(-0.2, 0.2), 4),
        'longitude': round(lon_base + np.random.uniform(-0.2, 0.2), 4),
        'severity': severity,
        'vehicles_involved': np.random.randint(1, 4) if severity == 'Minor' else np.random.randint(2, 6)
    })
accident_df = pd.DataFrame(accident_data)
accident_df.to_csv(os.path.join(out_dir, 'accident_data.csv'), index=False)

# 5. Toll Data
toll_data = []
for ts in timestamps:
    hour = ts.hour
    for r in routes:
        base_toll = int(np.random.choice([50, 80, 100, 150]))
        if (8 <= hour <= 11 or 17 <= hour <= 20) and ts.weekday() < 5:
            fee = int(base_toll * 1.5)
            surge = True
        else:
            fee = base_toll
            surge = False
            
        toll_data.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'route_id': r['route_id'],
            'toll_plaza_name': f"{r['start_point']} Plaza",
            'vehicle_type': np.random.choice(['Car/Jeep', 'LCV', 'Bus/Truck'], p=[0.7, 0.2, 0.1]),
            'toll_fee_inr': fee,
            'surge_pricing_active': surge
        })
toll_df = pd.DataFrame(toll_data)
toll_df = toll_df.sample(n=min(len(toll_df), 400)).sort_values('timestamp')
toll_df.to_csv(os.path.join(out_dir, 'toll_data.csv'), index=False)
print("CSVs generated successfully.")
