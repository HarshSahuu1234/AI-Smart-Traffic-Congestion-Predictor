import pandas as pd
import numpy as np
import json
import os

def load_data():
    """
    Step 1: Load all relevant datasets
    """
    # Find the base directory of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'datasets')
    
    # Load Route Master List
    route_df = pd.read_csv(os.path.join(data_dir, 'route_data.csv'))
    
    # Load Cleaned Traffic Data (which already contains Toll, Weather, and Accidents merged!)
    traffic_df = pd.read_csv(os.path.join(data_dir, 'cleaned_traffic_data.csv'))
    
    return route_df, traffic_df

def calculate_route_metrics(route_df, traffic_df):
    """
    Step 2 & 3: Compare routes and calculate specific metrics like ETA and Congestion.
    We will simulate a "Real-Time Request" by fetching the absolute latest timestamp in our dataset.
    """
    # Get the latest timestamp to simulate "current real-time conditions"
    latest_time = traffic_df['timestamp'].max()
    current_data = traffic_df[traffic_df['timestamp'] == latest_time]
    
    route_options = []
    
    # Loop through each available route to calculate current metrics
    for _, route in route_df.iterrows():
        r_id = route['route_id']
        
        # Find the real-time data for this specific route
        current_route_data = current_data[current_data['route_id'] == r_id]
        if current_route_data.empty:
            continue
            
        data_row = current_route_data.iloc[0]
        
        # 1. Congestion Score (0=Low, 1=Medium, 2=High)
        congestion = data_row['congestion_level_encoded']
        
        # 2. Dynamic ETA Calculation
        # Base ETA increases based on real-time congestion
        base_eta = route['base_eta_mins']
        if congestion == 2:
            eta_multiplier = 1.5  # High congestion = 50% more time
        elif congestion == 1:
            eta_multiplier = 1.2  # Medium congestion = 20% more time
        else:
            eta_multiplier = 1.0  # Low congestion = normal time
            
        realtime_eta = int(base_eta * eta_multiplier)
        
        # 3. Toll Fees (already includes surge pricing logic from preprocessing)
        toll_fee = data_row['toll_fee_inr']
        
        # 4. Safety Risks (Weather & Accidents)
        weather_impact = data_row['weather_severity']
        accident_impact = data_row['accident_impact']
        
        # Append all structured data for this route
        route_options.append({
            'route_id': str(r_id),
            'route_name': str(route['route_name']),
            'distance_km': float(route['distance_km']),
            'base_eta_mins': int(base_eta),
            'realtime_eta_mins': int(realtime_eta),
            'toll_fee_inr': float(toll_fee),
            'congestion_level': 'High' if congestion == 2 else ('Medium' if congestion == 1 else 'Low'),
            'weather_severity': int(weather_impact),
            'accident_impact': int(accident_impact)
        })
        
    return route_options

def calculate_ai_score(routes):
    """
    Step 4: Create weighted route scoring logic
    
    Our AI Scoring Formula: 
    Score = (0.4 * congestion) + (0.3 * ETA) + (0.2 * toll) + (0.1 * accident/weather risk)
    
    IMPORTANT: A LOWER score is better (It means less time, less money, and less danger)
    """
    congestion_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    for r in routes:
        congestion_val = congestion_map[r['congestion_level']]
        risk_val = r['weather_severity'] + r['accident_impact']
        
        score = (0.4 * congestion_val) + \
                (0.3 * r['realtime_eta_mins']) + \
                (0.2 * r['toll_fee_inr']) + \
                (0.1 * risk_val)
        
        r['ai_score'] = round(score, 2)
        
    return routes

def generate_recommendations(scored_routes):
    """
    Step 5: Recommend the absolute best routes based on different user needs
    """
    if not scored_routes:
        return {}

    # Fastest Route (Lowest Real-Time ETA)
    fastest = min(scored_routes, key=lambda x: x['realtime_eta_mins'])
    
    # Cheapest Route (Lowest Toll Fee)
    cheapest = min(scored_routes, key=lambda x: x['toll_fee_inr'])
    
    # AI Recommended (Lowest AI Score - Best overall balance)
    best_ai = min(scored_routes, key=lambda x: x['ai_score'])
    
    return {
        "Fastest_Route": fastest,
        "Cheapest_Route": cheapest,
        "AI_Recommended_Route": best_ai
    }

def generate_dynamic_reason(route):
    reason_parts = []
    
    # Congestion logic
    if route['congestion_level'] == 'Low':
        reason_parts.append("Lower congestion")
    elif route['congestion_level'] == 'Medium':
        reason_parts.append("Moderate congestion")
        
    # Toll logic
    if route['toll_fee_inr'] == 0:
        reason_parts.append("no toll")
    elif route['toll_fee_inr'] <= 80:
        reason_parts.append("low toll")
    elif route['toll_fee_inr'] <= 120:
        reason_parts.append("moderate toll")
        
    # Accident logic
    if route['accident_impact'] == 0:
        reason_parts.append("zero accident risk")
    elif route['accident_impact'] < 3:
        reason_parts.append("lower accident risk")
        
    return " + ".join(reason_parts).capitalize()

def main():
    print("Initializing Smart Route Recommendation Engine...\n")
    
    # 1. Load Data
    route_df, traffic_df = load_data()
    
    # 2. Get Real-Time Metrics for all routes
    route_options = calculate_route_metrics(route_df, traffic_df)
    
    # 3. Calculate AI Balance Scores
    scored_routes = calculate_ai_score(route_options)
    
    # 4. Generate Recommendations
    recommendations = generate_recommendations(scored_routes)
    
    # Generate dynamic reason
    ai_route = recommendations["AI_Recommended_Route"]
    reason_text = generate_dynamic_reason(ai_route)
    
    # Print formatted output
    print("--- REAL-TIME ROUTE RECOMMENDATIONS ---\n")
    
    print(f"Fastest Route: {recommendations['Fastest_Route']['route_name']}")
    print(f"ETA: {recommendations['Fastest_Route']['realtime_eta_mins']} mins\n")
    
    print(f"Cheapest Route: {recommendations['Cheapest_Route']['route_name']}")
    print(f"Toll: Rs. {int(recommendations['Cheapest_Route']['toll_fee_inr'])}\n")
    
    print(f"AI Recommended Route: {ai_route['route_name']}")
    print("Reason:")
    print(reason_text)
    
    print("\n---------------------------------------")

if __name__ == "__main__":
    main()
