import pandas as pd
import os
import json

def load_toll_data():
    """
    Step 1: Load toll and route datasets
    """
    # Locate the datasets directory automatically
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'datasets')
    
    route_df = pd.read_csv(os.path.join(data_dir, 'route_data.csv'))
    toll_df = pd.read_csv(os.path.join(data_dir, 'toll_data.csv'))
    
    return route_df, toll_df

def estimate_route_tolls(route_df, toll_df, vehicle_type='Car/Jeep', is_peak_hour=False):
    """
    Step 2 & 3: Estimate toll cost for each route, supporting multiple toll booths.
    """
    # Filter toll data for the specific vehicle type to ensure accurate estimation
    vehicle_tolls = toll_df[toll_df['vehicle_type'] == vehicle_type]
    
    route_toll_estimates = []
    
    for _, route in route_df.iterrows():
        r_id = route['route_id']
        r_name = route['route_name']
        dist_km = route['distance_km']
        
        # Get all historical toll records for this route
        route_plazas = vehicle_tolls[vehicle_tolls['route_id'] == r_id]
        
        plaza_fees = {}
        # Support multiple toll booths by finding all unique plazas on this route
        for plaza in route_plazas['toll_plaza_name'].unique():
            plaza_data = route_plazas[route_plazas['toll_plaza_name'] == plaza]
            
            # Calculation Logic:
            # If peak hour, we calculate the average fee when surge pricing is active.
            # Otherwise, we calculate the normal base fee.
            if is_peak_hour:
                surge_data = plaza_data[plaza_data['surge_pricing_active'] == True]
                fee = surge_data['toll_fee_inr'].mean() if not surge_data.empty else plaza_data['toll_fee_inr'].max()
            else:
                normal_data = plaza_data[plaza_data['surge_pricing_active'] == False]
                fee = normal_data['toll_fee_inr'].mean() if not normal_data.empty else plaza_data['toll_fee_inr'].min()
            
            # Fill missing data gracefully if no exact match is found
            if pd.isna(fee):
                fee = plaza_data['toll_fee_inr'].mean()
                
            plaza_fees[plaza] = round(fee, 2)
            
        # Step 4: Calculate total toll and toll per kilometer
        total_toll = round(sum(plaza_fees.values()), 2)
        
        # Avoid division by zero
        if dist_km > 0:
            cost_per_km = round(total_toll / dist_km, 2)
        else:
            cost_per_km = 0.0
            
        # Step 5: Dashboard-friendly structured output structure
        route_toll_estimates.append({
            'route_id': r_id,
            'route_name': r_name,
            'distance_km': dist_km,
            'total_toll_inr': total_toll,
            'cost_per_km_inr': cost_per_km,
            'toll_plazas': plaza_fees
        })
        
    return route_toll_estimates

def analyze_toll_estimates(estimates):
    """
    Step 4: Find the cheapest route
    """
    if not estimates:
        return {}
        
    # Sort by total toll cost to find the absolute cheapest
    sorted_estimates = sorted(estimates, key=lambda x: x['total_toll_inr'])
    
    cheapest = sorted_estimates[0]
    most_expensive = sorted_estimates[-1]
    
    return {
        "Cheapest_Route": cheapest,
        "Most_Expensive_Route": most_expensive,
        "All_Estimates": sorted_estimates
    }

def main():
    print("Initializing Smart Toll Estimation Module...\n")
    
    route_df, toll_df = load_toll_data()
    
    # We estimate based on a standard Car/Jeep during an Off-Peak hour
    estimates = estimate_route_tolls(route_df, toll_df, vehicle_type='Car/Jeep', is_peak_hour=False)
    
    results = analyze_toll_estimates(estimates)
    
    # Step 6: Generate readable comparisons between routes
    print("--- ROUTE TOLL COMPARISONS (Vehicle: Car/Jeep, Off-Peak) ---\n")
    
    for route in results['All_Estimates']:
        print(f"Route: {route['route_name']} ({route['distance_km']} km)")
        print(f"Total Toll: Rs. {route['total_toll_inr']}")
        print(f"Cost per KM: Rs. {route['cost_per_km_inr']}/km")
        print("Toll Plazas:")
        for plaza, fee in route['toll_plazas'].items():
            print(f"  - {plaza}: Rs. {fee}")
        print("-" * 40)
        
    print("\n--- TOLL ESTIMATION SUMMARY ---")
    print(f"Cheapest Route: {results['Cheapest_Route']['route_name']} (Rs. {results['Cheapest_Route']['total_toll_inr']})")
    print(f"Most Expensive Route: {results['Most_Expensive_Route']['route_name']} (Rs. {results['Most_Expensive_Route']['total_toll_inr']})\n")
    
    print("--- TOLL CALCULATION LOGIC EXPLANATION ---")
    print("1. Filtering: We filter historical toll data for the specific vehicle type (e.g., 'Car/Jeep').")
    print("2. Grouping: We map multiple toll plazas to their respective parent routes.")
    print("3. Pricing Context: If it's Peak Hour, we average the historical 'Surge' prices. If Off-Peak, we use normal base prices.")
    print("4. Value Metric: We divide the total toll fee by the route's distance to determine 'Cost Per KM', showing the true value-for-money.")

if __name__ == "__main__":
    main()
