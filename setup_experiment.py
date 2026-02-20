import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools
import json
import os
from typing import List, Tuple, Dict

# ==============================================================================
# 0. GLOBAL CONFIGURATION & CONSTANTS
# ==============================================================================
SEED = 42
np.random.seed(SEED)

# System Dimensions
NUM_WAREHOUSES = 5
NUM_CUSTOMERS = 20
NUM_PARTS = 8
NUM_PERIODS = 120   # 10 Years
S_MAX_LOOKUP = 1000  # Max stock level for pre-computing capacity

# Financial Parameters
FACILITY_COSTS = [20833.33, 16666.67, 14583.33, 18333.33, 23083.33]
CLOSING_COST = 400000.00
PART_PRICES = {
    0: 5000.00, 
    1: 20000.00, 
    2: 2500.00, 
    3: 15600.00,
    4: 3900.00, 
    5: 45000.00, 
    6: 7500.00, 
    7: 12000.00
}

# Logistics Parameters
LOGISTICS_PARAMS = {
    'avg_speed_kph': 75.0, 
    'order_proc_time': 4.0,       # Hours to process an order
    'cost_per_km_out': 3.0,       # Outbound shipping cost ($/km/unit)
    'cost_per_km_in': 3.0         # Inbound shipping cost ($/km/unit)
}

# Demand Generation Parameters
SEASONAL_FACTORS = [1.5, 0.5, 1.2, 0.8] # Winter, Spring, Summer, Fall
DEMAND_BASE_MULTIPLIER = 10.0         # Base volume scaling factor

# ==============================================================================
# 1. CONFIGURATION GENERATOR
# ==============================================================================
def generate_config_json():
    """
    Saves all hardcoded constants into a 'static_params.json' file.
    This allows the runner script to load parameters dynamically.
    """
    print("🔹 Step 1: Generating Static Configuration (JSON)...")
    
    config_data = {
        'NUM_WAREHOUSES': NUM_WAREHOUSES,
        'NUM_CUSTOMERS': NUM_CUSTOMERS,
        'NUM_PARTS': NUM_PARTS,
        'f_monthly': FACILITY_COSTS,
        'closing_cost_per_site': CLOSING_COST,
        'v_k': [PART_PRICES[k] for k in range(NUM_PARTS)], # List of prices sorted by Part ID
        'avg_speed_kph': LOGISTICS_PARAMS['avg_speed_kph'],
        'order_proc_time': LOGISTICS_PARAMS['order_proc_time'],
        'cost_per_km_out': LOGISTICS_PARAMS['cost_per_km_out'],
        'cost_per_km_in': LOGISTICS_PARAMS['cost_per_km_in']
    }
    
    with open('static_params.json', 'w') as f:
        json.dump(config_data, f, indent=4)
        
    print(f"   ✅ Saved 'static_params.json'")

# ==============================================================================
# 2. LAMBDA CAPACITY TABLE GENERATOR
# ==============================================================================
def solve_lambda_newton(s: int, alpha: float, tolerance=1e-6, max_iter=100) -> float:
    """
    Uses Newton-Raphson method to find the maximum Poisson mean (Lambda)
    that a given stock level (s) can support at a specific service level (alpha).
    
    Equation: Poisson_CDF(s-1, lambda) - alpha = 0
    """
    # Boundary conditions
    if s == 0: return 0.0
    if alpha >= 1.0: return 0.0      # Infinite stock needed for 100% service
    if alpha <= 0.0: return 9999.0   # Any capacity works for 0% service

    current_lambda = float(s) # Initial guess
    
    for _ in range(max_iter):
        # f(x) = CDF(s-1, lambda) - alpha
        f_val = stats.poisson.cdf(s - 1, current_lambda) - alpha
        
        # f'(x) = -PMF(s-1, lambda) (Derivative of Poisson CDF w.r.t Lambda)
        f_prime = -stats.poisson.pmf(s - 1, current_lambda)

        if abs(f_prime) < 1e-9: 
            break # Avoid division by zero
        
        # Newton Step: x_new = x - f(x)/f'(x)
        diff = f_val / f_prime
        new_lambda = current_lambda - diff
        
        # Guard against negative values
        if new_lambda < 0: 
            new_lambda = current_lambda / 2.0
            
        # Check convergence
        if abs(new_lambda - current_lambda) < tolerance: 
            return new_lambda
            
        current_lambda = new_lambda
        
    return current_lambda

def generate_lambda_table():
    """
    Pre-computes a lookup table for max capacity.
    Rows: Stock Level (s), Service Level (Alpha) -> Value: Max Lambda
    """
    print("🔹 Step 2: Generating Lambda Capacity Table...")
    data_rows = []
    
    # Loop through all stock levels 0 to 400
    for s in range(0, S_MAX_LOOKUP + 1):
        # Loop through relevant alphas 0.50 to 0.99
        for alpha in np.arange(0.50, 1.00, 0.01):
            alpha_rounded = round(alpha, 2)
            max_capacity = solve_lambda_newton(s, alpha_rounded)
            
            data_rows.append({
                "Stock_Level_s": s,
                "Service_Level_alpha": alpha_rounded,
                "Max_Lambda_Capacity": max_capacity
            })
            
    pd.DataFrame(data_rows).to_csv("lambda_capacity_table.csv", index=False)
    print(f"   ✅ Saved 'lambda_capacity_table.csv'")

# ==============================================================================
# 3. DEMAND GENERATION (BASE)
# ==============================================================================
def generate_base_demand_matrix() -> np.ndarray:
    """
    Generates the baseline demand scenario (Customer x Part x Period).
    Includes logic for:
    1. Price-Demand Correlation: Expensive parts have lower failure rates.
    2. Seasonality: Varying demand multipliers by month.
    """
    print("🔹 Step 3: Generating Base Demand...")
    
    # --- Logic 1: Price Correlation ---
    # Create an array of prices corresponding to part IDs
    prices = np.array([PART_PRICES[k] for k in range(NUM_PARTS)])
    
    # Generate random "base rates" for parts
    random_rates = np.random.uniform(0.05, 0.80, NUM_PARTS)
    
    # Sort prices (Low -> High) and rates (Low -> High)
    sorted_rates = np.sort(random_rates)       
    sorted_price_indices = np.argsort(prices)     
    
    # Map Lowest Price -> Highest Rate (Inverse correlation)
    gamma_k = np.zeros(NUM_PARTS)
    for i in range(NUM_PARTS):
        part_idx = sorted_price_indices[i] # Index of i-th cheapest part
        rate_val = sorted_rates[-(i+1)]    # Take from end of sorted rates (High)
        gamma_k[part_idx] = rate_val

    # --- Logic 2: Seasonality & Generation ---
    # Initialize 3D Matrix [Customers, Parts, Periods]
    demand_matrix = np.zeros((NUM_CUSTOMERS, NUM_PARTS, NUM_PERIODS), dtype=int)

    for t in range(NUM_PERIODS):
        # Determine season (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        month_idx = t % 12
        season_idx = month_idx // 3
        season_factor = SEASONAL_FACTORS[season_idx]
        
        # Calculate Poisson Mean (Lambda) for this period
        # Lambda = Base_Rate * Seasonal_Factor * Global_Multiplier
        period_lambdas = gamma_k * season_factor * DEMAND_BASE_MULTIPLIER
        
        # Generate random integer demand for all customers using Poisson
        demand_matrix[:, :, t] = np.random.poisson(period_lambdas, (NUM_CUSTOMERS, NUM_PARTS))
        
    return demand_matrix

def save_matrix_to_csv(demand_matrix: np.ndarray, filename: str):
    """
    Helper function to flatten the 3D numpy array into a readable CSV format.
    Only saves non-zero demand rows to save space.
    """
    num_customers, num_parts, num_periods = demand_matrix.shape
    rows = []
    
    for j in range(num_customers):
        for k in range(num_parts):
            for t in range(num_periods):
                qty = demand_matrix[j, k, t]
                # if qty > 0:
                rows.append({
                    "Customer_ID": j,
                    "Part_ID": k,
                    "Period": t,
                    "Demand": qty
                })
                    
    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"      📄 Also saved CSV: {filename}")

# ==============================================================================
# 4. DEMAND SCALING (EXACT +/- 5%)
# ==============================================================================
def scale_demand_matrix_per_part(base_matrix: np.ndarray, multiplier: float) -> np.ndarray:
    """
    Scales the demand matrix by a multiplier (e.g., 1.05 or 0.95) precisely.
    
    CRITICAL LOGIC:
    Since demand must be integer, simply multiplying by 1.05 and rounding
    often fails to produce exactly +5% total volume due to rounding errors.
    
    This function:
    1. Calculates the EXACT target total for each Part ID.
    2. Scales and rounds individual cells.
    3. Calculates the 'Gap' (Target - Actual).
    4. Randomly distributes the Gap (add or subtract 1) to active cells
       to ensure the final total matches the mathematical target perfectly.
    """
    scaled_matrix = np.zeros_like(base_matrix)
    num_customers, num_parts, num_periods = base_matrix.shape
    
    # Loop through every Part ID (K) to preserve product mix ratios
    for k in range(num_parts):
        
        # Extract the 2D slice for this part [Customers x Time]
        part_slice_base = base_matrix[:, k, :]
        current_total_volume = part_slice_base.sum()
        
        if current_total_volume == 0:
            continue
        
        # Step 1: Calculate Exact Mathematical Target
        target_total_volume = int(round(current_total_volume * multiplier))
        
        # Step 2: Initial Scale & Round
        part_slice_scaled = np.round(part_slice_base * multiplier).astype(int)
        
        # Step 3: Calculate the Gap (Rounding Error)
        current_scaled_sum = part_slice_scaled.sum()
        gap = target_total_volume - current_scaled_sum
        
        # Step 4: Fix the Gap (Distribute 'gap' units randomly)
        if gap != 0:
            # Find all coordinates (customer, period) that have non-zero demand
            # We only adjust active cells to preserve sparsity patterns.
            active_coords = np.argwhere(part_slice_base > 0)
            
            if len(active_coords) > 0:
                # If gap is larger than available cells, we must allow picking same cell twice
                allow_repeats = len(active_coords) < abs(gap)
                
                # Randomly choose indices to adjust
                chosen_indices = np.random.choice(len(active_coords), size=abs(gap), replace=allow_repeats)
                
                # Determine direction (+1 or -1)
                adjustment_value = 1 if gap > 0 else -1
                
                for idx in chosen_indices:
                    coord_tuple = tuple(active_coords[idx])
                    
                    # Safety check: Don't reduce demand below 0
                    if adjustment_value == -1 and part_slice_scaled[coord_tuple] <= 0:
                        continue
                        
                    part_slice_scaled[coord_tuple] += adjustment_value

        # Insert the corrected slice back into the 3D matrix
        scaled_matrix[:, k, :] = part_slice_scaled
        
    return scaled_matrix

def create_and_save_scenarios(base_matrix: np.ndarray):
    """
    Generates Base, High (+5%), and Low (-5%) scenarios and saves them.
    """
    print("🔹 Step 4: Saving Demand Scenarios (NPY + CSV)...")
    
    # 1. Save Base
    np.save("demand_Base.npy", base_matrix)
    save_matrix_to_csv(base_matrix, "demand_Base.csv")
    print(f"   ✅ Saved Base Scenario (Total Vol: {base_matrix.sum():,.0f})")
    
    # 2. Create & Save High (+5%)
    # Use the precise scaling function
    demand_high = scale_demand_matrix_per_part(base_matrix, 1.05)
    
    np.save("demand_High_5pct.npy", demand_high)
    save_matrix_to_csv(demand_high, "demand_High_5pct.csv")
    print(f"   ✅ Saved High Scenario (Total Vol: {demand_high.sum():,.0f})")
    
    # 3. Create & Save Low (-5%)
    demand_low = scale_demand_matrix_per_part(base_matrix, 0.95)
    
    np.save("demand_Low_5pct.npy", demand_low)
    save_matrix_to_csv(demand_low, "demand_Low_5pct.csv")
    print(f"   ✅ Saved Low Scenario  (Total Vol: {demand_low.sum():,.0f})")

# ==============================================================================
# 5. EXPERIMENT DESIGN MATRIX GENERATOR
# ==============================================================================
def generate_design_matrix_csv():
    """
    Creates the Full Factorial Design of Experiments (DOE).
    Combinations: 3 Demands * 3 Alphas * 3 Holdings * 3 Penalties = 81 Runs.
    """
    print("🔹 Step 5: Generating Experiment Design Matrix...")
    
    # Define Factor Levels
    levels = {
        "Demand": [("Base", 1.0), ("High_5pct", 1.05), ("Low_5pct", 0.95)],
        "Alpha": [0.95, 0.90, 0.85],
        "Holding": [0.25, 0.20, 0.15],
        "Penalty": [0.30, 0.25, 0.20]
    }
    
    # Create Cartesian Product of all levels
    combinations = list(itertools.product(
        levels["Demand"], 
        levels["Alpha"], 
        levels["Holding"], 
        levels["Penalty"]
    ))
    
    # Format rows for CSV
    formatted_data = []
    for run_id, combo in enumerate(combinations):
        demand_info, alpha, holding, penalty = combo
        demand_label = demand_info[0]
        demand_mult = demand_info[1]
        
        formatted_data.append({
            "Run_ID": run_id,
            "Demand_Label": demand_label,
            "Demand_Mult": demand_mult,
            "Alpha": alpha,
            "Holding_Rate": holding,
            "Penalty_Factor": penalty
        })
        
    # Save to CSV
    df = pd.DataFrame(formatted_data)
    df.to_csv("experiment_design_master.csv", index=False)
    print(f"   ✅ Saved 'experiment_design_master.csv' ({len(df)} Runs)")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    generate_config_json()
    generate_lambda_table()
    
    base_demand = generate_base_demand_matrix()
    create_and_save_scenarios(base_demand)
    
    generate_design_matrix_csv()
    
    print("\n🚀 Setup Complete. Ready for execution.")