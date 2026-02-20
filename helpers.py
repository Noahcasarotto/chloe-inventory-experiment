import sys
import pandas as pd
import numpy as np
import os

# ==============================================================================
# 1. LOGGING CLASS
# ==============================================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==============================================================================
# 2. DATA INSPECTION REPORT (NEW)
# ==============================================================================
def generate_data_inspection_report(params, demand_data, initial_stock, run_id, design_row):
    """
    Prints a comprehensive data audit for the current run context.
    matches the user's 'Review Snippet' format.
    """
    print("\n" + "="*60)
    print(f"   DATA REVIEW & VALIDATION REPORT: RUN {run_id}")
    print("="*60)

    # Extract Globals from params
    I = params['NUM_WAREHOUSES']
    J = params['NUM_CUSTOMERS']
    K = params['NUM_PARTS']
    T = demand_data.shape[2]
    s_max_ik = params['s_max_ik']
    
    # Extract Financials from Design Row
    hold_rate = design_row['Holding_Rate']
    penalty_factor = design_row['Penalty_Factor']
    
    # ---------------------------------------------------------
    # 1. SETS & DIMENSIONS
    # ---------------------------------------------------------
    s_max_min = min(s_max_ik.values())
    s_max_max = max(s_max_ik.values())
    print(f"\n[1] SYSTEM DIMENSIONS")
    print(f"  - Warehouses (I): {I}")
    print(f"  - Customers (J):  {J}")
    print(f"  - Parts (K):      {K}")
    print(f"  - Periods (T):    {T} months")
    print(f"  - Stock Levels:   per-(i,k), range {s_max_min} to {s_max_max}")

    # ---------------------------------------------------------
    # 2. GLOBAL FINANCIAL PARAMETERS
    # ---------------------------------------------------------
    print(f"\n[2] FINANCIAL SETTINGS")
    print(f"  - Holding Cost Rate:        {hold_rate * 100}% per year")
    print(f"  - Backorder Discount:       {penalty_factor * 100}% (Penalty on Part Value)")
    # Note: 'warehouse_utlization_percent' is logic, not a param, usually fixed at 10%
    print(f"  - Warehouse Util Threshold: 10%") 

    # ---------------------------------------------------------
    # 3. PART PARAMETERS
    # ---------------------------------------------------------
    print(f"\n[3] PART PROFILE")
    df_parts = pd.DataFrame({
        'Part_ID': range(K),
        'Price ($)': params['v_k'],
        'Monthly_Hold_Cost ($)': params['h_ik'][0, :] # Warehouse 0 as representative
    })
    df_parts['Annual_Hold ($)'] = df_parts['Price ($)'] * hold_rate
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_parts.round(4).to_string(index=False))

    # ---------------------------------------------------------
    # 4. FACILITY PARAMETERS
    # ---------------------------------------------------------
    print(f"\n[4] FACILITY PROFILE")
    closing_penalty = params.get('closing_cost_per_site', 400000.0)
    df_facilities = pd.DataFrame({
        'Whs_ID': range(I),
        'Monthly_Fixed_Cost ($)': params['f_monthly'],
        'Closing_Penalty ($)': [closing_penalty] * I
    })
    print(df_facilities.round(2).to_string(index=False))

    # Initial Stock
    initial_stock_data = {f'Warehouse {i}': {} for i in range(I)}
    for (wh_id, part_id), stock_val in initial_stock.items():
        initial_stock_data[f'Warehouse {wh_id}'][f'Part {part_id}'] = stock_val

    df_initial_stock = pd.DataFrame.from_dict(initial_stock_data, orient='index').transpose()
    print(f"  - Initial Stock (Part vs Warehouse):")
    print(df_initial_stock.to_string())

    # ---------------------------------------------------------
    # 5. TOPOLOGY & LOGISTICS
    # ---------------------------------------------------------
    print(f"\n[5] TOPOLOGY & LOGISTICS")
    
    # We recover distance roughly from cost if raw matrix isn't stored, 
    # but run_experiment will now store 'dist_matrix' in params for us.
    if 'dist_matrix' in params:
        d_mat = params['dist_matrix']
        print(f"  - Outbound (Facility -> Customer):")
        print(f"      Dist Min: {np.min(d_mat):.2f} km")
        print(f"      Dist Max: {np.max(d_mat):.2f} km")
    
    print(f"      Cost Min: ${np.min(params['c_ijk']):.2f} / unit")
    print(f"      Cost Max: ${np.max(params['c_ijk']):.2f} / unit")

    print(f"  - Inbound (Main DC -> Facility):")
    print(f"      Cost Min: ${np.min(params['c_inbound_i']):.2f} / unit")
    print(f"      Cost Max: ${np.max(params['c_inbound_i']):.2f} / unit")

    print(f"  - Time Windows (Hours):")
    print(f"      Min: {np.min(params['w_j']):.2f} h")
    print(f"      Max: {np.max(params['w_j']):.2f} h")

    # ---------------------------------------------------------
    # 6. DEMAND STATISTICS
    # ---------------------------------------------------------
    print(f"\n[6] DEMAND STATISTICS")
    total_demand = np.sum(demand_data)
    zero_periods = np.sum(demand_data == 0)
    total_cells = demand_data.size

    print(f"  - Total Network Demand: {total_demand:,.0f} units")
    print(f"  - Avg Demand per Month: {total_demand / T:,.2f} units")
    print(f"  - Sparsity (Zero Dem):  {100 * zero_periods / total_cells:.2f}%")

    print(f"  - Seasonal Patterns (Avg Monthly Demand):")
    seasons = ["Winter (High)", "Spring (Low)", "Summer (Med)", "Fall (Med)"]
    for s in range(4):
        periods_in_season = [t for t in range(T) if (t % 12) // 3 == s]
        if len(periods_in_season) > 0:
            avg_seasonal = np.mean(demand_data[:, :, periods_in_season])
            print(f"      {seasons[s]:<15}: {avg_seasonal:.4f} units/cust/part")

    # ---------------------------------------------------------
    # 7. LOGIC CHECKS
    # ---------------------------------------------------------
    print(f"\n[7] LOGIC CHECKS")
    
    # Reachability
    tau = params['tau']
    w_j = params['w_j']
    infeasible = []
    for j in range(J):
        valid_whs = np.where(tau[:, j] <= w_j[j])[0]
        if len(valid_whs) == 0: infeasible.append(j)

    if not infeasible:
        print(f"  - Reachability: SUCCESS. All {J} customers have valid warehouses.")
    else:
        print(f"  - Reachability: FAILURE! {len(infeasible)} customers unreachable.")
        print(f"    IDs: {infeasible}")

    print("="*60 + "\n")

# ==============================================================================
# 3. EXPLICIT VALIDATION AUDIT (Post-Run)
# ==============================================================================

def run_model_validation_audit(run_result, demand_data, initial_stock, params, run_name="Run 1", closed_warehouses=None):

    """
    Performs post-optimization validation ensuring solution integrity.
    """
    print(f"\n{'='*25} VALIDATION REPORT: {run_name} {'='*25}")

    if not run_result:
        print(">> No solution data found. Validation aborted.")
        return {"Valid": False}

    # Data Extraction
    x_vars = run_result.get('x_vars', {})
    Order = run_result.get('Order_Qty', {})
    End_Inv = run_result.get('End_Inv', {})
    
    I = params['NUM_WAREHOUSES']
    K = params['NUM_PARTS']

    if Order:
        T_max = max((k[2] for k in Order.keys()), default=0)
    else:
        T_max = 0
    T_solved = T_max + 1
    EPSILON = 1e-4

    # --- CHECK 1: GLOBAL FLOW CONSERVATION ---
    print(f"\n[Validation 1] Global Flow Conservation Check")

    if closed_warehouses is None:
        closed_warehouses = []

    closed_whs_inventory = sum(initial_stock[(i, k)]
                                for i in closed_warehouses
                                for k in range(K))

    total_initial = sum(initial_stock.values())
    total_inbound = sum(v.X for v in Order.values())
    
    total_outbound = 0
    for (i, j, k, t), var in x_vars.items():
        if var.X > EPSILON:
            total_outbound += demand_data[j, k, t]

    total_final = sum(End_Inv[i, k, T_solved-1].X for i in range(I) for k in range(K) if (i, k, T_solved-1) in End_Inv)

    # Adjust theoretical end state by subtracting closed warehouse inventory
    theoretical_end = total_initial + total_inbound - total_outbound - closed_whs_inventory
    discrepancy = theoretical_end - total_final

    print(f"  1. Initial System Inventory:  {total_initial:,.0f}")
    print(f"  2. Total System Inflow:     + {total_inbound:,.0f}")
    print(f"  3. Total System Outflow:    - {total_outbound:,.0f}")
    if closed_whs_inventory > 0:
        print(f"  4. Closed Warehouse Inventory (Not in Model): - {closed_whs_inventory:,.0f}")
    print(f"  ---------------------------------------")
    print(f"  = Theoretical Final State:    {theoretical_end:,.0f}")
    print(f"  = Observed Final State:       {total_final:,.0f}")
    
    valid_balance = abs(discrepancy) < 1.0
    if valid_balance:
        print("  >> PASSED: Global mass balance constraints satisfied.")
    else:
        print(f"  >> FAILED: Flow conservation violation. Discrepancy: {discrepancy:.4f}")

    # --- CHECK 2: NODE ACTIVITY ---
    print(f"\n[Validation 2] Node Activity & Flow Consistency Check")
    inconsistency = False
    for i in range(I):
        node_init = sum(initial_stock.get((i, k), 0) for k in range(K))
        node_in = sum(Order[i, k, t].X for k in range(K) for t in range(T_solved) if (i,k,t) in Order)
        node_out = sum(demand_data[j, k, t] for (wi, j, k, t), v in x_vars.items() if wi == i and v.X > EPSILON)

        if node_in < EPSILON:
            if node_out > (node_init + EPSILON):
                print(f"  >> WARNING: Node {i} Flow Violation (Out > Init with 0 In).")
                inconsistency = True
            elif node_out > EPSILON:
                print(f"  >> NOTICE: Node {i} is drawing down inventory (Phase-out).")
            else:
                print(f"  >> INFO: Node {i} is inactive (Closed).")

    if not inconsistency: print("  >> PASSED: All nodes adhere to local flow constraints.")

    # --- CHECK 3: NON-NEGATIVITY ---
    print(f"\n[Validation 3] Non-Negativity Check")
    violations = 0
    for k, v in End_Inv.items():
        if v.X < -EPSILON:
            print(f"  >> FAILED: Negative inventory at {k}: {v.X}")
            violations += 1
            if violations > 3: break
    if violations == 0: print("  >> PASSED: Non-negativity satisfied.")
    
    return {"Valid": valid_balance, "Error": discrepancy}

# ==============================================================================
# 4. SHIPMENT & PLAN SAVING
# ==============================================================================
def save_consolidated_inventory_plan(run_result, demand_data, filename_prefix, run_folder, run_id):
    """Saves granular Inventory Plan (Demand Served, Order, End Inv, Target Stock)."""
    if not run_result: return
    
    x_vars = run_result.get('x_vars', {})
    Order = run_result.get('Order_Qty', {})
    End_Inv = run_result.get('End_Inv', {})
    V_vars = run_result.get('V_vars', {})
    
    # 1. Aggregate Demand Served per Warehouse/Part/Period
    demand_served = {}
    for (i, j, k, t), var in x_vars.items():
        if var.X > 0.5:
            key = (i, k, t)
            demand_served[key] = demand_served.get(key, 0.0) + demand_data[j, k, t]

    # 2. Extract Target Stock Levels (S) from V variables
    target_stock_lookup = {}
    for (i, k, s, t), var in V_vars.items():
        if var.X > 0.5:
            target_stock_lookup[(i, k, t)] = s
            
    # 3. Build DataFrame
    all_keys = set(End_Inv.keys()) | set(Order.keys()) | set(demand_served.keys())
    data = []
    for (i, k, t) in all_keys:
        data.append({
            "Period": t, "Warehouse_ID": i, "Part_ID": k,
            "Demand_Served": demand_served.get((i,k,t), 0.0),
            "Order_Qty": Order[i, k, t].X if (i,k,t) in Order else 0.0,
            "End_Inventory": End_Inv[i, k, t].X if (i,k,t) in End_Inv else 0.0,
            "Target_Stock_Level": target_stock_lookup.get((i, k, t), 0)
        })
        
    df = pd.DataFrame(data).sort_values(['Warehouse_ID', 'Part_ID', 'Period'])
    output_file = f"{run_folder}/Run{run_id}_{filename_prefix}_InventoryPlan.csv"
    df.to_csv(output_file, index=False)
    print(f"    💾 Saved Run{run_id}_{filename_prefix}_InventoryPlan.csv")

def save_shipments_to_csv(run_result, demand_data, filename_prefix, run_folder, run_id):
    """Saves detailed customer shipments."""
    if not run_result: return
    x_vars = run_result.get('x_vars', {})
    data = []
    for (i, j, k, t), var in x_vars.items():
        if var.X > 0.5:
            data.append({
                "Period": t, "Warehouse_ID": i, "Customer_ID": j, "Part_ID": k,
                "Qty": demand_data[j, k, t]
            })
    output_file = f"{run_folder}/Run{run_id}_{filename_prefix}_Shipments.csv"
    pd.DataFrame(data).to_csv(output_file, index=False)
    print(f"    💾 Saved Run{run_id}_{filename_prefix}_Shipments.csv")

# ==============================================================================
# 5. UTILITIES
# ==============================================================================
def check_warehouse_utilization(res, params, demand_data=None):
    """
    Identifies warehouses with <10% volume to close in Scenario B.
    Prints a utilization table and returns (to_close_list, metrics_str).
    """
    if not res: 
        print(">> Cannot check utilization: No results.")
        return [], ""
    
    x_vars = res['x_vars']
    
    # Calculate Total Demand Served (Volume)
    vol = {}
    for i in range(params['NUM_WAREHOUSES']):
        vol[i] = 0.0
    total_network_vol = 0.0
    
    # Sum up volume from active x assignments
    for (i, j, k, t), var in x_vars.items():
        val = var.X
        if val > 0.01:
            qty = demand_data[j, k, t]
            contribution = qty * val
            vol[i] += contribution
            total_network_vol += contribution

    to_close = []
    metrics_parts = []
    
    print("\n[Warehouse Utilization Check]")
    print(f"{'Warehouse':<10} {'Volume':<15} {'Utilization (%)':<18} {'Action'}")
    print("-" * 60)

    for i in range(params['NUM_WAREHOUSES']):
        u_pct = (vol[i] / total_network_vol) * 100 if total_network_vol > 0 else 0
        
        action = "KEEP"
        if u_pct < 10.0:  # Hardcoded 10% threshold as per v5_1_3
            action = "CLOSE"
            to_close.append(i)
            
        print(f"W{i:<9} {vol[i]:<15,.1f} {u_pct:<18.2f} {action}")
        metrics_parts.append(f"W{i}:{u_pct:.1f}%")

    return to_close, " | ".join(metrics_parts)

def extract_phase2_demand(d, split=60): 
    return d[:, :, split:]

def build_phase2_stock(res, params, split=60):
    inv = res['End_Inv']
    I = params['NUM_WAREHOUSES']
    K = params['NUM_PARTS']

    initial_stock = {}
    t_last_phase1 = split - 1

    for i in range(I):
        for k in range(K):
            # Carry over the ending inventory from the final period of Phase 1
            inv_var = inv.get((i, k, t_last_phase1))
            if inv_var is not None:
                initial_stock[(i, k)] = inv_var.X
            else:
                initial_stock[(i, k)] = 0.0

    return initial_stock



# ==============================================================================
# 5. VALIDATION & REPORTING
# ==============================================================================

# Required keys for cost result dictionaries
REQUIRED_COST_KEYS = ["Total Cost", "Operating", "Transport_Outbound", "Transport_Inbound", "Holding", "Backorder_Penalty"]

def validate_result_dict(result, result_name, required_keys=REQUIRED_COST_KEYS):
    """
    Validates that a result dictionary contains all required keys.
    Raises KeyError with descriptive message if any key is missing.
    """
    if result is None:
        raise ValueError(f"VALIDATION ERROR: {result_name} is None - model may have failed.")
    
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        raise KeyError(f"VALIDATION ERROR: {result_name} is missing required keys: {missing_keys}. "
                       f"Available keys: {list(result.keys())}")

def extract_period_range_costs(res_full, d_full, params, start_t, end_t):
    """
    Extracts costs for a specific period range [start_t, end_t) from a solved model's results.
    Raises KeyError if required model outputs are missing.
    
    Args:
        res_full: Full model result dictionary with x_vars, Order_Qty, End_Inv
        d_full: Full demand array (Customers x Parts x Periods)
        params: Parameter dictionary
        start_t: Start period (inclusive)
        end_t: End period (exclusive)
    
    Returns:
        Dictionary with cost breakdown for the specified period range
    """
    # Validate result is not None
    if res_full is None:
        raise ValueError("VALIDATION ERROR: res_full is None - model may have failed to solve.")
    
    # Validate required model outputs exist
    if "x_vars" not in res_full:
        raise KeyError("VALIDATION ERROR: res_full missing 'x_vars' - model may have failed.")
    if "Order_Qty" not in res_full:
        raise KeyError("VALIDATION ERROR: res_full missing 'Order_Qty' - model may have failed.")
    if "End_Inv" not in res_full:
        raise KeyError("VALIDATION ERROR: res_full missing 'End_Inv' - model may have failed.")
    
    x_vars = res_full["x_vars"]
    Order = res_full["Order_Qty"]
    Inv = res_full["End_Inv"]

    I = params["NUM_WAREHOUSES"]
    J = params["NUM_CUSTOMERS"]
    K = params["NUM_PARTS"]

    f_monthly = params["f_monthly"]
    c_ijk = params["c_ijk"]
    c_in = params["c_inbound_i"]
    h_ik = params["h_ik"]
    alpha_ik = params["alpha_ik"]
    price_k = params["v_k"]  # Required - no fallback
    discount_factor = params["discount_factor"]
    
    # All warehouses open for Scenario A
    y_all_open = np.ones(I, dtype=int)
    num_periods = end_t - start_t

    # Operating Cost
    cost_ops = sum(f_monthly[i] * y_all_open[i] for i in range(I)) * num_periods

    cost_out = 0.0
    cost_in = 0.0
    cost_hold = 0.0
    cost_backorder = 0.0

    # Outbound + Backorder Risk
    for (i, j, k, t), var_obj in x_vars.items():
        if start_t <= t < end_t:
            if var_obj.X > 1e-6:
                dem = float(d_full[j, k, t])
                cost_out += c_ijk[i, j, k] * dem * var_obj.X
                risk_per_unit = (1.0 - alpha_ik[i, k]) * (discount_factor * price_k[k])
                cost_backorder += risk_per_unit * dem * var_obj.X

    # Inbound + Holding
    for i in range(I):
        for k in range(K):
            for t in range(start_t, end_t):
                # Inbound - Order variable must exist for all valid (i,k,t)
                q = Order[i, k, t].X if (i, k, t) in Order else 0.0  # Legitimate: some periods may have no order
                cost_in += c_in[i] * q

                # Holding
                if t == 0:
                    prev_inv = 0.0  # Initial inventory handled separately
                else:
                    prev_inv = Inv[i, k, t-1].X if (i, k, t-1) in Inv else 0.0  # Legitimate: boundary condition

                available_inv = prev_inv + q
                cost_hold += h_ik[i, k] * available_inv

    return {
        "Total Cost": cost_ops + cost_out + cost_in + cost_hold + cost_backorder,
        "Operating": cost_ops,
        "Transport_Outbound": cost_out,
        "Transport_Inbound": cost_in,
        "Holding": cost_hold,
        "Backorder_Penalty": cost_backorder,
        "Closing_Cost": 0.0,
    }

def print_comparison_report(run_id, res_A, res_B, res_B1, res_B2, res_A_first60=None,
                            time_A=0, time_B1=0, time_B2=0,
                            gap_A=0, gap_B1=0, gap_B2=0,
                            stop_rule_A="N/A", stop_rule_B1="N/A", stop_rule_B2="N/A",
                            warehouses_closed=None):
    """
    Prints: (1) A First 60 vs B1, (2) B1 table, (3) B2 table, (4) Final A vs B comparison for 120 periods.
    Raises KeyError if any required cost component is missing.
    """
    # Validate all result dictionaries upfront
    validate_result_dict(res_A, "res_A (Scenario A)")
    validate_result_dict(res_B, "res_B (Scenario B Combined)")
    validate_result_dict(res_B1, "res_B1 (Scenario B Phase 1)")
    validate_result_dict(res_B2, "res_B2 (Scenario B Phase 2)")

    if warehouses_closed is None:
        warehouses_closed = []

    # ========== TABLE 0: A (First 60) vs B1 Comparison ==========
    if res_A_first60 is not None:
        validate_result_dict(res_A_first60, "res_A_first60 (Scenario A First 60)")
        
        diff_60 = res_A_first60['Total Cost'] - res_B1['Total Cost']
        pct_diff_60 = (diff_60 / res_A_first60['Total Cost']) * 100 if res_A_first60['Total Cost'] > 0 else 0
        
        def fmt_row_60(label, val_a, val_b):
            diff_val = val_a - val_b
            diff_pct = (diff_val / val_a) * 100 if val_a > 0 else 0
            str_a = f"${val_a:,.0f}"
            str_b = f"${val_b:,.0f}"
            str_diff = f"${diff_val:+,.0f} ({diff_pct:+.2f}%)"
            print(f"{label:<22} {str_a:<18} {str_b:<18} {str_diff}")
        
        print("\n" + "=" * 80)
        print(f"   FIRST 60 PERIODS: A (extracted) vs B1 (Run {run_id})")
        print("=" * 80)
        print(f"{'Metric':<22} {'A (First 60)':<18} {'B1 (Isolated)':<18} {'Difference'}")
        print("-" * 80)
        fmt_row_60("Total Cost", res_A_first60['Total Cost'], res_B1['Total Cost'])
        print("-" * 80)
        fmt_row_60(" - Operating", res_A_first60['Operating'], res_B1['Operating'])
        fmt_row_60(" - Outbound Trans", res_A_first60['Transport_Outbound'], res_B1['Transport_Outbound'])
        fmt_row_60(" - Inbound Trans", res_A_first60['Transport_Inbound'], res_B1['Transport_Inbound'])
        fmt_row_60(" - Holding", res_A_first60['Holding'], res_B1['Holding'])
        fmt_row_60(" - Backorder Pen", res_A_first60['Backorder_Penalty'], res_B1['Backorder_Penalty'])
        print("=" * 80)
        if abs(diff_60) < 1:
            print("NOTE: Costs are effectively identical (< $1 difference)")
        else:
            print(f"NOTE: {'A costs more' if diff_60 > 0 else 'B1 costs more'} by ${abs(diff_60):,.0f} ({abs(pct_diff_60):.2f}%)")
        print()

    # ========== TABLE 1: Scenario B Phase 1 (First 60 Periods) ==========
    print("\n" + "=" * 60)
    print(f"   SCENARIO B - PHASE 1: First 60 Periods (Run {run_id})")
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':<20}")
    print("-" * 60)
    print(f"{'Total Cost':<25} ${res_B1['Total Cost']:,.0f}")
    print(f"{'  - Operating':<25} ${res_B1['Operating']:,.0f}")
    print(f"{'  - Outbound Trans':<25} ${res_B1['Transport_Outbound']:,.0f}")
    print(f"{'  - Inbound Trans':<25} ${res_B1['Transport_Inbound']:,.0f}")
    print(f"{'  - Holding':<25} ${res_B1['Holding']:,.0f}")
    print(f"{'  - Backorder Penalty':<25} ${res_B1['Backorder_Penalty']:,.0f}")
    print("-" * 60)
    print(f"{'Run Time (s)':<25} {time_B1:.1f}")
    print(f"{'Stopping Rule':<25} {stop_rule_B1}")
    print(f"{'MIP Gap (%)':<25} {gap_B1:.2f}")

    # ========== TABLE 2: Scenario B Phase 2 (Last 60 Periods) ==========
    print("\n" + "=" * 60)
    print(f"   SCENARIO B - PHASE 2: Last 60 Periods (Run {run_id})")
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':<20}")
    print("-" * 60)
    print(f"{'Total Cost':<25} ${res_B2['Total Cost']:,.0f}")
    print(f"{'  - Operating':<25} ${res_B2['Operating']:,.0f}")
    print(f"{'  - Outbound Trans':<25} ${res_B2['Transport_Outbound']:,.0f}")
    print(f"{'  - Inbound Trans':<25} ${res_B2['Transport_Inbound']:,.0f}")
    print(f"{'  - Holding':<25} ${res_B2['Holding']:,.0f}")
    print(f"{'  - Backorder Penalty':<25} ${res_B2['Backorder_Penalty']:,.0f}")
    print(f"{'  - Closing Cost':<25} ${res_B2.get('Closing_Cost', 0):,.0f}")  # Closing cost is optional (0 if no closures)
    print("-" * 60)
    print(f"{'Run Time (s)':<25} {time_B2:.1f}")
    print(f"{'Stopping Rule':<25} {stop_rule_B2}")
    print(f"{'MIP Gap (%)':<25} {gap_B2:.2f}")
    print(f"{'Warehouses Closed':<25} {str(warehouses_closed) if warehouses_closed else 'None'}")

    # ========== TABLE 3: Final Comparison (120 Periods) ==========
    diff = res_A['Total Cost'] - res_B['Total Cost']
    pct_savings = (diff / res_A['Total Cost']) * 100 if res_A['Total Cost'] > 0 else 0

    def fmt_row(label, val_a, total_a, val_b, total_b):
        pct_a = (val_a / total_a) * 100 if total_a > 0 else 0
        pct_b = (val_b / total_b) * 100 if total_b > 0 else 0
        str_a = f"${val_a:,.0f} ({pct_a:5.2f}%)"
        str_b = f"${val_b:,.0f} ({pct_b:5.2f}%)"
        print(f"{label:<22} {str_a:<25} {str_b:<25}")

    print("\n" + "=" * 75)
    print(f"   FINAL COMPARISON: Run {run_id} (Full 120 Periods)")
    print("=" * 75)
    print(f"{'Metric':<22} {'Scenario A (Baseline)':<25} {'Scenario B (B1 + B2)':<25}")
    print("-" * 75)
    print(f"{'Total Cost':<22} ${res_A['Total Cost']:<24,.0f} ${res_B['Total Cost']:<24,.0f}")
    print("-" * 75)

    total_a, total_b = res_A['Total Cost'], res_B['Total Cost']
    fmt_row(" - Operating",      res_A['Operating'], total_a, res_B['Operating'], total_b)
    fmt_row(" - Outbound Trans", res_A['Transport_Outbound'], total_a, res_B['Transport_Outbound'], total_b)
    fmt_row(" - Inbound Trans",  res_A['Transport_Inbound'], total_a, res_B['Transport_Inbound'], total_b)
    fmt_row(" - Holding",        res_A['Holding'], total_a, res_B['Holding'], total_b)
    fmt_row(" - Backorder Pen",  res_A['Backorder_Penalty'], total_a, res_B['Backorder_Penalty'], total_b)
    fmt_row(" - Closing Cost",   res_A.get('Closing_Cost', 0), total_a, res_B.get('Closing_Cost', 0), total_b)  # Optional
    
    print("-" * 75)
    print(f"{'Run Time (s)':<22} {time_A:<25.1f} {time_B1 + time_B2:.1f} (B1: {time_B1:.1f} | B2: {time_B2:.1f})")
    print(f"{'Stopping Rule':<22} {stop_rule_A:<25} B1: {stop_rule_B1} | B2: {stop_rule_B2}")
    print(f"{'MIP Gap (%)':<22} {gap_A:<25.2f} B1: {gap_B1:.2f} | B2: {gap_B2:.2f}")
    print(f"{'Warehouses Closed':<22} {'N/A':<25} {str(warehouses_closed)}")
    print("-" * 75)

    if diff > 0:
        print(f"RECOMMENDATION: PROCEED WITH RESTRUCTURING")
        print(f"  > Net Savings:   ${diff:,.0f}")
        print(f"  > Improvement:   {pct_savings:.2f}%")
    else:
        print(f"RECOMMENDATION: MAINTAIN STATUS QUO")
        print(f"  > Net Loss:      ${-diff:,.0f}")
    print("=" * 75)



# ==============================================================================
# 7. DATABASE LOGGING
# ==============================================================================
def append_run_to_database(run_id, design_row, res_A, res_B, util_metrics="",
                           time_A=0, time_B1=0, time_B2=0,
                           gap_A=0, gap_B1=0, gap_B2=0,
                           stop_rule_A="N/A", stop_rule_B1="N/A", stop_rule_B2="N/A",
                           warehouses_closed=None,
                           db_file="Results/Master_Experiment_DB.csv"):
    """
    Appends a single row of metrics to a CSV database for SQL analysis.
    Raises KeyError if any required cost component is missing.
    """
    # Validate result dictionaries upfront
    validate_result_dict(res_A, "res_A (Scenario A)")
    validate_result_dict(res_B, "res_B (Scenario B Combined)")

    if warehouses_closed is None:
        warehouses_closed = []

    # 1. Build Dictionary of Inputs
    row_data = {
        "Run_ID": run_id,
        "Demand_Scenario": design_row['Demand_Label'],
        "Alpha_Input": design_row['Alpha'],
        "Holding_Rate": design_row['Holding_Rate'],
        "Penalty_Factor": design_row['Penalty_Factor']
    }

    # 2. Scenario A Metrics (strict - no fallbacks)
    for k in REQUIRED_COST_KEYS:
        row_data[f"A_{k.replace(' ', '_')}"] = res_A[k]  # Direct access - will raise KeyError if missing
    
    row_data["A_Time_Seconds"] = time_A
    row_data["A_Gap_Pct"] = gap_A
    row_data["A_Stopping_Rule"] = stop_rule_A

    # 3. Scenario B Metrics (strict for required keys, optional for Closing_Cost)
    for k in REQUIRED_COST_KEYS:
        row_data[f"B_{k.replace(' ', '_')}"] = res_B[k]  # Direct access - will raise KeyError if missing
    row_data["B_Closing_Cost"] = res_B.get('Closing_Cost', 0)  # Optional - may legitimately be 0
    
    row_data["Net_Savings"] = res_A['Total Cost'] - res_B['Total Cost']
    row_data["Pct_Savings"] = (row_data["Net_Savings"] / res_A['Total Cost']) * 100 if res_A['Total Cost'] > 0 else 0

    # 4. Scenario B Timing & Solver Info
    row_data["B_First60_Time_Seconds"] = time_B1
    row_data["B_Last60_Time_Seconds"] = time_B2
    row_data["B_Total_Time_Seconds"] = time_B1 + time_B2
    row_data["B_First60_Gap_Pct"] = gap_B1
    row_data["B_Last60_Gap_Pct"] = gap_B2
    row_data["B_First60_Stopping_Rule"] = stop_rule_B1
    row_data["B_Last60_Stopping_Rule"] = stop_rule_B2
    
    # 5. Warehouse Utilization & Closures
    row_data["Utilization_Breakdown"] = util_metrics
    row_data["Warehouses_Closed"] = str(warehouses_closed)
    row_data["Num_Warehouses_Closed"] = len(warehouses_closed)

    # 6. Save to CSV
    df = pd.DataFrame([row_data])
    
    if not os.path.exists(db_file):
        df.to_csv(db_file, index=False)
    else:
        df.to_csv(db_file, mode='a', header=False, index=False)
    
    print(f"    📊 Logged Run {run_id} to '{db_file}'")
