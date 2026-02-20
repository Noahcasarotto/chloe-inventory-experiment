
import pandas as pd
import numpy as np
import time
import gurobipy as gp
import os, sys, shutil, json
import helpers
from model_core import solve_inventory_model

# ==============================================================================
# 1. DATA LOADING (Config + Topology)
# ==============================================================================
def load_environment_data():
    """
    Loads all static configuration and topology files.
    Returns a dictionary of calculated parameters ready for Gurobi.
    """
    print("🔹 Loading Environment Data...")
    
    # A. Load Configuration (JSON)
    if not os.path.exists("static_params.json"):
        raise FileNotFoundError("CRITICAL: 'static_params.json' missing. Run setup_experiment.py first.")
    
    with open("static_params.json", 'r') as f:
        config = json.load(f)
        
    # Convert lists back to numpy arrays
    config['f_monthly'] = np.array(config['f_monthly'])
    config['v_k'] = np.array(config['v_k'])
    
    # B. Load Distance Matrix
    dist_file = "distance_matrix.csv"
    if not os.path.exists(dist_file):
        raise FileNotFoundError(f"CRITICAL: '{dist_file}' not found.")
    
    try:
        df_dist = pd.read_csv(dist_file, encoding='utf-8')
    except UnicodeDecodeError:
        df_dist = pd.read_csv(dist_file, encoding='latin1')

    # Parse Matrix (Rows 21-25 -> Facilities, Cols 1-20 -> Customers)
    dist_matrix_outbound = df_dist.iloc[21:26, 1:21].to_numpy(dtype=float)
    dist_dc_to_facility = df_dist.iloc[20, 22:27].to_numpy(dtype=float)

    # C. Load Lambda Capacity Table
    lambda_file = "lambda_capacity_table.csv"
    if not os.path.exists(lambda_file):
        raise FileNotFoundError(f"CRITICAL: '{lambda_file}' missing.")
    
    df_lambda = pd.read_csv(lambda_file)
    lambda_lookup = {}
    for _, row in df_lambda.iterrows():
        s_val = int(row['Stock_Level_s'])
        alpha_val = round(float(row['Service_Level_alpha']), 2)
        lambda_lookup[(s_val, alpha_val)] = float(row['Max_Lambda_Capacity'])

    # D. Calculate Logistics Parameters
    # 1. Outbound Costs (c_ijk)
    c_ijk = np.zeros((config['NUM_WAREHOUSES'], config['NUM_CUSTOMERS'], config['NUM_PARTS']))
    for k in range(config['NUM_PARTS']):
        c_ijk[:, :, k] = config['cost_per_km_out'] * dist_matrix_outbound

    # 2. Inbound Costs
    c_inbound_i = dist_dc_to_facility * config['cost_per_km_in']

    # 3. Time Windows
    tau = dist_matrix_outbound / config['avg_speed_kph']
    min_travel_times = np.min(tau, axis=0)
    w_j = np.zeros(config['NUM_CUSTOMERS'])
    
    for j in range(config['NUM_CUSTOMERS']):
        fastest = min_travel_times[j]
        if fastest + config['order_proc_time'] <= 24.0:
            w_j[j] = 24.0
        else:
            w_j[j] = fastest + config['order_proc_time'] + 2.0

    # Merge everything into one clean dictionary
    env_params = config.copy()
    env_params.update({
        'c_ijk': c_ijk,
        'c_inbound_i': c_inbound_i,
        'tau': tau,
        'w_j': w_j,
        'lambda_lookup': lambda_lookup,
        'dist_matrix': dist_matrix_outbound # Saved for Inspection Report
    })
    
    return env_params

# ==============================================================================
# 2. MASTER EXECUTION LOOP
# ==============================================================================
def run_batch(start_id=None, end_id=None, run_ids=None, solver_config=None):

    # 1. Load Static Assets Once
    ENV_PARAMS = load_environment_data()
    df_design = pd.read_csv("experiment_design_master.csv")

    # ----------------------------------------------------
    # Per-(i,k) S_MAX Calculation (Pre-Experiment)
    # ----------------------------------------------------
    print("\n🔹 PRE-CALCULATION: Computing per-warehouse S_MAX bounds...")

    I = ENV_PARAMS['NUM_WAREHOUSES']
    J = ENV_PARAMS['NUM_CUSTOMERS']
    K = ENV_PARAMS['NUM_PARTS']
    tau = ENV_PARAMS['tau']
    w_j = ENV_PARAMS['w_j']

    reachable = {}
    for i in range(I):
        reachable[i] = [j for j in range(J) if tau[i, j] <= w_j[j]]
        print(f"    Warehouse {i}: {len(reachable[i])} reachable customers")

    unique_demand_labels = df_design['Demand_Label'].unique()
    demand_cache = {}
    for label in unique_demand_labels:
        d_filename = f"demand_{label}.npy"
        if os.path.exists(d_filename):
            demand_cache[label] = np.load(d_filename)

    s_max_ik = {}
    for i in range(I):
        for k in range(K):
            peak = 0
            for label, d_temp in demand_cache.items():
                T_len = d_temp.shape[2]
                for t in range(T_len):
                    period_demand = sum(int(d_temp[j, k, t]) for j in reachable[i])
                    if period_demand > peak:
                        peak = period_demand
            s_max_ik[(i, k)] = max(int(np.ceil(peak * 1.3)), 10)

    total_v_old = I * K * int(np.ceil(max(
        np.max(np.sum(d, axis=0)) for d in demand_cache.values()
    ) * 1.5))
    total_v_new = sum(s_max_ik.values())
    print(f"    ✅ Per-(i,k) S_MAX computed. V variable reduction: {total_v_old:,} -> {total_v_new:,} "
          f"({100*(1 - total_v_new/total_v_old):.0f}% fewer per period)\n")

    
    # 2. Start Gurobi Env
    GLOBAL_ENV = gp.Env()
    GLOBAL_ENV.start()

    # Determine which runs to process
    if run_ids is not None:
        runs_to_process = run_ids
        print(f"\n🚀 STARTING SPECIFIC RUNS: {run_ids}")
        print(f"   Total Runs: {len(run_ids)}")
        # print(f"   Estimated Time: {len(run_ids) * 1100 / 60:.1f} minutes\n")
    else:
        runs_to_process = range(start_id, end_id + 1)
        print(f"\n🚀 STARTING BATCH: Runs {start_id} to {end_id}")
    
    for run_id in runs_to_process:

        # 3. Get Run Settings
        row = df_design.loc[df_design['Run_ID'] == run_id].iloc[0]
        

        # ... [Directory creation / Logging setup] ...
        # Ensure we use your preferred structure
        run_dir = f"Results/Run_{run_id}"

        if os.path.exists(run_dir): shutil.rmtree(run_dir)
        os.makedirs(run_dir)

        # Save current stdout (could be master logger) before redirecting to per-run log
        parent_stdout = sys.stdout
        sys.stdout = helpers.Logger(f"{run_dir}/Run{run_id}_log.txt")
        print(f"🔹 EXECUTION: Run {run_id} | Demand: {row['Demand_Label']} | Alpha: {row['Alpha']}")

        # 4. Load Demand
        d_filename = f"demand_{row['Demand_Label']}.npy"
        if not os.path.exists(d_filename):
            print(f"❌ Error: {d_filename} not found.")
            continue
        d_curr = np.load(d_filename)
        
        # 6. Apply Design Variables (Alpha, Holding, Penalty) to Params
        params = ENV_PARAMS.copy()
        params['s_max_ik'] = s_max_ik
        params['alpha_ik'] = np.full((params['NUM_WAREHOUSES'], params['NUM_PARTS']), row['Alpha'])
        params['h_ik'] = np.array([(params['v_k'] * row['Holding_Rate']) / 12.0 for _ in range(params['NUM_WAREHOUSES'])])
        params['discount_factor'] = row['Penalty_Factor']

        if solver_config:
            params.update(solver_config)

        # # --- SCENARIO A: Full 120 Periods ---
        print("\n--- SCENARIO A: BASELINE (120 Periods) ---")

        # Initialize with 10 units of each part in each warehouse
        init_A = {}
        for i in range(params['NUM_WAREHOUSES']):
            for k in range(params['NUM_PARTS']):
                init_A[(i, k)] = 10

        # All warehouses open
        y_A = np.ones(params['NUM_WAREHOUSES'], dtype=int)
        
        # Generate Inspection Report
        helpers.generate_data_inspection_report(params, d_curr, init_A, run_id, row)

        #---------------------------------------------
        # Solve Scenario A
        #---------------------------------------------
        t_start_A = time.time()
        res_A = solve_inventory_model(f"Run_{run_id}_A", d_curr, y_A, 0.0, params, init_A, GLOBAL_ENV, output_dir=run_dir)
        t_end_A = time.time()
        time_A = t_end_A - t_start_A

        # Run Validation Audit
        helpers.run_model_validation_audit(res_A, d_curr, init_A, params, "Scenario A")

        # Save Results
        helpers.save_consolidated_inventory_plan(res_A, d_curr, "ScenarioA", run_dir, run_id)

        # Save Shipments
        helpers.save_shipments_to_csv(res_A, d_curr, "ScenarioA", run_dir, run_id)

        #---------------------------------------------
        # --- SCENARIO B: Two-Phase Approach ---
        #---------------------------------------------
        print("\n--- SCENARIO B: TWO-PHASE APPROACH ---")
        # Phase 1: First 60 periods (all warehouses open)
        print("\n  [Phase 1] First 60 Periods (All Warehouses Open)")
        d_first60 = d_curr[:, :, :60]
        init_B1 = {(i, k): 10 for i in range(params['NUM_WAREHOUSES']) for k in range(params['NUM_PARTS'])}
        y_B1 = np.ones(params['NUM_WAREHOUSES'], dtype=int)

        t_start_B1 = time.time()
        res_B1 = solve_inventory_model(f"Run_{run_id}_B1", d_first60, y_B1, 0.0, params, init_B1, GLOBAL_ENV, output_dir=run_dir)
        time_B1 = time.time() - t_start_B1
        helpers.run_model_validation_audit(res_B1, d_first60, init_B1, params, "Scenario B Phase 1")
        helpers.save_consolidated_inventory_plan(res_B1, d_first60, "ScenarioB_Phase1", run_dir, run_id)
        helpers.save_shipments_to_csv(res_B1, d_first60, "ScenarioB_Phase1", run_dir, run_id)

        # Utilization Check (on Phase 1 results)
        to_close, util_metrics = helpers.check_warehouse_utilization(res_B1, params, demand_data=d_first60)
        print(f"\n    Warehouses to Close: {to_close if to_close else 'None'}")

        #---------------------------------------------
        # Phase 2: Last 60 periods (reduced warehouses)
        #---------------------------------------------
        print(f"\n  [Phase 2] Last 60 Periods ({params['NUM_WAREHOUSES'] - len(to_close)} Warehouses Open)")
        d_last60 = d_curr[:, :, 60:]
        init_B2 = helpers.build_phase2_stock(res_B1, params, split=60)
        
        y_B2 = np.ones(params['NUM_WAREHOUSES'], dtype=int)
        y_B2[to_close] = 0

        # Calculate Closing Penalty 
        closing_penalty = len(to_close) * params['closing_cost_per_site']

        t_start_B2 = time.time()
        res_B2 = solve_inventory_model(f"Run_{run_id}_B2", d_last60, y_B2, closing_penalty, params, init_B2, GLOBAL_ENV, output_dir=run_dir)
        time_B2 = time.time() - t_start_B2

        helpers.run_model_validation_audit(res_B2, d_last60, init_B2, params, "Scenario B Phase 2", closed_warehouses=to_close)
        helpers.save_consolidated_inventory_plan(res_B2, d_last60, "ScenarioB_Phase2", run_dir, run_id)
        helpers.save_shipments_to_csv(res_B2, d_last60, "ScenarioB_Phase2", run_dir, run_id)


        # --- Combine Scenario B Costs ---
        res_B_combined = {
            "Total Cost": res_B1['Total Cost'] + res_B2['Total Cost'],
            "Operating": res_B1['Operating'] + res_B2['Operating'],
            "Transport_Outbound": res_B1['Transport_Outbound'] + res_B2['Transport_Outbound'],
            "Transport_Inbound": res_B1['Transport_Inbound'] + res_B2['Transport_Inbound'],
            "Holding": res_B1['Holding'] + res_B2['Holding'],
            "Backorder_Penalty": res_B1['Backorder_Penalty'] + res_B2['Backorder_Penalty'],
            "Closing_Cost": res_B2['Closing_Cost'],  # Only Phase 2 has closing cost
        }

        # --- Extract A's first 60 periods for comparison with B1 ---
        res_A_first60 = helpers.extract_period_range_costs(res_A, d_curr, params, start_t=0, end_t=60)

        # --- Final Comparison ---
        print("\n--- FINAL COMPARISON (120 Periods) ---")
        helpers.print_comparison_report(
            run_id=run_id,
            res_A=res_A,
            res_B=res_B_combined,
            res_B1=res_B1,
            res_B2=res_B2,
            res_A_first60=res_A_first60,
            time_A=time_A,
            time_B1=time_B1,
            time_B2=time_B2,
            gap_A=res_A.get('MIP_Gap', 0),
            gap_B1=res_B1.get('MIP_Gap', 0),
            gap_B2=res_B2.get('MIP_Gap', 0),
            stop_rule_A=res_A.get('Stopping_Rule', 'N/A'),
            stop_rule_B1=res_B1.get('Stopping_Rule', 'N/A'),
            stop_rule_B2=res_B2.get('Stopping_Rule', 'N/A'),
            warehouses_closed=to_close
        )
        # Log to Master Database
        helpers.append_run_to_database(
            run_id=run_id,
            design_row=row,
            res_A=res_A,
            res_B=res_B_combined,
            util_metrics=util_metrics,
            time_A=time_A,
            time_B1=time_B1,
            time_B2=time_B2,
            gap_A=res_A.get('MIP_Gap', 0),
            gap_B1=res_B1.get('MIP_Gap', 0),
            gap_B2=res_B2.get('MIP_Gap', 0),
            stop_rule_A=res_A.get('Stopping_Rule', 'N/A'),
            stop_rule_B1=res_B1.get('Stopping_Rule', 'N/A'),
            stop_rule_B2=res_B2.get('Stopping_Rule', 'N/A'),
            warehouses_closed=to_close
        )
        # Restore parent stdout (could be master logger or terminal)
        sys.stdout = parent_stdout
        print(f"✅ Run {run_id} Complete.")







       