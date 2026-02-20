import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_inventory_model(name, demand_data, y_config, one_time_cost, params, initial_stock, env, output_dir="."):
    """
    Solves the Fixed-Network Multi-Period Inventory Model using Gurobi.
    
    Args:
        name (str): Name for the Gurobi model instance.
        demand_data (np.array): 3D array of demand [Customers, Parts, Periods].
        y_config (np.array): 1D array indicating open (1) or closed (0) warehouses.
        one_time_cost (float): Fixed cost penalty (e.g., closing costs) to add to the objective.
        params (dict): Dictionary containing all static parameters (c_ijk, h_ik, etc.).
        initial_stock (dict): Dictionary {(i,k): qty} for starting inventory.
        env (gp.Env): Gurobi environment to use.
        Solves the model and saves .log, .lp, and .ilp files to output_dir.

    Returns:
        dict: Results dictionary containing 'Total Cost', 'x_vars', 'Order_Qty', etc., or None if infeasible.
    """

    # 1. Unpack Dimensions & Parameters
    # ---------------------------------
    num_periods = demand_data.shape[2] 
    I = params['NUM_WAREHOUSES']
    J = params['NUM_CUSTOMERS']
    K = params['NUM_PARTS']
    s_max_ik = params['s_max_ik']

    tau = params['tau']
    w_j = params['w_j']
    discount_factor = params['discount_factor']

    print(f"\n--- Starting Solver: {name} ---")
    print(f"    Time Horizon: {num_periods} periods")
    print(f"    Open Warehouses: {np.sum(y_config)} / {I}")

    # 2. Initialize Model
    # ---------------------------------
    # Use the passed environment 'env' to ensure thread safety / license usage
    m = gp.Model(name, env=env)
    
    # Force Log file to specific directory
    log_path = os.path.join(output_dir, f"{name}.log")
    m.setParam('LogFile', log_path)

    m.setParam("MIPGap", 0.005)
    m.setParam("TimeLimit", params.get('time_limit', 2000))
    m.setParam("MIPFocus", 1)

    threads = params.get('threads', 0)
    if threads > 0:
        m.setParam("Threads", threads)


    # 3. Create Variables
    # ---------------------------------
    # Optimization: Only create x for reachable customers AND open warehouses
    valid_arcs = []
    for i in range(I):
        for j in range(J):
            # Check reachability (time <= window) AND if warehouse is open
            if params['tau'][i, j] <= params['w_j'][j] and y_config[i] == 1:
                valid_arcs.append((i, j))

    # x: Demand served (Binary) - defined for valid arcs
    x = m.addVars(valid_arcs, range(K), range(num_periods),
                  vtype=GRB.BINARY, lb=0.0, ub=1.0, name="x")

    # V: Inventory Level Choice (Only for open warehouses)
    open_whs = [i for i in range(I) if y_config[i] == 1]
    
    # V[i, k, s, t]: Binary choice of target stock level 's'
    # Per-(i,k) bounds drastically reduce variable count
    v_indices = []
    for i in open_whs:
        for k in range(K):
            s_bound = s_max_ik[(i, k)]
            for s in range(1, s_bound + 1):
                for t in range(num_periods):
                    v_indices.append((i, k, s, t))

    V = m.addVars(v_indices, vtype=GRB.BINARY, name="V")
    print(f"    V variables: {len(v_indices):,} (per-(i,k) bounded)")

    # Physical Flow Variables
    # Order_Qty: Amount shipped from DC to warehouse i
    Order_Qty = m.addVars(open_whs, range(K), range(num_periods),
                          vtype=GRB.CONTINUOUS, lb=0.0, name="Order")

    # End_Inv: Inventory on hand at the end of the period
    End_Inv = m.addVars(open_whs, range(K), range(num_periods),
                        vtype=GRB.CONTINUOUS, lb=0.0, name="Inv")

    # 4. Objective Function
    # ---------------------------------
    # A. Operating Cost (Fixed Monthly cost for open facilities)
    obj_ops = gp.quicksum(
        params['f_monthly'][i] * y_config[i]
        for i in range(I)
        for t in range(num_periods)
    )

    # B1. Outbound Transportation Cost (Facility -> Customer)
    # Only sum over valid arcs and where demand > 0
    obj_trans = gp.quicksum(
        params['c_ijk'][i, j, k] * demand_data[j, k, t] * x[i, j, k, t]
        for i, j in valid_arcs
        for k in range(K)
        for t in range(num_periods)
        if demand_data[j, k, t] > 0
    )

    # B2. Inbound Transportation Cost (DC -> Facility)
    # Depends on Order_Qty (Shipments)
    obj_trans_in = gp.quicksum(
        params['c_inbound_i'][i] * Order_Qty[i, k, t]
        for i in open_whs
        for k in range(K)
        for t in range(num_periods)
    )

    # B3. Expected Backorder Risk Penalty
    # Cost = (Risk Probability) * (Penalty Factor) * (Price)
    # Risk Probability = (1 - alpha)
    # Pre-calculate terms for efficiency
    obj_penalty_list = []
    for i, j in valid_arcs:
        for k in range(K):
            # Calculate the "Risk Surcharge"
            price = params.get('v_k')[k]
            alpha = params['alpha_ik'][i, k]
            risk_cost_per_unit = (1.0 - alpha) * (discount_factor * price)

            if risk_cost_per_unit > 0:
                for t in range(num_periods):
                    if demand_data[j, k, t] > 0:
                        obj_penalty_list.append(
                            risk_cost_per_unit * demand_data[j, k, t] * x[i, j, k, t]
                        )
    obj_penalty_cost = gp.quicksum(obj_penalty_list)

    # C. Holding Cost
    # Logic: We hold stock for the period based on max available (Start + Order)
    obj_hold = 0
    for i in open_whs:
        for k in range(K):
            for t in range(num_periods):
                # If t=0, previous inventory is Initial Stock
                prev_inv = initial_stock[(i,k)] if t == 0 else End_Inv[i, k, t-1]
                available_inv = prev_inv + Order_Qty[i, k, t]
                obj_hold += params['h_ik'][i, k] * available_inv

    # Set Total Objective
    # Total = Ops + Outbound + Inbound + Risk + Hold + One-Time-Costs (Closing)
    m.setObjective(obj_ops + obj_trans + obj_trans_in + obj_penalty_cost + obj_hold + one_time_cost, GRB.MINIMIZE)

    # 5. Constraints
    # ----------------------------------
    # C1: Single Sourcing
    # Every customer demand must be served by exactly one valid warehouse
    for j in range(J):
        # Identify valid sources for this customer from open warehouses
        sources = [i for i in open_whs if params['tau'][i, j] <= params['w_j'][j]]

        # Safety Check: If no sources, model is infeasible for this customer
        if not sources:
            print(f"    CRITICAL WARNING: Customer {j} has NO valid open warehouses!")
            return None

        for k in range(K):
            for t in range(num_periods):
                if demand_data[j, k, t] > 0:
                    m.addConstr(gp.quicksum(x[i, j, k, t] for i in sources) == 1)

    # C2: Capacity (Service Level)
    for i in open_whs:
        valid_cust = [j for j in range(J) if params['tau'][i, j] <= params['w_j'][j]]

        for k in range(K):
            target_alpha = round(params['alpha_ik'][i, k], 2)
            s_bound = s_max_ik[(i, k)]
            for t in range(num_periods):
                lhs = gp.quicksum(demand_data[j, k, t] * x[i, j, k, t] for j in valid_cust)
                rhs = gp.quicksum(params['lambda_lookup'].get((s, target_alpha), 0.0) * V[i, k, s, t]
                                  for s in range(1, s_bound + 1))
                m.addConstr(lhs <= rhs)

    # C3: Inventory Selection
    for i in open_whs:
        for k in range(K):
            s_bound = s_max_ik[(i, k)]
            for t in range(num_periods):
                m.addConstr(gp.quicksum(V[i, k, s, t] for s in range(1, s_bound + 1)) <= 1)

    # C4: Inventory Flow Conservation
    for i in open_whs:
        valid_cust = [j for j in range(J) if params['tau'][i, j] <= params['w_j'][j]]

        for k in range(K):
            s_bound = s_max_ik[(i, k)]
            for t in range(num_periods):
                sales_qty = gp.quicksum(demand_data[j, k, t] * x[i, j, k, t] for j in valid_cust)
                prev_inv = initial_stock[(i,k)] if t == 0 else End_Inv[i, k, t-1]

                m.addConstr(End_Inv[i, k, t] == prev_inv + Order_Qty[i, k, t] - sales_qty,
                            name=f"Flow_{i}_{k}_{t}")

                target_S_level = gp.quicksum(s * V[i, k, s, t] for s in range(1, s_bound + 1))
                m.addConstr(prev_inv + Order_Qty[i, k, t] >= target_S_level,
                            name=f"TargetAvailability_{i}_{k}_{t}")

    # ==========================================
    # 6. Optimize & Return
    # ==========================================
    m.optimize()
    

    # Process Results
    if m.SolCount > 0:
        # Save Solution File
        sol_path = os.path.join(output_dir, f"{name}.sol")
        m.write(sol_path)
        print(f"    ✅ Saved solution file: {sol_path}")

        backorder_penalty_val = obj_penalty_cost.getValue()

        res = {
            "status": "OPTIMAL",
            "Total Cost": m.objVal,
            "Operating": obj_ops if isinstance(obj_ops, float) else obj_ops.getValue(),
            "Transport_Outbound": obj_trans.getValue(),
            "Transport_Inbound": obj_trans_in.getValue(),
            "Holding": obj_hold.getValue(),
            "Backorder_Penalty": backorder_penalty_val,
            "Closing_Cost": one_time_cost,
            "Total_Penalty": backorder_penalty_val + one_time_cost,
            "Solve_Time": m.Runtime,
            "MIP_Gap": m.MIPGap * 100,  # Convert to percentage
            "Stopping_Rule": f"Time Limit ({m.Params.TimeLimit:.0f}s)" if m.status == GRB.TIME_LIMIT else f"Gap ({m.Params.MIPGap*100:.1f}%)",
            "Model_Obj": m,
            "x_vars": x,
            "V_vars": V,
            "Order_Qty": Order_Qty,
            "End_Inv": End_Inv
        }
        
        # Log success
        if m.status == GRB.OPTIMAL:
            print(f"    Success! Optimal Cost: ${res['Total Cost']:,.2f} (Solved in {m.Runtime:.2f}s)")
        else:
            print(f"    Solver Stopped (Status {m.status}). Best Solution: ${res['Total Cost']:,.2f}")
            
        return res

    elif m.status == GRB.INFEASIBLE:
        print(f"    FAILURE: Model '{name}' is Infeasible.")
        m.computeIIS()
        m.write(f"{name}_debug.ilp")
        return None
    else:
        print(f"    Optimization Failed. Status Code: {m.status}")
        return None