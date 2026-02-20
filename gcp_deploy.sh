#!/bin/bash
# ============================================================
# GCP Deployment Script for Gurobi Inventory Experiment
# ============================================================
# Prerequisites:
#   1. gcloud CLI installed and authenticated (gcloud auth login)
#   2. A GCP project with billing enabled
#   3. ~/gurobi.lic exists (your WLS academic license)
#
# Usage:
#   chmod +x gcp_deploy.sh
#   ./gcp_deploy.sh              # Create VM + upload + install
#   ./gcp_deploy.sh run          # Start the experiment
#   ./gcp_deploy.sh status       # Check progress
#   ./gcp_deploy.sh download     # Download results
#   ./gcp_deploy.sh destroy      # Delete the VM (stop billing!)
# ============================================================

set -e

# ----- CONFIGURATION (edit these) -----
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
ZONE="us-central1-a"
INSTANCE_NAME="gurobi-solver"
MACHINE_TYPE="c2d-standard-32"   # 32 vCPUs, 128 GB RAM, ~$1.16/hr
NUM_WORKERS=4                     # parallel worker processes
THREADS_PER_WORKER=8              # Gurobi threads per worker (32 / 4 = 8)
MIP_GAP=0.005                     # 0.5% gap (change to 0.01 for faster 1% gap)
TIME_LIMIT=2000                   # seconds per solve
RUNS="all"                        # "all" for 0-80, or "0-26" etc.

# Local project directory
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  GCP Gurobi Experiment Deployer"
echo "============================================"
echo "  Project:  $PROJECT_ID"
echo "  Zone:     $ZONE"
echo "  Machine:  $MACHINE_TYPE"
echo "  Workers:  $NUM_WORKERS x $THREADS_PER_WORKER threads"
echo "============================================"

# ---- Helper: SSH wrapper ----
gssh() {
    gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="$1"
}

gscp_up() {
    gcloud compute scp "$1" "$INSTANCE_NAME:$2" --zone="$ZONE"
}

gscp_down() {
    gcloud compute scp --recurse "$INSTANCE_NAME:$1" "$2" --zone="$ZONE"
}

# ============================================================
# COMMAND: setup (default)
# ============================================================
do_setup() {
    echo ""
    echo "[1/4] Creating VM: $INSTANCE_NAME ($MACHINE_TYPE)..."
    gcloud compute instances create "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=30GB \
        --boot-disk-type=pd-ssd \
        --scopes=default \
        --quiet

    echo "  Waiting for VM to be ready..."
    sleep 15

    echo ""
    echo "[2/4] Installing Python dependencies..."
    gssh "sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip > /dev/null 2>&1 && pip3 install --quiet gurobipy numpy pandas scipy"

    echo ""
    echo "[3/4] Uploading project files..."
    gssh "mkdir -p ~/experiment"

    # Upload Python code
    for f in main_runner.py run_experiment.py model_core.py setup_experiment.py helpers.py; do
        gscp_up "$PROJ_DIR/$f" "~/experiment/$f"
    done

    # Upload data files
    for f in static_params.json distance_matrix.csv experiment_design_master.csv lambda_capacity_table.csv; do
        gscp_up "$PROJ_DIR/$f" "~/experiment/$f"
    done

    # Upload demand files
    for f in demand_Base.npy demand_Base.csv demand_High_5pct.npy demand_High_5pct.csv demand_Low_5pct.npy demand_Low_5pct.csv; do
        gscp_up "$PROJ_DIR/$f" "~/experiment/$f"
    done

    # Upload Gurobi license
    if [ -f ~/gurobi.lic ]; then
        gscp_up ~/gurobi.lic "~/gurobi.lic"
        echo "  Gurobi license uploaded."
    else
        echo "  WARNING: ~/gurobi.lic not found! You'll need to upload it manually."
    fi

    echo ""
    echo "[4/4] Verifying setup..."
    gssh "cd ~/experiment && python3 -c \"import gurobipy; print('Gurobi OK:', gurobipy.gurobi.version())\""
    gssh "cd ~/experiment && python3 -c \"import numpy, pandas, scipy; print('Dependencies OK')\""
    gssh "ls -la ~/experiment/*.py ~/experiment/*.json ~/experiment/*.csv ~/experiment/*.npy 2>/dev/null | wc -l | xargs -I{} echo '  {} files uploaded'"

    echo ""
    echo "============================================"
    echo "  VM READY!"
    echo "  Cost: ~\$1.16/hr while running"
    echo ""
    echo "  Next steps:"
    echo "    ./gcp_deploy.sh run       # Start the experiment"
    echo "    ./gcp_deploy.sh status    # Check progress"
    echo "    ./gcp_deploy.sh download  # Get results when done"
    echo "    ./gcp_deploy.sh destroy   # DELETE VM (stop billing!)"
    echo "============================================"
}

# ============================================================
# COMMAND: run
# ============================================================
do_run() {
    echo "Starting experiment on VM..."
    echo "  Runs: $RUNS  |  Workers: $NUM_WORKERS  |  Gap: $MIP_GAP"

    # Run inside tmux so it persists after SSH disconnect
    gssh "tmux new-session -d -s experiment 'cd ~/experiment && python3 main_runner.py --runs $RUNS --workers $NUM_WORKERS --threads $((NUM_WORKERS * THREADS_PER_WORKER)) --gap $MIP_GAP --timelimit $TIME_LIMIT 2>&1 | tee Results/run_output.log; echo EXPERIMENT_COMPLETE'" || true

    echo ""
    echo "Experiment launched in background tmux session."
    echo "  Monitor:  ./gcp_deploy.sh status"
    echo "  SSH in:   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -t 'tmux attach -t experiment'"
}

# ============================================================
# COMMAND: status
# ============================================================
do_status() {
    echo "Checking experiment status..."
    echo ""

    # Check if tmux session exists
    gssh "tmux has-session -t experiment 2>/dev/null && echo 'STATUS: RUNNING' || echo 'STATUS: NOT RUNNING (finished or not started)'"

    echo ""
    echo "--- Completed Runs ---"
    gssh "cd ~/experiment && ls -d Results/Run_*/Run*_log.txt 2>/dev/null | wc -l | xargs -I{} echo '{} runs completed'" || echo "  No results yet"

    echo ""
    echo "--- Worker Logs (last 3 lines each) ---"
    gssh "cd ~/experiment && for f in Results/worker_*.log; do [ -f \"\$f\" ] && echo \"=== \$f ===\" && tail -3 \"\$f\" && echo ''; done" || echo "  No worker logs yet"

    echo ""
    echo "--- Master DB ---"
    gssh "cd ~/experiment && [ -f Results/Master_Experiment_DB.csv ] && wc -l Results/Master_Experiment_DB.csv || echo 'Not created yet'"
}

# ============================================================
# COMMAND: download
# ============================================================
do_download() {
    echo "Downloading results from VM..."
    mkdir -p "$PROJ_DIR/Results_GCP"
    gscp_down "~/experiment/Results/*" "$PROJ_DIR/Results_GCP/"
    echo ""
    echo "Results saved to: $PROJ_DIR/Results_GCP/"
    echo "  Don't forget: ./gcp_deploy.sh destroy"
}

# ============================================================
# COMMAND: destroy
# ============================================================
do_destroy() {
    echo "WARNING: This will permanently delete the VM and all data on it."
    echo "Make sure you've downloaded results first!"
    read -p "Type 'yes' to confirm: " confirm
    if [ "$confirm" = "yes" ]; then
        gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
        echo "VM deleted. Billing stopped."
    else
        echo "Cancelled."
    fi
}

# ============================================================
# MAIN DISPATCH
# ============================================================
case "${1:-setup}" in
    setup)   do_setup ;;
    run)     do_run ;;
    status)  do_status ;;
    download) do_download ;;
    destroy) do_destroy ;;
    *)
        echo "Usage: $0 {setup|run|status|download|destroy}"
        exit 1
        ;;
esac
