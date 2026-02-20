import time
import sys
import os
import argparse
import subprocess
from datetime import datetime
from run_experiment import run_batch
import helpers


def parse_runs(runs_str):
    """Parse run specification: '0-80', '0,1,5,10', 'all', or '0-26,54-80'."""
    if runs_str.lower() == 'all':
        return list(range(81))

    ids = []
    for part in runs_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def split_runs(run_ids, num_workers):
    """Distribute run IDs across workers as evenly as possible."""
    chunks = [[] for _ in range(num_workers)]
    for i, rid in enumerate(run_ids):
        chunks[i % num_workers].append(rid)
    return [c for c in chunks if c]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gurobi inventory experiments")
    parser.add_argument("--runs", type=str, default="all",
                        help="Which runs: 'all', '0-80', '0,1,5,10', or '0-26,54-80'")
    parser.add_argument("--threads", type=int, default=0,
                        help="Gurobi threads per solve (0 = auto)")
    parser.add_argument("--gap", type=float, default=0.005,
                        help="MIP gap target (default 0.005 = 0.5%%)")
    parser.add_argument("--timelimit", type=int, default=2000,
                        help="Solver time limit per solve in seconds")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (each gets a subset of runs)")

    args = parser.parse_args()
    run_ids = parse_runs(args.runs)

    if args.workers > 1:
        chunks = split_runs(run_ids, args.workers)
        threads_per_worker = max(1, args.threads // args.workers) if args.threads > 0 else 0

        print(f"Launching {len(chunks)} parallel workers for {len(run_ids)} runs")
        print(f"  Threads per worker: {threads_per_worker if threads_per_worker > 0 else 'auto'}")
        print(f"  MIP Gap: {args.gap*100:.1f}%  |  Time Limit: {args.timelimit}s")

        procs = []
        for i, chunk in enumerate(chunks):
            chunk_str = ",".join(str(r) for r in chunk)
            cmd = [
                sys.executable, __file__,
                "--runs", chunk_str,
                "--threads", str(threads_per_worker),
                "--gap", str(args.gap),
                "--timelimit", str(args.timelimit),
                "--workers", "1",
            ]
            log_file = f"Results/worker_{i}.log"
            os.makedirs("Results", exist_ok=True)
            print(f"  Worker {i}: runs {chunk} -> {log_file}")
            fh = open(log_file, "w")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            procs.append((p, fh, i, chunk))

        print(f"\nAll workers launched. Waiting for completion...")
        start = time.time()
        for p, fh, i, chunk in procs:
            p.wait()
            fh.close()
            status = "OK" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
            print(f"  Worker {i} finished: {status} (runs {chunk})")

        elapsed = (time.time() - start) / 60.0
        print(f"\nAll workers done in {elapsed:.1f} minutes.")
        sys.exit(0)

    os.makedirs("Results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log_file = f"Results/Master_Log_{timestamp}.txt"
    sys.stdout = helpers.Logger(master_log_file)

    print(f" Master Log Started: {master_log_file}")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Runs: {run_ids}")
    print(f"   Threads: {args.threads if args.threads > 0 else 'auto'}")
    print(f"   MIP Gap: {args.gap*100:.1f}%  |  Time Limit: {args.timelimit}s")
    print("=" * 60)

    start_time = time.time()

    run_batch(
        run_ids=run_ids,
        solver_config={
            'threads': args.threads,
            'mip_gap': args.gap,
            'time_limit': args.timelimit,
        }
    )

    elapsed = (time.time() - start_time) / 60.0
    print(f"\nJob Finished in {elapsed:.2f} minutes.")
    print(f"   Master log saved to: {master_log_file}")

    sys.stdout = sys.__stdout__
    print(f"All runs complete. Master log: {master_log_file}")
