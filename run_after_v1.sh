#!/bin/bash
# Wait for all v1 runs to complete, then start v2 and v3
cd /home/yutaka-sprague/CLIRB_analyses

echo "[$(date)] Waiting for v1 runs to complete..."
while pgrep -f "run_experiments.py --rat" | grep -v "v2\|v3" | grep -q .; do
    sleep 60
done

echo "[$(date)] V1 runs complete. Starting v2 and v3..."

# V2 runs
python experiments/run_experiments_v2.py --rat R1 --config primary > logs/v2_R1_primary.log 2>&1 &
python experiments/run_experiments_v2.py --rat R2 --config primary > logs/v2_R2_primary.log 2>&1 &
echo "[$(date)] V2 runs started (PIDs: $!)"

# V3 runs
python experiments/run_experiments_v3.py --rat R1 --config primary > logs/v3_R1_primary.log 2>&1 &
python experiments/run_experiments_v3.py --rat R1 --config secondary > logs/v3_R1_secondary.log 2>&1 &
python experiments/run_experiments_v3.py --rat R2 --config primary > logs/v3_R2_primary.log 2>&1 &
python experiments/run_experiments_v3.py --rat R3 --config primary > logs/v3_R3_primary.log 2>&1 &
echo "[$(date)] V3 runs started"

echo "[$(date)] All follow-up experiments launched."
