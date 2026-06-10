#!/usr/bin/env bash
# Queue a sample of new-vs-old SLEAP vs DANNCE comparison videos:
# one session per rat x recent calibration regime (2025_11_13, 2025_12_08,
# 2026_02_06). Runs sequentially so the GPU and SMB bandwidth aren't shared.
#
# Each run logs to logs/compare_<rat>_<session>.log and the script continues
# even if one run fails (set -e is intentionally NOT used).
set -o pipefail

source /home/yutaka-sprague/anaconda3/etc/profile.d/conda.sh
conda activate clirb_analysis

cd /home/yutaka-sprague/CLIRB_analyses

NEW_MODEL=/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/models/260603_134002.single_instance.n=2298.og
# OLD_MODEL left as the script default (250731_105225...n=10383.og)

START_FRAME=5000
N_FRAMES=1000
CAMERA=0
BATCH=10
CHUNK=150

LOG_DIR=logs
mkdir -p "$LOG_DIR"

# rat session pairs: one per (rat x calib) cell, chosen 2026-06-04.
PAIRS=(
  "R1 2025_11_17_1"   # calib 2025_11_13
  "R1 2025_12_08_1"   # calib 2025_12_08
  "R1 2026_02_06_1"   # calib 2026_02_06
  "R2 2025_11_17_1"   # calib 2025_11_13
  "R2 2025_12_08_1"   # calib 2025_12_08
  "R2 2026_02_06_1"   # calib 2026_02_06
  "R3 2025_11_17_1"   # calib 2025_11_13
  "R3 2025_12_07_2"   # calib 2025_12_08
  "R3 2026_02_06_1"   # calib 2026_02_06
)

echo "=== render_compare_sample: ${#PAIRS[@]} runs queued at $(date) ==="
i=0
for pair in "${PAIRS[@]}"; do
  set -- $pair
  rat=$1; session=$2
  i=$((i + 1))
  log="$LOG_DIR/compare_${rat}_${session}.log"
  echo ""
  echo "=== [$i/${#PAIRS[@]}] $rat/$session -> $log  ($(date +%H:%M:%S)) ==="
  python -m corrector.compare_models_video \
      --new_model "$NEW_MODEL" \
      --rat "$rat" --session "$session" \
      --camera "$CAMERA" --start_frame "$START_FRAME" --n_frames "$N_FRAMES" \
      --batch "$BATCH" --chunk_frames "$CHUNK" \
      > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "    OK  ($(grep -h 'saved ' "$log" | tail -1))"
  else
    echo "    FAILED rc=$rc  (see $log)"
  fi
done

echo ""
echo "=== render_compare_sample: done at $(date) ==="
echo "=== residual summaries (OVERALL line per session) ==="
for pair in "${PAIRS[@]}"; do
  set -- $pair
  log="$LOG_DIR/compare_${1}_${2}.log"
  ov=$(grep -h "OVERALL" "$log" 2>/dev/null | tail -1)
  imp=$(grep -h "improved by new model" "$log" 2>/dev/null | tail -1)
  echo "$1/$2:  $ov   |  $imp"
done
