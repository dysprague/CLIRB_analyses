#!/usr/bin/env bash
set -euo pipefail
PY=/home/yutaka-sprague/anaconda3/envs/clirb_analysis/bin/python
CK=corrector/checkpoints/R1R2R3_temporal_mlp_2d_reproj_v3.pt
SP_SUB=R1R2R3_temporal_mlp_2d_reproj_v3_singlepanel
TRIP_SUB=R1R2R3_temporal_mlp_2d_reproj_v3_3dtriptych
cd /home/yutaka-sprague/CLIRB_analyses

SESSIONS=("R1 2026_02_09_1" "R1 2026_02_18_2" \
          "R2 2026_02_09_1" "R2 2026_02_18_2" \
          "R3 2026_02_10_1" "R3 2026_02_18_2")

for entry in "${SESSIONS[@]}"; do
    rat=${entry% *}; sess=${entry#* }
    $PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat "$rat" \
        --session "$sess" --camera 0 --n_frames 1000 \
        --single_panel --out_subdir "$SP_SUB"
done

for entry in "${SESSIONS[@]}"; do
    rat=${entry% *}; sess=${entry#* }
    $PY -u -m corrector.render_world_3d_triptych --ckpt "$CK" --rat "$rat" \
        --session "$sess" --n_frames 1000 --out_subdir "$TRIP_SUB"
done
