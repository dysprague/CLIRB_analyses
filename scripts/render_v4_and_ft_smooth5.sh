#!/usr/bin/env bash
# Same as render_v4_and_ft.sh but with --smooth_size 5 (vs 15).
# Output subdirs use _smooth5_ in place of _smooth15_.
set -euo pipefail
PY=/home/yutaka-sprague/anaconda3/envs/clirb_analysis/bin/python
cd /home/yutaka-sprague/CLIRB_analyses

ALL_SESSIONS=("R1 2026_02_09_1" "R1 2026_02_18_2" \
              "R2 2026_02_09_1" "R2 2026_02_18_2" \
              "R3 2026_02_10_1" "R3 2026_02_18_2")
R1_SESSIONS=("R1 2026_02_09_1" "R1 2026_02_18_2")
R2_SESSIONS=("R2 2026_02_09_1" "R2 2026_02_18_2")
R3_SESSIONS=("R3 2026_02_10_1" "R3 2026_02_18_2")

run_set() {
    local ckpt=$1; local sub_sp=$2; local sub_trip=$3; shift 3
    for entry in "$@"; do
        rat=${entry% *}; sess=${entry#* }
        $PY -u -m corrector.render_world_overlay --ckpt "$ckpt" --rat "$rat" \
            --session "$sess" --camera 0 --n_frames 1000 \
            --single_panel --smooth_size 5 --out_subdir "$sub_sp"
    done
    for entry in "$@"; do
        rat=${entry% *}; sess=${entry#* }
        $PY -u -m corrector.render_world_3d_triptych --ckpt "$ckpt" --rat "$rat" \
            --session "$sess" --n_frames 1000 \
            --smooth_size 5 --out_subdir "$sub_trip"
    done
}

# v4 global, smooth=5, all 6 sessions
run_set corrector/checkpoints/R1R2R3_temporal_mlp_2d_reproj_v4.pt \
    R1R2R3_temporal_mlp_2d_reproj_v4_smooth5_singlepanel \
    R1R2R3_temporal_mlp_2d_reproj_v4_smooth5_3dtriptych \
    "${ALL_SESSIONS[@]}"

# R1 fine-tune, 2 R1 sessions
run_set corrector/checkpoints/R1_reproj_ft_from_v3.pt \
    R1_reproj_ft_smooth5_singlepanel \
    R1_reproj_ft_smooth5_3dtriptych \
    "${R1_SESSIONS[@]}"

# R2 fine-tune, 2 R2 sessions
run_set corrector/checkpoints/R2_reproj_ft_from_v3.pt \
    R2_reproj_ft_smooth5_singlepanel \
    R2_reproj_ft_smooth5_3dtriptych \
    "${R2_SESSIONS[@]}"

# R3 fine-tune, 2 R3 sessions
run_set corrector/checkpoints/R3_reproj_ft_from_v3.pt \
    R3_reproj_ft_smooth5_singlepanel \
    R3_reproj_ft_smooth5_3dtriptych \
    "${R3_SESSIONS[@]}"
