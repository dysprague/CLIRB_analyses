#!/usr/bin/env bash
# Render temporal_mlp output as single-panel overlays (SLEAP raw cyan + corrected green).
# Mix of best/weak sessions per rat by kp_mse improvement.
set -euo pipefail
PY=/home/yutaka-sprague/anaconda3/envs/clirb_analysis/bin/python
CK=corrector/checkpoints/R1R2R3_world_temporal_mlp.pt
SUB=R1R2R3_world_temporal_mlp_singlepanel
cd /home/yutaka-sprague/CLIRB_analyses

# R1: big win and weak gain
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R1 --session 2026_02_09_1 --camera 0 --n_frames 1000 --single_panel --out_subdir "$SUB"
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R1 --session 2026_02_18_2 --camera 0 --n_frames 1000 --single_panel --out_subdir "$SUB"

# R2: biggest win and weak gain
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R2 --session 2026_02_09_1 --camera 0 --n_frames 1000 --single_panel --out_subdir "$SUB"
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R2 --session 2026_02_18_2 --camera 0 --n_frames 1000 --single_panel --out_subdir "$SUB"

# R3: biggest win and weak gain
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R3 --session 2026_02_10_1 --camera 0 --n_frames 1000 --single_panel --out_subdir "$SUB"
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R3 --session 2026_02_18_2 --camera 0 --n_frames 1000 --single_panel --out_subdir "$SUB"
