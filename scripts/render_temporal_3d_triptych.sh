#!/usr/bin/env bash
# Render 3D triptych (raw / corrected / DANNCE side-by-side) for the 6 sessions
# matching the single-panel renders. Uses temporal_mlp.
set -euo pipefail
PY=/home/yutaka-sprague/anaconda3/envs/clirb_analysis/bin/python
CK=corrector/checkpoints/R1R2R3_world_temporal_mlp.pt
SUB=R1R2R3_world_temporal_mlp_3dtriptych
cd /home/yutaka-sprague/CLIRB_analyses

$PY -u -m corrector.render_world_3d_triptych --ckpt "$CK" --rat R1 --session 2026_02_09_1 --n_frames 1000 --out_subdir "$SUB"
$PY -u -m corrector.render_world_3d_triptych --ckpt "$CK" --rat R1 --session 2026_02_18_2 --n_frames 1000 --out_subdir "$SUB"
$PY -u -m corrector.render_world_3d_triptych --ckpt "$CK" --rat R2 --session 2026_02_09_1 --n_frames 1000 --out_subdir "$SUB"
$PY -u -m corrector.render_world_3d_triptych --ckpt "$CK" --rat R2 --session 2026_02_18_2 --n_frames 1000 --out_subdir "$SUB"
$PY -u -m corrector.render_world_3d_triptych --ckpt "$CK" --rat R3 --session 2026_02_10_1 --n_frames 1000 --out_subdir "$SUB"
$PY -u -m corrector.render_world_3d_triptych --ckpt "$CK" --rat R3 --session 2026_02_18_2 --n_frames 1000 --out_subdir "$SUB"
