#!/usr/bin/env bash
# Render the 6 v2 comparison sessions with the v3_reg checkpoint.
set -euo pipefail
PY=/home/yutaka-sprague/anaconda3/envs/clirb_analysis/bin/python
CK=corrector/checkpoints/R1R2R3_2d_v3_reg.pt
cd /home/yutaka-sprague/CLIRB_analyses

$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R1 --session 2026_02_06_1 --camera 0 --n_frames 1000
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R1 --session 2026_02_13_1 --camera 0 --n_frames 1000
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R2 --session 2026_02_05_2 --camera 0 --n_frames 1000
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R2 --session 2026_02_13_1 --camera 0 --n_frames 1000
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R3 --session 2026_02_06_1 --camera 0 --n_frames 1000
$PY -u -m corrector.render_world_overlay --ckpt "$CK" --rat R3 --session 2026_02_18_2 --camera 0 --n_frames 1000
