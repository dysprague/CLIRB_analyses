"""Stage 19 -- rank all 23 keypoints by SLEAP-vs-DANNCE tracking MSE.

Per dual-modality session: project both modalities through the SAME pipeline
the online rule (and every feature-space script in this investigation) uses --
z-flip (SLEAP only) -> normalize_skeleton (egocentric, per-frame translate +
yaw-rotate) -> 5-frame causal median (ev.project_session_to_pcs's 'rotated_med'
output) -- then take the per-keypoint squared Euclidean distance between the
two egocentric keypoint arrays, frame by frame. This is the same "rot" array
axial/snout_z/bodyrot are built from in stage17-18, so a keypoint's MSE here
is directly comparable to (and explains) those results.

Pooled over every frame of every dual-modality session (no rear-conditioning),
per rat and pooled across rats.

Writes:
  results/spatial_confound/stage19_keypoint_mse.csv   (per rat + pooled, per keypoint)
  results/figures/spatial_confound/F20_keypoint_mse_ranking.png
"""
import os, sys, glob
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial import dual as du
from config import NODE_IDX

OUT = 'results/spatial_confound'
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]
N_KP = len(KP_NAMES)


def dual_sessions(rat):
    out = []
    for f in sorted(glob.glob(f'data/cache/{rat}/*.npz')):
        session = os.path.basename(f)[:-4]
        raw = sc.get_raw_session(rat, session, want_dual=True)
        if raw is not None and raw.get('has_dual', False):
            out.append(session)
    return out


# Any real session's overall SLEAP-DANNCE RMSE is 8-32mm (checked across all
# 172 dual sessions). A whole-session gross misalignment (corrupt/truncated
# recording, bad frame sync) blows this up by 2-3 orders of magnitude -- guard
# against that polluting the pooled mean, which is quadratically sensitive to
# exactly this kind of outlier.
SESSION_RMSE_GUARD_MM = 100.0


def session_sq_err(rat, session):
    """(n_frames, 23) per-keypoint squared Euclidean distance (mm^2) between
    the egocentric SLEAP and DANNCE keypoint arrays, or None."""
    raw = sc.get_raw_session(rat, session, want_dual=True)
    tm = tp.load_template(rat, session)
    if tm is None:
        return None
    d3 = du.dannce_aligned(raw)
    if d3 is None:
        return None
    n = min(len(raw['triang_3d']), len(d3))
    if n < 10:
        return None
    _, s_rot = ev.project_session_to_pcs(raw['triang_3d'][:n], tm, flip_z=True)
    _, d_rot = ev.project_session_to_pcs(d3[:n], tm, flip_z=False)
    diff = s_rot[:, :N_KP, :] - d_rot[:, :N_KP, :]
    sq = (diff ** 2).sum(axis=-1)  # (n, 23)
    overall_rmse = np.sqrt(sq.mean())
    if overall_rmse > SESSION_RMSE_GUARD_MM:
        print(f'  EXCLUDED {rat}/{session}: whole-session RMSE={overall_rmse:.0f}mm '
              f'(n={n} frames) -- gross misalignment, not real tracking error', flush=True)
        return None
    return sq


def main():
    sq_sum = {rat: np.zeros(N_KP, dtype=np.float64) for rat in ['R4', 'R5', 'R6']}
    n_frames = {rat: 0 for rat in ['R4', 'R5', 'R6']}
    n_sessions = {rat: 0 for rat in ['R4', 'R5', 'R6']}

    for rat in ['R4', 'R5', 'R6']:
        sessions = dual_sessions(rat)
        print(f'{rat}: {len(sessions)} dual-modality sessions', flush=True)
        for i, session in enumerate(sessions):
            se = session_sq_err(rat, session)
            if se is None:
                continue
            sq_sum[rat] += se.sum(axis=0)
            n_frames[rat] += se.shape[0]
            n_sessions[rat] += 1
            if i % 20 == 0:
                print(f'  [{i+1}/{len(sessions)}] {session}: {se.shape[0]} frames', flush=True)
        print(f'{rat}: done, {n_sessions[rat]} sessions, {n_frames[rat]} frames', flush=True)

    rows = []
    for rat in ['R4', 'R5', 'R6']:
        mse = sq_sum[rat] / max(n_frames[rat], 1)
        for k, name in enumerate(KP_NAMES):
            rows.append(dict(rat=rat, keypoint=name, mse_mm2=mse[k],
                             rmse_mm=np.sqrt(mse[k]), n_frames=n_frames[rat],
                             is_egocentric_origin=(name == 'SpineM')))
    total_frames = sum(n_frames.values())
    pooled_sum = sum(sq_sum.values())
    pooled_mse = pooled_sum / max(total_frames, 1)
    for k, name in enumerate(KP_NAMES):
        rows.append(dict(rat='pooled', keypoint=name, mse_mm2=pooled_mse[k],
                         rmse_mm=np.sqrt(pooled_mse[k]), n_frames=total_frames,
                         is_egocentric_origin=(name == 'SpineM')))

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/stage19_keypoint_mse.csv', index=False)
    print(f'\nwrote {OUT}/stage19_keypoint_mse.csv')
    print('(SpineM is the egocentric-normalization origin -- RMSE=0 by construction, not real '
          'tracking agreement; excluded from the rankings below, kept in the CSV for completeness.)')

    print(f'\nPooled ranking, all 3 rats, {total_frames} frames (lowest = best agreement):')
    pooled = d[(d.rat == 'pooled') & ~d.is_egocentric_origin].sort_values('rmse_mm')
    for rank, row in enumerate(pooled.itertuples(), 1):
        print(f'  {rank:2d}. {row.keypoint:10s}  RMSE={row.rmse_mm:6.2f} mm  MSE={row.mse_mm2:8.1f} mm^2')

    for rat in ['R4', 'R5', 'R6']:
        print(f'\n{rat} ranking ({n_frames[rat]} frames, {n_sessions[rat]} sessions):')
        g = d[(d.rat == rat) & ~d.is_egocentric_origin].sort_values('rmse_mm')
        for rank, row in enumerate(g.itertuples(), 1):
            print(f'  {rank:2d}. {row.keypoint:10s}  RMSE={row.rmse_mm:6.2f} mm')


if __name__ == '__main__':
    main()
