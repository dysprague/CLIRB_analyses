"""Stage 20 -- rank all 23 keypoints by SLEAP-vs-DANNCE MSE in RAW 3D WORLD
SPACE, not the egocentric (SpineM-centered, heading-rotated) space stage19
used.

Motivation: stage19's per-keypoint distances were computed on
`ev.project_session_to_pcs`'s "rotated_med" output, which per-frame
translates every keypoint by -SpineM and rotates by a heading angle derived
from the pose itself. If SLEAP and DANNCE disagree on SpineM's position (or
on the heading-defining keypoints), that disagreement gets baked into EVERY
OTHER keypoint's "error" too -- SpineM itself is trivially 0 by construction
(it IS the origin), which is a confound, not a result.

Fix: align DANNCE's world frame onto SLEAP's ONCE PER SESSION via a rigid
(rotation + isotropic scale + translation) Procrustes fit -- the same
"session-level camera-rig calibration" convention already used everywhere
else in this project for a world-space SLEAP/DANNCE comparison (stage9b,
stage14a, stage15, corrector/evaluate_world.py): a sector-balanced pooled
sample of frames feeds an iteratively-reweighted (outlier-trimmed) Procrustes
fit, then that ONE fitted transform is applied to every frame of that
session. This removes only the rigid camera-to-camera calibration offset
between the two rigs, not any per-frame, per-keypoint pose information --
unlike egocentric normalization, it does not reference SpineM (or any other
single keypoint) at all.

Writes:
  results/spatial_confound/stage20_keypoint_mse_world.csv
  results/spatial_confound/stage20_session_fits.csv   (per-session fit diagnostics)
"""
import os, sys, glob
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import templates as tp
from clirb.spatial import dual as du
from clirb.spatial.geometry import ARENA, sector_index
from corrector.world_alignment import fit_procrustes
from config import NODE_IDX

OUT = 'results/spatial_confound'
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]
N_KP = len(KP_NAMES)
SECTOR_W = 30
N_PER_SECTOR = 300

# Guard against a whole-session gross misalignment (same rationale as
# stage19): the fit itself flags a bad session via a large post-fit residual
# on its own held-out sample. Calibrated empirically -- see stage20 log.
FIT_RESIDUAL_GUARD_MM = 60.0


def robust_proc(A, B, iters=5, keep=0.7):
    """Iteratively-reweighted Procrustes: fit, drop the worst 30% of points
    by residual, refit. A, B: (N, 23, 3). apply(A) ~ B when done -- so calling
    robust_proc(dannce_pts, sleap_pts) gives a fit whose apply() maps
    DANNCE -> SLEAP (the validated convention, see SPATIAL_CONFOUND_RESULTS.md
    on the robust_proc(tgt, X) argument-order gotcha)."""
    idx = np.arange(len(A)); f = None
    for _ in range(iters):
        f = fit_procrustes(A[idx], B[idx], try_z_flip=True)
        r = np.linalg.norm(f['apply'](A) - B, axis=-1).mean(axis=1)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 200:
            break
    return f


def dual_sessions(rat):
    out = []
    for f in sorted(glob.glob(f'data/cache/{rat}/*.npz')):
        session = os.path.basename(f)[:-4]
        raw = sc.get_raw_session(rat, session, want_dual=True)
        if raw is not None and raw.get('has_dual', False):
            out.append(session)
    return out


def sector_balanced_sample(com_xy, n_per_sector=N_PER_SECTOR, rng=None):
    rng = rng or np.random.default_rng(0)
    theta = ARENA.theta(com_xy)
    sec = sector_index(theta, SECTOR_W)
    samp = []
    for k in range(12):
        ii = np.where(sec == k)[0]
        if len(ii):
            samp.extend(rng.choice(ii, size=min(len(ii), n_per_sector), replace=False))
    return np.array(sorted(samp), dtype=int)


def session_result(rat, session, rng):
    """Returns (sq_or_None, fit_info_or_None). fit_info is None only when the
    session is unusable before any fit is attempted (no template/DANNCE/too
    short); sq is None (but fit_info populated) when the fit itself flags the
    session as excluded."""
    raw = sc.get_raw_session(rat, session, want_dual=True)
    tm = tp.load_template(rat, session)
    if tm is None:
        return None, None
    d3 = du.dannce_aligned(raw)
    if d3 is None:
        return None, None
    n = min(len(raw['triang_3d']), len(d3))
    if n < 10:
        return None, None
    sl = np.asarray(raw['triang_3d'][:n], dtype=np.float32)
    dn = np.asarray(d3[:n], dtype=np.float32)

    samp = sector_balanced_sample(sl.mean(axis=1)[:, :2], rng=rng)
    if len(samp) < 500:
        return None, None
    fit = robust_proc(dn[samp], sl[samp])  # apply(): DANNCE -> SLEAP frame

    dn_aligned = fit['apply'](dn)
    sq = ((sl - dn_aligned) ** 2).sum(axis=-1)  # (n, 23), mm^2
    overall_rmse = float(np.sqrt(sq.mean()))
    fit_info = dict(rat=rat, session=session, n_frames=n, n_fit_pts=len(samp),
                    scale=fit['s'], z_flipped=fit['z_flipped'],
                    fit_residual_mm=fit['residual'], overall_rmse_mm=overall_rmse)
    if fit['residual'] > FIT_RESIDUAL_GUARD_MM or overall_rmse > 2 * FIT_RESIDUAL_GUARD_MM:
        fit_info['excluded'] = True
        return None, fit_info
    fit_info['excluded'] = False
    return sq, fit_info


def main():
    rng = np.random.default_rng(0)
    sq_sum = {rat: np.zeros(N_KP, dtype=np.float64) for rat in ['R4', 'R5', 'R6']}
    n_frames = {rat: 0 for rat in ['R4', 'R5', 'R6']}
    n_sessions = {rat: 0 for rat in ['R4', 'R5', 'R6']}
    fit_rows = []
    sess_kp_rows = []

    for rat in ['R4', 'R5', 'R6']:
        sessions = dual_sessions(rat)
        print(f'{rat}: {len(sessions)} dual-modality sessions', flush=True)
        for i, session in enumerate(sessions):
            sq, fit_info = session_result(rat, session, rng)
            if fit_info is None:
                continue
            fit_rows.append(fit_info)
            if sq is None:
                print(f'  EXCLUDED {rat}/{session}: fit_residual={fit_info["fit_residual_mm"]:.1f}mm '
                      f'overall_rmse={fit_info["overall_rmse_mm"]:.1f}mm', flush=True)
                continue
            sq_sum[rat] += sq.sum(axis=0)
            n_frames[rat] += sq.shape[0]
            n_sessions[rat] += 1
            sess_kp_row = dict(rat=rat, session=session, n_frames=int(sq.shape[0]))
            sess_kp_row.update({f'sqsum_{name}': float(v) for name, v in zip(KP_NAMES, sq.sum(axis=0))})
            sess_kp_rows.append(sess_kp_row)
            if i % 20 == 0:
                print(f'  [{i+1}/{len(sessions)}] {session}: {sq.shape[0]} frames, '
                      f'fit_residual={fit_info["fit_residual_mm"]:.1f}mm, scale={fit_info["scale"]:.4f}',
                      flush=True)
        print(f'{rat}: done, {n_sessions[rat]} sessions, {n_frames[rat]} frames', flush=True)

    pd.DataFrame(fit_rows).to_csv(f'{OUT}/stage20_session_fits.csv', index=False)
    print(f'\nwrote {OUT}/stage20_session_fits.csv')
    pd.DataFrame(sess_kp_rows).to_csv(f'{OUT}/stage20_session_keypoint_sums.csv', index=False)
    print(f'wrote {OUT}/stage20_session_keypoint_sums.csv ({len(sess_kp_rows)} sessions, '
          f'per-keypoint sum-of-squared-error -- for stratifying by pipeline version later '
          f'without rerunning the Procrustes fits)')

    rows = []
    for rat in ['R4', 'R5', 'R6']:
        mse = sq_sum[rat] / max(n_frames[rat], 1)
        for k, name in enumerate(KP_NAMES):
            rows.append(dict(rat=rat, keypoint=name, mse_mm2=mse[k],
                             rmse_mm=np.sqrt(mse[k]), n_frames=n_frames[rat]))
    total_frames = sum(n_frames.values())
    pooled_mse = sum(sq_sum.values()) / max(total_frames, 1)
    for k, name in enumerate(KP_NAMES):
        rows.append(dict(rat='pooled', keypoint=name, mse_mm2=pooled_mse[k],
                         rmse_mm=np.sqrt(pooled_mse[k]), n_frames=total_frames))

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/stage20_keypoint_mse_world.csv', index=False)
    print(f'\nwrote {OUT}/stage20_keypoint_mse_world.csv')

    print(f'\nWorld-space pooled ranking, all 3 rats, {total_frames} frames (lowest = best agreement):')
    pooled = d[d.rat == 'pooled'].sort_values('rmse_mm')
    for rank, row in enumerate(pooled.itertuples(), 1):
        print(f'  {rank:2d}. {row.keypoint:10s}  RMSE={row.rmse_mm:6.2f} mm  MSE={row.mse_mm2:8.1f} mm^2')

    for rat in ['R4', 'R5', 'R6']:
        print(f'\n{rat} ranking ({n_frames[rat]} frames, {n_sessions[rat]} sessions):')
        g = d[d.rat == rat].sort_values('rmse_mm')
        for rank, row in enumerate(g.itertuples(), 1):
            print(f'  {rank:2d}. {row.keypoint:10s}  RMSE={row.rmse_mm:6.2f} mm')


if __name__ == '__main__':
    main()
