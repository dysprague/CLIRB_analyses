"""Stage 21 -- reliability of 16 candidate interpretable online-matching
features, measured as SLEAP-vs-DANNCE agreement in the SAME Procrustes-
aligned raw 3D world space stage20/§8q established (no SpineM/egocentric
confound -- see SPATIAL_CONFOUND_RESULTS.md §8q/§8r before re-deriving this
from scratch).

Per dual-modality session:
  1. Fit ONE session-level rigid+scale Procrustes transform (sector-balanced
     sample, outlier-trimmed) mapping DANNCE -> SLEAP's world frame -- exactly
     stage20's method and guard (exclude if fit residual/overall RMSE is a
     gross-misalignment outlier).
  2. Apply the SAME 5-frame causal median filter the online rule itself uses
     (`ev.causal_median_5_kp`, warmup='online') to both SLEAP's own keypoints
     and DANNCE's now-aligned keypoints, independently -- so these candidate
     features are evaluated exactly as the online rule would actually see
     them (temporal smoothing is a real part of the online pipeline, and is
     unrelated to the SpineM/egocentric confound the world-space redo fixed).
  3. Compute each of the 7 features from each modality's filtered keypoints,
     pool the per-frame SLEAP-vs-DANNCE agreement over every frame of every
     session (no rear-conditioning).

Features:
  snout_z          : Snout Z height (world space)
  ear_snout_mean_z : mean Z of {Snout, EarL, EarR}
  ear_mean_z       : mean Z of {EarL, EarR} (no snout)
  angle_lower_xy   : signed XY angle, (SpineM-TailBase) -> (SpineF-SpineM)
  angle_upper_xy   : signed XY angle, (SpineF-SpineM) -> (Snout-SpineF)
  angle_mid_xy     : signed XY angle, (SpineM-SpineL) -> (SpineF-SpineM)
  angle_skip_xy    : signed XY angle, (SpineF-SpineL) -> (Snout-SpineF)
  angle_torsion_xy : signed XY angle, (HipR-HipL) -> (ShoulderR-ShoulderL) -- body twist
  angle_body_head_xy : signed XY angle, (SpineF-TailBase) -> (mean(EarL,EarR)-SpineF) -- turn
  angle_mid_head_xy : signed XY angle, (SpineM-SpineL) -> (mean(EarL,EarR)-SpineM) -- "mid" that
                     skips SpineF and goes straight to the ears
  angle_torsion_ear_xy : signed XY angle, (HipR-HipL) -> (EarR-EarL) -- torsion using the ear
                     axis instead of the shoulder axis
  dist_snout_tb    : 3D distance, Snout <-> TailBase
  dist_ears_tb     : 3D distance, mean(EarL,EarR) <-> TailBase
  dist_spineF_tb   : 3D distance, SpineF <-> TailBase -- core torso length (no snout)
  dist_earmean_hipmean : 3D distance, mean(EarL,EarR) <-> mean(HipL,HipR) -- head-to-hip span
  com_velocity     : 3D velocity vector of the 23-keypoint center of mass (mm/s)

For the eight angle features, also report the fraction of frames where SLEAP's
and DANNCE's signed angle AGREE IN SIGN (both curling the same direction).

Writes:
  results/spatial_confound/stage21_feature_reliability.csv
  results/spatial_confound/stage21_session_fits.csv
"""
import os, sys, glob
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb import features as feat
from clirb.spatial import dual as du
from clirb.spatial.geometry import ARENA, sector_index
from corrector.world_alignment import fit_procrustes
from config import NODE_IDX

OUT = 'results/spatial_confound'
SECTOR_W = 30
N_PER_SECTOR = 300
FIT_RESIDUAL_GUARD_MM = 60.0
FPS = ev.FPS

SNOUT, EARL, EARR = NODE_IDX['Snout'], NODE_IDX['EarL'], NODE_IDX['EarR']
SPINEF, SPINEM, SPINEL, TAILBASE = (NODE_IDX['SpineF'], NODE_IDX['SpineM'],
                                    NODE_IDX['SpineL'], NODE_IDX['TailBase'])
HIPL, HIPR, SHOULDERL, SHOULDERR = (NODE_IDX['HipL'], NODE_IDX['HipR'],
                                    NODE_IDX['ShoulderL'], NODE_IDX['ShoulderR'])


def robust_proc(A, B, iters=5, keep=0.7):
    """apply(A) ~ B when done. robust_proc(dannce_pts, sleap_pts) -> apply()
    maps DANNCE -> SLEAP (validated convention, see SPATIAL_CONFOUND_RESULTS.md)."""
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
    sec = sector_index(ARENA.theta(com_xy), SECTOR_W)
    samp = []
    for k in range(12):
        ii = np.where(sec == k)[0]
        if len(ii):
            samp.extend(rng.choice(ii, size=min(len(ii), n_per_sector), replace=False))
    return np.array(sorted(samp), dtype=int)


signed_angle_xy = feat.signed_angle_xy


def wrap_deg(diff):
    return (diff + 180.0) % 360.0 - 180.0


# Feature definitions live in clirb.features (the single source of truth, §8s-§8z).
# This script's reliability analysis consumes velocity as a 3-vector, not a scalar.
def compute_features(kp):
    return feat.compute_features(kp, velocity='vector')


ANGLE_FEATURES = {'angle_lower_xy', 'angle_upper_xy', 'angle_mid_xy', 'angle_skip_xy',
                  'angle_torsion_xy', 'angle_body_head_xy', 'angle_mid_head_xy',
                  'angle_torsion_ear_xy'}
VECTOR_FEATURES = {'com_velocity'}
UNITS = dict(snout_z='mm', ear_snout_mean_z='mm', ear_mean_z='mm',
            angle_lower_xy='deg', angle_upper_xy='deg', angle_mid_xy='deg', angle_skip_xy='deg',
            angle_torsion_xy='deg', angle_body_head_xy='deg', angle_mid_head_xy='deg',
            angle_torsion_ear_xy='deg',
            dist_snout_tb='mm', dist_ears_tb='mm', dist_spineF_tb='mm', dist_earmean_hipmean='mm',
            com_velocity='mm/s')


def session_result(rat, session, rng):
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
    sl_raw = np.asarray(raw['triang_3d'][:n], dtype=np.float32)
    dn_raw = np.asarray(d3[:n], dtype=np.float32)

    samp = sector_balanced_sample(sl_raw.mean(axis=1)[:, :2], rng=rng)
    if len(samp) < 500:
        return None, None
    fit = robust_proc(dn_raw[samp], sl_raw[samp])
    dn_aligned = fit['apply'](dn_raw)
    overall_rmse = float(np.sqrt(((sl_raw - dn_aligned) ** 2).sum(axis=-1).mean()))
    fit_info = dict(rat=rat, session=session, n_frames=n, scale=fit['s'],
                    fit_residual_mm=fit['residual'], overall_rmse_mm=overall_rmse)
    if fit['residual'] > FIT_RESIDUAL_GUARD_MM or overall_rmse > 2 * FIT_RESIDUAL_GUARD_MM:
        fit_info['excluded'] = True
        return None, fit_info
    fit_info['excluded'] = False

    sl = ev.causal_median_5_kp(sl_raw, warmup='online')
    dn = ev.causal_median_5_kp(dn_aligned, warmup='online')
    f_sl = compute_features(sl)
    f_dn = compute_features(dn)
    return (f_sl, f_dn), fit_info


def main():
    rng = np.random.default_rng(0)
    feat_names = list(UNITS)
    acc = {rat: {f: dict(sqsum=0.0, n=0, sign_agree=0) for f in feat_names} for rat in ['R4', 'R5', 'R6']}
    fit_rows = []

    for rat in ['R4', 'R5', 'R6']:
        sessions = dual_sessions(rat)
        print(f'{rat}: {len(sessions)} dual-modality sessions', flush=True)
        for i, session in enumerate(sessions):
            out, fit_info = session_result(rat, session, rng)
            if fit_info is None:
                continue
            fit_rows.append(fit_info)
            if out is None:
                print(f'  EXCLUDED {rat}/{session}: fit_residual={fit_info["fit_residual_mm"]:.1f}mm '
                      f'overall_rmse={fit_info["overall_rmse_mm"]:.1f}mm', flush=True)
                continue
            f_sl, f_dn = out
            for feat in feat_names:
                a, b = f_sl[feat], f_dn[feat]
                if feat in VECTOR_FEATURES:
                    sq = ((a - b) ** 2).sum(axis=-1)
                    acc[rat][feat]['sqsum'] += float(sq.sum())
                    acc[rat][feat]['n'] += sq.shape[0]
                elif feat in ANGLE_FEATURES:
                    diff = wrap_deg(a - b)
                    acc[rat][feat]['sqsum'] += float((diff ** 2).sum())
                    acc[rat][feat]['n'] += diff.shape[0]
                    acc[rat][feat]['sign_agree'] += int((np.sign(a) == np.sign(b)).sum())
                else:
                    diff = a - b
                    acc[rat][feat]['sqsum'] += float((diff ** 2).sum())
                    acc[rat][feat]['n'] += diff.shape[0]
            if i % 20 == 0:
                print(f'  [{i+1}/{len(sessions)}] {session}: {f_sl["snout_z"].shape[0]} frames', flush=True)
        print(f'{rat}: done', flush=True)

    pd.DataFrame(fit_rows).to_csv(f'{OUT}/stage21_session_fits.csv', index=False)
    print(f'\nwrote {OUT}/stage21_session_fits.csv')

    rows = []
    for rat in ['R4', 'R5', 'R6']:
        for feat in feat_names:
            a = acc[rat][feat]
            rmse = np.sqrt(a['sqsum'] / max(a['n'], 1))
            row = dict(rat=rat, feature=feat, unit=UNITS[feat], rmse=rmse, n=a['n'])
            if feat in ANGLE_FEATURES:
                row['sign_agree_pct'] = 100.0 * a['sign_agree'] / max(a['n'], 1)
            rows.append(row)
    for feat in feat_names:
        sqsum = sum(acc[rat][feat]['sqsum'] for rat in acc)
        n = sum(acc[rat][feat]['n'] for rat in acc)
        rmse = np.sqrt(sqsum / max(n, 1))
        row = dict(rat='pooled', feature=feat, unit=UNITS[feat], rmse=rmse, n=n)
        if feat in ANGLE_FEATURES:
            sa = sum(acc[rat][feat]['sign_agree'] for rat in acc)
            row['sign_agree_pct'] = 100.0 * sa / max(n, 1)
        rows.append(row)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/stage21_feature_reliability.csv', index=False)
    print(f'\nwrote {OUT}/stage21_feature_reliability.csv')

    print('\n=== Pooled (all 3 rats) ===')
    for row in d[d.rat == 'pooled'].itertuples():
        extra = f'   sign-agreement={row.sign_agree_pct:.1f}%' if hasattr(row, 'sign_agree_pct') and not np.isnan(row.sign_agree_pct) else ''
        print(f'  {row.feature:18s}  RMSE={row.rmse:8.3f} {row.unit:5s}  n={row.n}{extra}')

    for rat in ['R4', 'R5', 'R6']:
        print(f'\n=== {rat} ===')
        for row in d[d.rat == rat].itertuples():
            extra = f'   sign-agreement={row.sign_agree_pct:.1f}%' if hasattr(row, 'sign_agree_pct') and not np.isnan(row.sign_agree_pct) else ''
            print(f'  {row.feature:18s}  RMSE={row.rmse:8.3f} {row.unit:5s}{extra}')


if __name__ == '__main__':
    main()
