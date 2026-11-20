"""R9 2026_10_28_1: adaptive-DANNCE passes vs conf^2-weighted, median-filtered SLEAP.

The water-port mechanism obstructs specific DANNCE cameras (REPORT appendix
"the size bias is the water-port mechanism").  `dannce_adaptive` attacks that
by re-running sDANNCE with the obstructed camera SLOTS fed by a different
camera's video plus the matching calibration swap, so the network still sees
six views but no obstructed one:

  base        the shipped 6-camera prediction (dannce/DANNCE/predict_results_finetune)
  c2d6        Camera2 slot <- Camera6              (pass_c2d6)
  c2d6_c5d4   Camera2 <- Camera6, Camera5 <- Camera4 (pass_c2d6_c5d4)

Reference is the conf^2-weighted SLEAP triangulation, 5-frame causal median --
the representation the online rule acts on.  SLEAP is not ground truth; what
this measures is whether the adaptive pass moves DANNCE TOWARD the independent
3-camera reconstruction.  Two things make that interpretable rather than
circular: SLEAP's cameras are a different physical set, and the direction of
the Procrustes SCALE is diagnostic on its own -- the port obstruction was shown
to shrink the DANNCE skeleton, so a pass that removes the obstruction should
move `s` toward 1 regardless of what SLEAP says.

Metrics, following the conventions in newcal_sleap_dannce_eval.py:
  A  world space after one robust rigid+isotropic-scale Procrustes per variant
     (sector-balanced sample), per keypoint and overall; plus the fitted scale
  B  egocentric space (normalize_skeleton_batch), which removes global pose and
     leaves posture only
  C  the two interpretable features, overall and split by distance to the port
  D  stratified by plan.npz's frame_pass -- the frames the planner actually
     wanted each pass for, versus the frames it was content with baseline
  E  DANNCE's own per-camera p_max, as a truth-free check that the swap
     improved the views it was meant to

Reads only from olveczky_lab.  Writes results/newcal/adaptive_dannce_r9_*.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
import scipy.io as sio

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, features as cf
from clirb.spatial import dual as du
from clirb.spatial import quality as ql
from skeleton import normalize_skeleton_batch
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample
from scripts.newcal_weighted_reanalysis import dlt_chunked, fill_gaps
from config import NODE_IDX, EDGES

DATA_ROOT = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
OUT = 'results/newcal'
RAT, SESSION = 'R9', '2026_10_28_1'
EXPONENT = 2.0
NEAR, FAR = 150.0, 300.0
TURN_DEG = 45.0
GROSS_FRAME_RMSE_MM = 40.0
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]

SESS = f'{DATA_ROOT}/{RAT}/{SESSION}'
VARIANTS = [
    ('base',      f'{SESS}/dannce/DANNCE/predict_results_finetune/save_data_AVG.mat'),
    ('c2d6',      f'{SESS}/dannce_adaptive/pass_c2d6/DANNCE/predict_results/save_data_AVG.mat'),
    ('c2d6_c5d4', f'{SESS}/dannce_adaptive/pass_c2d6_c5d4/DANNCE/predict_results/save_data_AVG.mat'),
]


def load_pred(path):
    """(T, 23, 3) world keypoints and (T, 23) p_max from a save_data_AVG.mat."""
    m = sio.loadmat(path, simplify_cells=True)
    pred = np.asarray(m['pred'], np.float64)          # (T, 3, 23)
    return pred.transpose(0, 2, 1), np.asarray(m['p_max'], np.float64)


def on_sleap_grid(d, idx, n):
    """Apply dannce_idx_for_sleap_cams the way clirb.spatial.dual does."""
    i = np.clip(idx, 0, len(d) - 1)
    return d[i][:n]


def weighted_sleap(raw, n):
    """conf^2-weighted DLT on processed frames, scattered + gap-filled to the
    acquisition grid.  Identical to newcal_feature_three_way.build()."""
    k2 = np.asarray(raw['sleap_keys_2D'])
    fm = np.asarray(raw['frame_mapping_cam0'], int)
    nproc = min(len(fm), len(k2))
    P, K, D, R, T = ql.load_calib(ql.calib_dir(RAT, SESSION))
    und = ql.undistort_all(k2[:nproc, :, :, :2].astype(np.float64), K, D)
    conf = np.clip(k2[:nproc, :, :, 2].astype(np.float64), 0, None)
    w = conf ** EXPONENT
    under = (w > 0).sum(axis=1) < 2
    w[np.broadcast_to(under[:, None, :], w.shape)] = 1.0
    kp = dlt_chunked(und, P, w)
    out = np.full((n, kp.shape[1], 3), np.nan, np.float32)
    fmp = fm[:nproc]
    ok = (fmp >= 0) & (fmp < n) & np.isfinite(kp).all(axis=(1, 2))[:len(fmp)]
    out[fmp[ok]] = kp[ok]
    known = np.zeros(n, bool); known[fmp[ok]] = True
    return fill_gaps(out, known), float(1 - known.mean())


def smoothed(arr):
    """5-frame causal median then z up-positive -- what the rule sees."""
    a = ev.causal_median_5_kp(np.asarray(arr, np.float32), warmup='online').copy()
    a[:, :, 2] *= -1.0
    return a


def stats(d, extra=None):
    rec = dict(rmse=float(np.sqrt(np.mean(d ** 2))),
               med_abs=float(np.median(np.abs(d))),
               p90_abs=float(np.percentile(np.abs(d), 90)),
               bias=float(np.mean(d)))
    if extra:
        rec.update(extra)
    return rec


def main():
    raw = sc.get_raw_session(RAT, SESSION, want_dual=True)
    assert raw is not None and raw.get('has_dual', False)

    idx = np.asarray(raw['dannce_idx_for_sleap_cams'], int).ravel()[1:]
    sl_eq = np.asarray(raw['triang_3d'], np.float32)
    n = min(len(sl_eq), len(idx))
    sl_eq = sl_eq[:n]; idx = idx[:n]
    sl_wt, interp = weighted_sleap(raw, n)
    print(f'{RAT}/{SESSION}: {n} frames on the SLEAP grid, '
          f'{100*interp:.1f}% interpolated in the weighted variant', flush=True)

    # sanity: the cached DANNCE should be the baseline prediction
    cached = du.dannce_aligned(raw)[:n]

    # --- strata ------------------------------------------------------------
    port = np.asarray(raw['port_xy']).ravel()[:2]
    com = np.median(sl_wt, axis=1)[:, :2]
    dist = np.hypot(com[:, 0] - port[0], com[:, 1] - port[1])
    plan = np.load(f'{SESS}/dannce_adaptive/plan.npz', allow_pickle=True)
    fp_full = np.asarray(plan['frame_pass'], int)
    pass_names = [str(x) for x in plan['pass_names']]
    fp = fp_full[np.clip(idx, 0, len(fp_full) - 1)]
    print('planner frame_pass over the whole recording: ' +
          ', '.join(f'{nm}={100*(fp_full==k).mean():.1f}%'
                    for k, nm in enumerate(pass_names)) +
          f', baseline={100*(fp_full<0).mean():.1f}%', flush=True)

    masks = {
        'all': np.ones(n, bool),
        'near port (<150mm)': dist < NEAR,
        'mid (150-300mm)': (dist >= NEAR) & (dist <= FAR),
        'far (>300mm)': dist > FAR,
        'planner: baseline ok': fp < 0,
        'planner: wanted a swap': fp >= 0,
    }

    sl_s = smoothed(sl_wt)
    sl_eq_s = smoothed(sl_eq)
    sl_ego = normalize_skeleton_batch(sl_s)[0]
    f_sl = cf.compute_features(sl_s)
    f_sl_eq = cf.compute_features(sl_eq_s)

    samp, _ = balanced_sample(com, rng=np.random.default_rng(0))
    print(f'Procrustes sample: {len(samp)} frames', flush=True)

    kp_rows, ov_rows, feat_rows, cam_rows = [], [], [], []
    dn_store = {}
    pmax_full = np.asarray(plan['pred_max'], np.float64)

    for tag, path in VARIANTS:
        pred, pmax = load_pred(path)
        dn = on_sleap_grid(pred, idx, n).astype(np.float32)
        if tag == 'base':
            agree = np.nanmax(np.abs(dn - cached))
            print(f'  cached dannce_keys_3D vs base predict_results: '
                  f'max |diff| = {agree:.4g} mm', flush=True)

        fit = robust_proc(dn[samp], sl_wt[samp])
        dn_w = fit['apply'](dn).astype(np.float32)
        dn_s = smoothed(dn_w)
        dn_ego = normalize_skeleton_batch(dn_s)[0]
        dn_store[tag] = dn_s
        f_dn = cf.compute_features(dn_s)

        err = np.linalg.norm(dn_s - sl_s, axis=-1)            # (n, 23) mm
        ego = np.linalg.norm(dn_ego - sl_ego, axis=-1)
        frame_rmse = np.sqrt((err ** 2).mean(axis=1))

        for stratum, sel in masks.items():
            if sel.sum() < 500:
                continue
            ov_rows.append(dict(
                variant=tag, stratum=stratum, n_frames=int(sel.sum()),
                frac_of_session=float(sel.mean()),
                procrustes_scale=fit['s'], procrustes_residual_mm=fit['residual'],
                world_rmse_mm=float(np.sqrt((err[sel] ** 2).mean())),
                world_med_mm=float(np.median(err[sel])),
                ego_rmse_mm=float(np.sqrt((ego[sel] ** 2).mean())),
                ego_med_mm=float(np.median(ego[sel])),
                frame_rmse_med=float(np.median(frame_rmse[sel])),
                frame_rmse_p90=float(np.percentile(frame_rmse[sel], 90)),
                gross_frac=float((frame_rmse[sel] > GROSS_FRAME_RMSE_MM).mean()),
                dannce_peak_resp=float(np.nanmean(pmax[np.clip(idx, 0, len(pmax)-1)][sel])),
            ))

        for j, name in enumerate(KP_NAMES):
            kp_rows.append(dict(
                variant=tag, keypoint=name,
                world_rmse_mm=float(np.sqrt((err[:, j] ** 2).mean())),
                world_med_mm=float(np.median(err[:, j])),
                ego_rmse_mm=float(np.sqrt((ego[:, j] ** 2).mean())),
                near_rmse_mm=float(np.sqrt((err[masks['near port (<150mm)'], j] ** 2).mean())),
                far_rmse_mm=float(np.sqrt((err[masks['far (>300mm)'], j] ** 2).mean())),
            ))

        for feat in ['ear_mean_z', 'cum_heading_change']:
            A = np.asarray(f_sl[feat], float)
            Aeq = np.asarray(f_sl_eq[feat], float)
            B = np.asarray(f_dn[feat], float)
            for stratum, sel in masks.items():
                if sel.sum() < 500:
                    continue
                for ref, Rv in (('sleap_conf2', A), ('sleap_equal', Aeq)):
                    d = Rv[sel] - B[sel]
                    rec = dict(variant=tag, feature=feat, reference=ref,
                               stratum=stratum, n_frames=int(sel.sum()),
                               unit='mm' if feat == 'ear_mean_z' else 'deg',
                               **stats(d),
                               pearson_r=float(np.corrcoef(Rv[sel], B[sel])[0, 1]))
                    if feat == 'cum_heading_change':
                        m = np.abs(Rv[sel]) >= TURN_DEG
                        rec['frac_real_turn'] = float(m.mean())
                        rec['sign_agree_turn_pct'] = (
                            float(100 * (np.sign(Rv[sel]) == np.sign(B[sel]))[m].mean())
                            if m.sum() > 100 else np.nan)
                    feat_rows.append(rec)

        print(f'  {tag:10s} s={fit["s"]:.4f} resid={fit["residual"]:.2f}mm  '
              f'world RMSE {np.sqrt((err**2).mean()):.2f}mm  '
              f'ego {np.sqrt((ego**2).mean()):.2f}mm', flush=True)

    # --- F: body proportions -------------------------------------------------
    # The port obstruction was shown to FORESHORTEN the DANNCE skeleton along
    # its own axis while leaving it roughly the right width (REPORT appendix
    # "the size bias is the water-port mechanism").  A global Procrustes scale
    # cannot see that -- it is one isotropic number -- so measure the bones.
    def bone_lengths(a):
        return np.stack([np.linalg.norm(a[:, i] - a[:, j], axis=-1)
                         for i, j in EDGES], axis=1)       # (n, n_bones)

    SPAN = [('body axis SpineF-TailBase', 'SpineF', 'TailBase'),
            ('body axis SpineF-SpineL', 'SpineF', 'SpineL'),
            ('hip width HipL-HipR', 'HipL', 'HipR'),
            ('shoulder width', 'ShoulderL', 'ShoulderR'),
            ('head EarL-EarR', 'EarL', 'EarR'),
            ('snout-SpineF', 'Snout', 'SpineF')]
    bl_sl = bone_lengths(sl_s)
    bone_rows = []
    for tag in dn_store:
        bl = bone_lengths(dn_store[tag])
        for b, (i, j) in enumerate(EDGES):
            bone_rows.append(dict(
                variant=tag, bone=f'{KP_NAMES[i]}-{KP_NAMES[j]}',
                sleap_median_mm=float(np.median(bl_sl[:, b])),
                dannce_median_mm=float(np.median(bl[:, b])),
                ratio=float(np.median(bl[:, b]) / np.median(bl_sl[:, b])),
                dannce_cv=float(np.std(bl[:, b]) / np.mean(bl[:, b]))))
        for name, a, bb in SPAN:
            ia, ib = KP_NAMES.index(a), KP_NAMES.index(bb)
            ls = np.linalg.norm(sl_s[:, ia] - sl_s[:, ib], axis=-1)
            ld = np.linalg.norm(dn_store[tag][:, ia] - dn_store[tag][:, ib], axis=-1)
            bone_rows.append(dict(
                variant=tag, bone=f'[span] {name}',
                sleap_median_mm=float(np.median(ls)),
                dannce_median_mm=float(np.median(ld)),
                ratio=float(np.median(ld) / np.median(ls)),
                dannce_cv=float(np.std(ld) / np.mean(ld))))

    # --- E: per-camera DANNCE confidence ------------------------------------
    for c in range(pmax_full.shape[1]):
        v = pmax_full[:, c]
        cam_rows.append(dict(camera=f'Camera{c+1}', mean_pmax=float(np.nanmean(v)),
                             median_pmax=float(np.nanmedian(v)),
                             frac_below_thresh=float(np.nanmean(v < float(plan['threshold']))),
                             worst_frac=float(np.nanmean(
                                 np.argmin(pmax_full, axis=1) == c))))

    kp = pd.DataFrame(kp_rows); ov = pd.DataFrame(ov_rows)
    ft = pd.DataFrame(feat_rows); cam = pd.DataFrame(cam_rows)
    bone = pd.DataFrame(bone_rows)
    kp.to_csv(f'{OUT}/adaptive_dannce_r9_keypoints.csv', index=False)
    ov.to_csv(f'{OUT}/adaptive_dannce_r9_overall.csv', index=False)
    ft.to_csv(f'{OUT}/adaptive_dannce_r9_features.csv', index=False)
    cam.to_csv(f'{OUT}/adaptive_dannce_r9_cameras.csv', index=False)
    bone.to_csv(f'{OUT}/adaptive_dannce_r9_bones.csv', index=False)

    pd.set_option('display.width', 250)
    order = [t for t, _ in VARIANTS]
    print(f'\n{"="*96}\nE. DANNCE per-camera confidence over the whole recording'
          f'\n{"="*96}')
    print(cam.round(4).to_string(index=False))

    print(f'\n{"="*96}\nA/B. Full keypoint space vs conf^2 SLEAP\n{"="*96}')
    for stratum in masks:
        s_ = ov[ov.stratum == stratum]
        if not len(s_):
            continue
        s_ = s_.set_index('variant').reindex(order).reset_index()
        print(f'\n  [{stratum}]  {int(s_.n_frames.iloc[0])} frames '
              f'({100*s_.frac_of_session.iloc[0]:.1f}% of session)')
        print(s_[['variant', 'procrustes_scale', 'procrustes_residual_mm',
                  'world_rmse_mm', 'world_med_mm', 'ego_rmse_mm', 'ego_med_mm',
                  'frame_rmse_med', 'frame_rmse_p90', 'gross_frac',
                  'dannce_peak_resp']].round(4).to_string(index=False))

    print(f'\n{"="*96}\nPer keypoint (world RMSE vs conf^2 SLEAP, mm)\n{"="*96}')
    t = kp.pivot_table(index='keypoint', columns='variant',
                       values='world_rmse_mm').reindex(columns=order)
    t['delta_vs_base'] = t['c2d6_c5d4'] - t['base']
    t['pct_better'] = 100 * (t['base'] - t['c2d6_c5d4']) / t['base']
    print(t.sort_values('pct_better', ascending=False).round(3).to_string())

    print(f'\n{"="*96}\nPer keypoint (EGOCENTRIC RMSE, mm)\n{"="*96}')
    t2 = kp.pivot_table(index='keypoint', columns='variant',
                        values='ego_rmse_mm').reindex(columns=order)
    t2['pct_better'] = 100 * (t2['base'] - t2['c2d6_c5d4']) / t2['base']
    print(t2.sort_values('pct_better', ascending=False).round(3).to_string())

    print(f'\n{"="*96}\nC. Features vs conf^2 SLEAP\n{"="*96}')
    for feat in ['ear_mean_z', 'cum_heading_change']:
        for stratum in masks:
            s_ = ft[(ft.feature == feat) & (ft.stratum == stratum) &
                    (ft.reference == 'sleap_conf2')]
            if not len(s_):
                continue
            s_ = s_.set_index('variant').reindex(order).reset_index()
            cols = ['variant', 'rmse', 'med_abs', 'p90_abs', 'bias', 'pearson_r']
            if feat == 'cum_heading_change':
                cols += ['sign_agree_turn_pct']
            print(f'\n  --- {feat} ({s_.unit.iloc[0]}) | {stratum} ---')
            print(s_[cols].round(4).to_string(index=False))

    print(f'\n{"="*96}\nC2. Same features vs EQUAL-weight SLEAP (for reference)\n{"="*96}')
    for feat in ['ear_mean_z', 'cum_heading_change']:
        s_ = ft[(ft.feature == feat) & (ft.stratum == 'all') &
                (ft.reference == 'sleap_equal')].set_index('variant').reindex(order).reset_index()
        print(f'\n  --- {feat} | all ---')
        print(s_[['variant', 'rmse', 'med_abs', 'bias', 'pearson_r']].round(4).to_string(index=False))

    print(f'\n{"="*96}\nF. Body proportions (DANNCE median length / SLEAP median length)'
          f'\n   1.00 = same size.  The port obstruction foreshortens; a global'
          f'\n   Procrustes scale cannot see an anisotropic shrink.\n{"="*96}')
    sp = bone[bone.bone.str.startswith('[span]')]
    t3 = sp.pivot_table(index='bone', columns='variant', values='ratio').reindex(columns=order)
    t3['sleap_mm'] = sp[sp.variant == 'base'].set_index('bone')['sleap_median_mm']
    print(t3.round(4).to_string())
    print('\n  all bones pooled (median ratio / mean within-session CV):')
    bb = bone[~bone.bone.str.startswith('[span]')]
    for v in order:
        sub = bb[bb.variant == v]
        print(f'    {v:10s} median ratio {sub.ratio.median():.4f}   '
              f'IQR {sub.ratio.quantile(.25):.4f}-{sub.ratio.quantile(.75):.4f}   '
              f'mean bone CV {sub.dannce_cv.mean():.4f}')

    print(f'\nwrote {OUT}/adaptive_dannce_r9_{{keypoints,overall,features,cameras,bones}}.csv')


if __name__ == '__main__':
    main()
