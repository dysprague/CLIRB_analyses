"""F1 of template matching between SLEAP and DANNCE, using a REAR-DEFINED
template built from `ear_mean_z` and `cum_heading_change`.

The feature-reliability appendix measured these two features frame by frame
(RMSE, correlation, sign agreement).  That says how well the signals agree; it
does not say whether a corridor matcher driven by them fires on the same
events.  This script answers that, following the convention stage22 established
(SPATIAL_CONFOUND_RESULTS.md section 8t), with one deliberate change.

The convention, kept:
    window W = 30 frames ending at a ground-truth frame
    template = the MEAN window, computed separately for each modality
    bounds   = std(SLEAP windows) + std(DANNCE windows), shared between them
    accepted_errors = max(1, round(1.5 x n_features))
    refractory = W frames, then ev.correct_matches to re-centre
    a SLEAP match and a DANNCE match are the same event if they fall within
    PAIR_THRESH = 10 frames

The change: stage22's ground truth came from each session's own PC-space
template, which R7/R8/R9 do not have.  Here the ground truth is the set of
DANNCE REAR END-FRAMES from the hysteresis detector at the rig's deployed
thresholds (head_min 140 / reset 100 / 0.5 s hold).  Using DANNCE's rears to
define the template for BOTH modalities is the point -- it keeps SLEAP's own
detection bias out of the template, so the comparison is not circular.

Precision is "of the events SLEAP's matcher fired, how many did DANNCE's also
fire", recall is the converse; both are SLEAP-vs-DANNCE agreement, not
agreement with behaviour.

Writes results/newcal/feature_template_matching.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb import features as cf
from clirb.spatial import dual as du
from scripts.newcal_rear_detector import detect_rears_fast
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample
from scripts.newcal_eras import ERAS, ERA_SHORT

OUT = 'results/newcal'
W = ev.TEMPLATE_LEN          # 30
REFRACTORY = W
PAIR_THRESH = 10
DEPLOYED = dict(rise=140.0, reset=100.0, min_frames=10)
COMBOS = [('ear_mean_z',), ('cum_heading_change',),
          ('ear_mean_z', 'cum_heading_change')]
MIN_GT = 5


def windows_at(traj, ends, w=W):
    ends = np.asarray(ends, dtype=int)
    ok = (ends >= w - 1) & (ends < len(traj))
    e = ends[ok]
    if not len(e):
        return np.empty((0, w, traj.shape[1])), ok
    idx = (e - (w - 1))[:, None] + np.arange(w)[None, :]
    return traj[idx], ok


def overlap(dannce_matches, sleap_matches, thresh=PAIR_THRESH):
    s = np.asarray(sleap_matches, dtype=int)
    d = np.asarray(dannce_matches, dtype=int)
    if not len(d) and not len(s):
        return 0, 0, 0
    if not len(d):
        return 0, len(s), 0
    if not len(s):
        return 0, 0, len(d)
    c_s = np.array([np.argmin(np.abs(d - t)) for t in s])
    c_d = np.array([np.argmin(np.abs(s - t)) for t in d])
    both = int((np.abs(s - d[c_s]) < thresh).sum())
    s_only = int((np.abs(s - d[c_s]) >= thresh).sum())
    d_only = int((np.abs(d - s[c_d]) >= thresh).sum())
    return both, s_only, d_only


def session_features(rat, session):
    raw = sc.get_raw_session(rat, session, want_dual=True)
    if raw is None or not raw.get('has_dual', False):
        return None
    d3 = du.dannce_aligned(raw)
    if d3 is None:
        return None
    n = min(len(raw['triang_3d']), len(d3))
    if n < 1000:
        return None
    sl_raw = np.asarray(raw['triang_3d'][:n], np.float32)
    dn_raw = np.asarray(d3[:n], np.float32)
    samp, _ = balanced_sample(sl_raw.mean(axis=1)[:, :2], rng=np.random.default_rng(0))
    if len(samp) < 500:
        return None
    fit = robust_proc(dn_raw[samp], sl_raw[samp])
    S = ev.causal_median_5_kp(sl_raw, warmup='online').copy()
    D = ev.causal_median_5_kp(fit['apply'](dn_raw), warmup='online').copy()
    S[:, :, 2] *= -1.0
    D[:, :, 2] *= -1.0
    return cf.compute_features(S), cf.compute_features(D), float(fit['residual'])


def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]
    rows = []
    for i, r in enumerate(inv.itertuples(), 1):
        out = session_features(r.rat, r.session)
        if out is None:
            continue
        f_sl, f_dn, resid = out
        gt_eps = detect_rears_fast(f_dn['ear_mean_z'], **DEPLOYED)
        gt = gt_eps[:, 1] if len(gt_eps) else np.array([], int)
        for combo in COMBOS:
            s_traj = np.stack([f_sl[f] for f in combo], axis=1)
            d_traj = np.stack([f_dn[f] for f in combo], axis=1)
            s_seg, _ = windows_at(s_traj, gt)
            d_seg, _ = windows_at(d_traj, gt)
            if len(s_seg) < MIN_GT:
                continue
            s_tmpl, d_tmpl = s_seg.mean(axis=0), d_seg.mean(axis=0)
            bounds = s_seg.std(axis=0) + d_seg.std(axis=0)
            acc = max(1, round(1.5 * len(combo)))
            sm, _ = ev.get_template_match_indices(s_traj, s_tmpl, bounds,
                                                  refractory_frames=REFRACTORY,
                                                  accepted_errors=acc)
            dm, _ = ev.get_template_match_indices(d_traj, d_tmpl, bounds,
                                                  refractory_frames=REFRACTORY,
                                                  accepted_errors=acc)
            sm = ev.correct_matches(s_traj, s_tmpl, sm) if len(sm) else np.array([], int)
            dm = ev.correct_matches(d_traj, d_tmpl, dm) if len(dm) else np.array([], int)
            tp, fp, fn = overlap(dm, sm)
            rows.append(dict(rat=r.rat, session=r.session, era=r.era,
                             combo='+'.join(combo), n_features=len(combo),
                             n_gt=len(gt), n_sleap=len(sm), n_dannce=len(dm),
                             tp=tp, fp=fp, fn=fn,
                             procrustes_residual_mm=resid))
        print(f'[{i}/{len(inv)}] {r.rat}/{r.session} ({r.era}) gt={len(gt)}', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/feature_template_matching.csv', index=False)

    pd.set_option('display.width', 220)
    print('\n=== Pooled precision / recall / F1 by era and feature set ===')
    g = d.groupby(['combo', 'era'])[['tp', 'fp', 'fn', 'n_gt', 'n_sleap', 'n_dannce']].sum()
    g['precision'] = g.tp / (g.tp + g.fp).replace(0, np.nan)
    g['recall'] = g.tp / (g.tp + g.fn).replace(0, np.nan)
    g['f1'] = 2 * g.precision * g.recall / (g.precision + g.recall)
    g['n_sessions'] = d.groupby(['combo', 'era']).size()
    order = [e for e in ERAS]
    for combo in ['+'.join(c) for c in COMBOS]:
        sub = g.loc[combo].reindex(order)
        print(f'\n--- {combo} ---')
        print(sub[['n_sessions', 'n_gt', 'n_sleap', 'n_dannce',
                   'precision', 'recall', 'f1']].round(3).to_string())
    print(f'\nwrote {OUT}/feature_template_matching.csv')


if __name__ == '__main__':
    main()
