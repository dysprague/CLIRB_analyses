"""Stage 22 -- precision/recall of SLEAP-vs-DANNCE overlap using SUBSETS of
the stage21 interpretable features, restricted to corrector-OFF + new-SLEAP-
model sessions only (the most-accurate tracking era per stage20b/§8r).

Goal: find a small (1-3 feature) constrained, interpretable feature set that
gets close to precision>=80% AND recall>=80% for online behavior matching,
trading precision for recall if needed.

Methodology -- identical 3-stage ground-truth rebuild as stage18/§8m-§8o,
just with the stage21 world-space+causal-median features standing in for
stage18's egocentric-PCA feature spaces in the FINAL rebuild-and-match step:

  1. Pass 1: SLEAP matches the REAL production template (PC0/PC1, bounds x1.25).
  2. Rebuild a DANNCE-only template from DANNCE's PC0/PC1 values at those
     Pass-1 SLEAP windows -> ground-truth DANNCE match set ("gt").
  3. Separately: align DANNCE into SLEAP's world frame (session-level rigid
     Procrustes, sector-balanced, same guard as stage20/21), causal-median
     filter both modalities, compute all 10 stage21 features for each.
  4. For each CANDIDATE FEATURE SUBSET (all 10 singles + all 45 pairs):
     rebuild BOTH a SLEAP and a DANNCE template from that subset's values at
     the "gt" windows (bounds = sum of stds, accepted_errors =
     round(1.5*n_features), same convention as stage18/§8o).
  5. Match independently on each side, pair within 10 frames -> TP/FP/FN.

Session restriction: corrector_ckpt == null AND the new (n=9975) SLEAP model
(behavior_params.corrector_ckpt / process_params.model_path in each
session's own campy_config.yaml) -- the "post-switch: new_model +
corrector_OFF" stratum from stage20b (~32 sessions).

Writes:
  results/spatial_confound/stage22_combo_results.csv   (one row per combo x session)
  results/spatial_confound/stage22_combo_summary.csv    (pooled + per-rat, one row per combo)
"""
import os, sys, glob, itertools
import numpy as np
import pandas as pd
import yaml

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
DATA_ROOT = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
SECTOR_W = 30
N_PER_SECTOR = 300
W = ev.TEMPLATE_LEN
REFRACTORY = W
PAIR_THRESH = 10
FIT_RESIDUAL_GUARD_MM = 60.0
FPS = ev.FPS
RNG = np.random.default_rng(0)

SNOUT, EARL, EARR = NODE_IDX['Snout'], NODE_IDX['EarL'], NODE_IDX['EarR']
SPINEF, SPINEM, SPINEL, TAILBASE = (NODE_IDX['SpineF'], NODE_IDX['SpineM'],
                                    NODE_IDX['SpineL'], NODE_IDX['TailBase'])
HIPL, HIPR, SHOULDERL, SHOULDERR = (NODE_IDX['HipL'], NODE_IDX['HipR'],
                                    NODE_IDX['ShoulderL'], NODE_IDX['ShoulderR'])

HEIGHT_FEATURES = ['snout_z', 'ear_snout_mean_z', 'ear_mean_z']
TURN_FEATURES = ['angle_lower_xy', 'angle_upper_xy', 'angle_mid_xy', 'angle_skip_xy',
                 'angle_torsion_xy', 'angle_body_head_xy']
STRETCH_FEATURES = ['dist_snout_tb', 'dist_ears_tb', 'dist_spineF_tb', 'dist_earmean_hipmean']
FEATURE_NAMES = HEIGHT_FEATURES + TURN_FEATURES + STRETCH_FEATURES + ['speed']

MAX_COMBO_SIZE = 2   # exhaustive singles + pairs over all 14 features...
COMBOS = []
for r in range(1, MAX_COMBO_SIZE + 1):
    COMBOS.extend(itertools.combinations(FEATURE_NAMES, r))
# ...plus targeted triples: best height feature + every turn x every stretch combination,
# directly answering "does height + turn + crouch/stretch work as a 3-feature set."
for turn in TURN_FEATURES:
    for stretch in STRETCH_FEATURES:
        COMBOS.append(('ear_mean_z', turn, stretch))
print(f'{len(COMBOS)} feature combinations to test (all 1s and 2s, '
     f'plus {len(TURN_FEATURES) * len(STRETCH_FEATURES)} targeted height+turn+stretch triples)')


# --------------------------------------------------------------------------- #
# Session selection: corrector OFF + new SLEAP model, via each session's own
# campy_config.yaml (same convention as stage20b_stratify_by_pipeline_version.py)
# --------------------------------------------------------------------------- #
def session_meta(rat, session):
    cfg = os.path.join(DATA_ROOT, rat, session, 'sleap', 'campy_config.yaml')
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg) as fh:
            c = yaml.safe_load(fh)
        model = c.get('process_params', {}).get('model_path', 'MISSING')
        corr_raw = c.get('behavior_params', {}).get('corrector_ckpt', None)
        corrector = 'ON' if corr_raw else 'OFF'
    except Exception:
        return None
    model_short = 'old_n2298' if '2298' in model else ('new_n9975' if '9975' in model else model)
    return dict(model=model_short, corrector=corrector)


def off_new_sessions(rat):
    out = []
    for f in sorted(glob.glob(f'data/cache/{rat}/*.npz')):
        session = os.path.basename(f)[:-4]
        raw = sc.get_raw_session(rat, session, want_dual=True)
        if raw is None or not raw.get('has_dual', False):
            continue
        meta = session_meta(rat, session)
        if meta is None:
            continue
        if meta['model'] == 'new_n9975' and meta['corrector'] == 'OFF':
            out.append(session)
    return out


# --------------------------------------------------------------------------- #
# World-space alignment (stage20/21 convention)
# --------------------------------------------------------------------------- #
def robust_proc(A, B, iters=5, keep=0.7):
    idx = np.arange(len(A)); f = None
    for _ in range(iters):
        f = fit_procrustes(A[idx], B[idx], try_z_flip=True)
        r = np.linalg.norm(f['apply'](A) - B, axis=-1).mean(axis=1)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 200:
            break
    return f


def sector_balanced_sample(com_xy, n_per_sector=N_PER_SECTOR, rng=RNG):
    sec = sector_index(ARENA.theta(com_xy), SECTOR_W)
    samp = []
    for k in range(12):
        ii = np.where(sec == k)[0]
        if len(ii):
            samp.extend(rng.choice(ii, size=min(len(ii), n_per_sector), replace=False))
    return np.array(sorted(samp), dtype=int)


signed_angle_xy = feat.signed_angle_xy


# Feature definitions live in clirb.features (the single source of truth, §8s-§8z).
compute_all_features = feat.compute_features


# --------------------------------------------------------------------------- #
# stage18-style ground-truth rebuild + generic feature-space matching
# --------------------------------------------------------------------------- #
def rebuild_and_bounds(seg_a, seg_b):
    ta, tb = seg_a.mean(axis=0), seg_b.mean(axis=0)
    bounds = seg_a.std(axis=0) + seg_b.std(axis=0)
    return ta, tb, bounds


def windows_at(traj, ends, w=W):
    ends = np.asarray(ends, dtype=np.int64)
    ok = (ends >= w - 1) & (ends < len(traj))
    e = ends[ok]
    idx = (e - (w - 1))[:, None] + np.arange(w)[None, :]
    return traj[idx], e


def get_match_overlap(dannce_matches, sleap_matches, thresh=PAIR_THRESH):
    sleap_only = np.array([], dtype=int); dannce_only = np.array([], dtype=int)
    both = np.array([], dtype=int)
    if len(dannce_matches) == 0 and len(sleap_matches) == 0:
        return sleap_only, dannce_only, both
    if len(dannce_matches) == 0:
        return np.asarray(sleap_matches), dannce_only, both
    if len(sleap_matches) == 0:
        return sleap_only, np.asarray(dannce_matches), both
    s = np.asarray(sleap_matches); d = np.asarray(dannce_matches)
    c_for_s = np.array([np.argmin(np.abs(d - t)) for t in s])
    c_for_d = np.array([np.argmin(np.abs(s - t)) for t in d])
    both = s[np.abs(s - d[c_for_s]) < thresh]
    sleap_only = s[np.abs(s - d[c_for_s]) >= thresh]
    dannce_only = d[np.abs(d - s[c_for_d]) >= thresh]
    return sleap_only, dannce_only, both


def session_data(rat, session):
    """Returns dict(gt, feat_sl, feat_dn) or None if unusable/excluded."""
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

    # ---- PC0/PC1 pipeline: Pass-1 SLEAP anchor -> DANNCE ground truth
    s_pcs, s_rot = ev.project_session_to_pcs(raw['triang_3d'][:n], tm, flip_z=True)
    d_pcs, d_rot = ev.project_session_to_pcs(d3[:n], tm, flip_z=False)
    pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)[:2]

    tmpl0 = tm['template'][:, :2]
    bounds0 = ev.template_bounds(tm) * 1.25
    m1, _ = ev.get_template_match_indices(s_pcs[:, pcu], tmpl0, bounds0, refractory_frames=REFRACTORY)
    if len(m1) == 0:
        return None
    m1 = ev.correct_matches(s_pcs, tmpl0, m1)

    s_seg1, _ = windows_at(s_pcs[:, pcu], m1)
    d_seg1, _ = windows_at(d_pcs[:, pcu], m1)
    if len(s_seg1) == 0:
        return None
    dannce_tmpl0, _, bounds1 = rebuild_and_bounds(d_seg1, s_seg1)

    gt, _ = ev.get_template_match_indices(d_pcs[:, pcu], dannce_tmpl0, bounds1, refractory_frames=REFRACTORY)
    if len(gt) == 0:
        return None
    gt = ev.correct_matches(d_pcs, dannce_tmpl0, gt)

    # ---- world-space alignment + causal median filter -> the 10 features
    sl_raw = np.asarray(raw['triang_3d'][:n], dtype=np.float32)
    dn_raw = np.asarray(d3[:n], dtype=np.float32)
    samp = sector_balanced_sample(sl_raw.mean(axis=1)[:, :2])
    if len(samp) < 500:
        return None
    fit = robust_proc(dn_raw[samp], sl_raw[samp])
    dn_aligned = fit['apply'](dn_raw)
    overall_rmse = float(np.sqrt(((sl_raw - dn_aligned) ** 2).sum(axis=-1).mean()))
    if fit['residual'] > FIT_RESIDUAL_GUARD_MM or overall_rmse > 2 * FIT_RESIDUAL_GUARD_MM:
        return None

    sl = ev.causal_median_5_kp(sl_raw, warmup='online')
    dn = ev.causal_median_5_kp(dn_aligned, warmup='online')
    feat_sl = compute_all_features(sl)
    feat_dn = compute_all_features(dn)
    return dict(gt=gt, feat_sl=feat_sl, feat_dn=feat_dn)


def match_combo(combo, sd):
    gt, feat_sl, feat_dn = sd['gt'], sd['feat_sl'], sd['feat_dn']
    s_traj = np.stack([feat_sl[f] for f in combo], axis=1)
    d_traj = np.stack([feat_dn[f] for f in combo], axis=1)
    s_seg, _ = windows_at(s_traj, gt)
    d_seg, _ = windows_at(d_traj, gt)
    if len(s_seg) < 5:
        return None
    sleap_tmpl, dannce_tmpl, bounds = rebuild_and_bounds(s_seg, d_seg)
    acc = max(1, round(1.5 * len(combo)))
    sm, _ = ev.get_template_match_indices(s_traj, sleap_tmpl, bounds,
                                          refractory_frames=REFRACTORY, accepted_errors=acc)
    dm, _ = ev.get_template_match_indices(d_traj, dannce_tmpl, bounds,
                                          refractory_frames=REFRACTORY, accepted_errors=acc)
    sm = ev.correct_matches(s_traj, sleap_tmpl, sm) if len(sm) else np.array([], dtype=int)
    dm = ev.correct_matches(d_traj, dannce_tmpl, dm) if len(dm) else np.array([], dtype=int)
    sleap_only, dannce_only, both = get_match_overlap(dm, sm)
    return len(both), len(sleap_only), len(dannce_only)


def main():
    sessions_by_rat = {rat: off_new_sessions(rat) for rat in ['R4', 'R5', 'R6']}
    for rat, ss in sessions_by_rat.items():
        print(f'{rat}: {len(ss)} corrector-OFF + new-model sessions', flush=True)

    rows = []
    for rat, sessions in sessions_by_rat.items():
        for i, session in enumerate(sessions):
            sd = session_data(rat, session)
            if sd is None:
                print(f'  [{rat}/{session}] unusable/excluded, skipping', flush=True)
                continue
            for combo in COMBOS:
                r = match_combo(combo, sd)
                if r is None:
                    continue
                tp_, fp_, fn_ = r
                rows.append(dict(rat=rat, session=session, combo='+'.join(combo),
                                 n_features=len(combo), tp=tp_, fp=fp_, fn=fn_))
            print(f'  [{i+1}/{len(sessions)}] {rat}/{session} done', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/stage22_combo_results.csv', index=False)
    print(f'\nwrote {OUT}/stage22_combo_results.csv ({len(d)} combo-session rows)')

    summ_rows = []
    for combo, g in d.groupby('combo'):
        tp_, fp_, fn_ = g.tp.sum(), g.fp.sum(), g.fn.sum()
        prec = tp_ / max(tp_ + fp_, 1); rec = tp_ / max(tp_ + fn_, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
        summ_rows.append(dict(rat='pooled', combo=combo, n_features=g.n_features.iloc[0],
                              tp=tp_, fp=fp_, fn=fn_, precision=prec, recall=rec, f1=f1, f2=f2,
                              meets_target=(prec >= 0.8 and rec >= 0.8)))
        for rat, gr in g.groupby('rat'):
            tp_, fp_, fn_ = gr.tp.sum(), gr.fp.sum(), gr.fn.sum()
            prec = tp_ / max(tp_ + fp_, 1); rec = tp_ / max(tp_ + fn_, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
            summ_rows.append(dict(rat=rat, combo=combo, n_features=gr.n_features.iloc[0],
                                  tp=tp_, fp=fp_, fn=fn_, precision=prec, recall=rec, f1=f1, f2=f2,
                                  meets_target=(prec >= 0.8 and rec >= 0.8)))

    s = pd.DataFrame(summ_rows)
    s.to_csv(f'{OUT}/stage22_combo_summary.csv', index=False)
    print(f'wrote {OUT}/stage22_combo_summary.csv')

    pooled = s[s.rat == 'pooled'].sort_values('f2', ascending=False)
    print(f'\n=== Top 20 combos, pooled, sorted by F2 (recall-weighted) ===')
    for row in pooled.head(20).itertuples():
        flag = '  <-- MEETS 80/80 TARGET' if row.meets_target else ''
        print(f'  [{row.n_features}] {row.combo:55s}  P={row.precision:.3f}  R={row.recall:.3f}  '
             f'F1={row.f1:.3f}  F2={row.f2:.3f}{flag}')

    hit = pooled[pooled.meets_target]
    print(f'\n{len(hit)}/{len(pooled)} combos meet precision>=0.8 AND recall>=0.8 (pooled).')


if __name__ == '__main__':
    main()
