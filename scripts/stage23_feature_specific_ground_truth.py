"""Stage 23 -- like stage22, but with a FEATURE-SPECIFIC ground truth instead
of one shared PC0/PC1-anchored ground truth for every combo.

User's explicit request after reviewing stage22/§8t/§8u's methodology: rebuild
the ground-truth anchor set SEPARATELY for each feature set being tested, in
that feature set's OWN space, so we are directly comparing "what DANNCE
identifies using feature set F" against "what SLEAP identifies using feature
set F" -- not routing every feature set through a shared rearing (PC0/PC1)
ground truth the way stage18/stage22 did.

Per session, per CANDIDATE FEATURE SET F (unchanged from stage22: 14 singles,
91 pairs, 24 targeted ear_mean_z+turn+stretch triples = 129 total):

  1. Pass 1 (shared across all F, unchanged from stage22): SLEAP matches the
     REAL production template (PC0/PC1, bounds x1.25) -> candidate windows m1.
     This is still the only place PC0/PC1 enters the pipeline -- it just seeds
     *which times are candidates at all*, exactly as in stage18/22.
  2. Rebuild a DANNCE-only template IN FEATURE SPACE F from DANNCE's own
     F-values at the m1 windows (bounds = std(SLEAP F-seg) + std(DANNCE
     F-seg), accepted_errors = round(1.5*n_features(F))).
  3. Match DANNCE's own F-trajectory against that F-space template ->
     gt_F, the ground-truth anchor set FOR THIS FEATURE SET SPECIFICALLY.
  4. Rebuild BOTH a SLEAP template and a DANNCE template in feature space F
     from gt_F (not from a shared PC0/PC1-anchored gt).
  5. Match each modality independently against its own F-space template,
     pair within 10 frames -> TP/FP/FN.

Same session restriction as stage22 (corrector-OFF + new-SLEAP-model era) and
the same world-space Procrustes alignment / causal-median filter / corrupted-
session guard for computing the 14 features.

Writes:
  results/spatial_confound/stage23_combo_results.csv
  results/spatial_confound/stage23_combo_summary.csv
  results/spatial_confound/stage23_gt_sizes.csv   (n ground-truth events per combo x session --
                                                    a feature that can't self-consistently
                                                    reproduce its own candidate windows in DANNCE
                                                    will show up here as a tiny/empty gt_F)
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
                 'angle_torsion_xy', 'angle_body_head_xy', 'angle_mid_head_xy',
                 'angle_torsion_ear_xy']
STRETCH_FEATURES = ['dist_snout_tb', 'dist_ears_tb', 'dist_spineF_tb', 'dist_earmean_hipmean']
FEATURE_NAMES = HEIGHT_FEATURES + TURN_FEATURES + STRETCH_FEATURES + ['speed']

MAX_COMBO_SIZE = 2
COMBOS = []
for r in range(1, MAX_COMBO_SIZE + 1):
    COMBOS.extend(itertools.combinations(FEATURE_NAMES, r))
for turn in TURN_FEATURES:
    for stretch in STRETCH_FEATURES:
        COMBOS.append(('ear_mean_z', turn, stretch))
print(f'{len(COMBOS)} feature combinations to test (all 1s and 2s, '
     f'plus {len(TURN_FEATURES) * len(STRETCH_FEATURES)} targeted height+turn+stretch triples)')


# --------------------------------------------------------------------------- #
# Session selection: corrector OFF + new SLEAP model (identical to stage22)
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
# World-space alignment (stage20/21/22 convention, unchanged)
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
# Generic helpers (unchanged from stage18/22)
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
    """Returns dict(m1, feat_sl, feat_dn) or None if unusable/excluded.
    Unlike stage22, this does NOT compute a shared ground truth -- that now
    happens per-combo in match_combo, using m1 as the (feature-independent)
    seed candidate set."""
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

    # ---- Pass 1 (shared seed, unchanged): SLEAP vs the REAL production template
    s_pcs, s_rot = ev.project_session_to_pcs(raw['triang_3d'][:n], tm, flip_z=True)
    pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)[:2]
    tmpl0 = tm['template'][:, :2]
    bounds0 = ev.template_bounds(tm) * 1.25
    m1, _ = ev.get_template_match_indices(s_pcs[:, pcu], tmpl0, bounds0, refractory_frames=REFRACTORY)
    if len(m1) == 0:
        return None
    m1 = ev.correct_matches(s_pcs, tmpl0, m1)

    # ---- world-space alignment + causal median filter -> the 14 features
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
    return dict(m1=m1, feat_sl=feat_sl, feat_dn=feat_dn)


def match_combo(combo, sd):
    """Feature-specific ground truth: rebuild the DANNCE-only template AND
    the ground-truth match set IN THIS FEATURE SET's own space (steps 2-3),
    then rebuild both evaluation templates from that feature-specific ground
    truth (step 4) and match independently (step 5)."""
    m1, feat_sl, feat_dn = sd['m1'], sd['feat_sl'], sd['feat_dn']
    s_traj = np.stack([feat_sl[f] for f in combo], axis=1)
    d_traj = np.stack([feat_dn[f] for f in combo], axis=1)
    acc = max(1, round(1.5 * len(combo)))

    # ---- step 2: DANNCE-only template IN FEATURE SPACE F, from the shared m1 seed windows
    s_seg1, _ = windows_at(s_traj, m1)
    d_seg1, _ = windows_at(d_traj, m1)
    if len(s_seg1) < 5:
        return None, 0
    dannce_tmpl0, _, bounds1 = rebuild_and_bounds(d_seg1, s_seg1)

    # ---- step 3: DANNCE matches its own F-space template -> gt_F (feature-specific ground truth)
    gt_f, _ = ev.get_template_match_indices(d_traj, dannce_tmpl0, bounds1,
                                            refractory_frames=REFRACTORY, accepted_errors=acc)
    if len(gt_f) == 0:
        return None, 0
    gt_f = ev.correct_matches(d_traj, dannce_tmpl0, gt_f)

    # ---- step 4: rebuild BOTH SLEAP and DANNCE templates in feature space F from gt_F
    s_seg, _ = windows_at(s_traj, gt_f)
    d_seg, _ = windows_at(d_traj, gt_f)
    if len(s_seg) < 5:
        return None, len(gt_f)
    sleap_tmpl, dannce_tmpl, bounds = rebuild_and_bounds(s_seg, d_seg)

    # ---- step 5: match each modality independently against its own template, pair within 10 frames
    sm, _ = ev.get_template_match_indices(s_traj, sleap_tmpl, bounds,
                                          refractory_frames=REFRACTORY, accepted_errors=acc)
    dm, _ = ev.get_template_match_indices(d_traj, dannce_tmpl, bounds,
                                          refractory_frames=REFRACTORY, accepted_errors=acc)
    sm = ev.correct_matches(s_traj, sleap_tmpl, sm) if len(sm) else np.array([], dtype=int)
    dm = ev.correct_matches(d_traj, dannce_tmpl, dm) if len(dm) else np.array([], dtype=int)
    sleap_only, dannce_only, both = get_match_overlap(dm, sm)
    return (len(both), len(sleap_only), len(dannce_only)), len(gt_f)


def main():
    sessions_by_rat = {rat: off_new_sessions(rat) for rat in ['R4', 'R5', 'R6']}
    for rat, ss in sessions_by_rat.items():
        print(f'{rat}: {len(ss)} corrector-OFF + new-model sessions', flush=True)

    rows, gt_rows = [], []
    for rat, sessions in sessions_by_rat.items():
        for i, session in enumerate(sessions):
            sd = session_data(rat, session)
            if sd is None:
                print(f'  [{rat}/{session}] unusable/excluded, skipping', flush=True)
                continue
            for combo in COMBOS:
                r, n_gt = match_combo(combo, sd)
                combo_name = '+'.join(combo)
                gt_rows.append(dict(rat=rat, session=session, combo=combo_name, n_gt=n_gt))
                if r is None:
                    continue
                tp_, fp_, fn_ = r
                rows.append(dict(rat=rat, session=session, combo=combo_name,
                                 n_features=len(combo), tp=tp_, fp=fp_, fn=fn_))
            print(f'  [{i+1}/{len(sessions)}] {rat}/{session} done', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/stage23_combo_results.csv', index=False)
    print(f'\nwrote {OUT}/stage23_combo_results.csv ({len(d)} combo-session rows)')

    gt = pd.DataFrame(gt_rows)
    gt.to_csv(f'{OUT}/stage23_gt_sizes.csv', index=False)
    print(f'wrote {OUT}/stage23_gt_sizes.csv')
    print('\nMedian feature-specific ground-truth events per session, by combo (smallest 10):')
    med = gt.groupby('combo').n_gt.median().sort_values()
    for combo, v in med.head(10).items():
        print(f'  {combo:55s}  median n_gt={v:.0f}')

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
    s.to_csv(f'{OUT}/stage23_combo_summary.csv', index=False)
    print(f'wrote {OUT}/stage23_combo_summary.csv')

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
