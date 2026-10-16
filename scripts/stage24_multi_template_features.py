"""Stage 24 -- evaluate candidate features across FOUR independent behavioral
templates (rearing, turning, rear+turn, locomotion), not just the single
rearing-anchored ground truth stage18-23 used throughout.

Motivation: §8u/§8v found that testing a turn feature against a REARING
ground truth structurally disadvantages it (turn direction isn't
rearing-characteristic) and flagged the correct fix -- build an independent
ground-truth anchor set for turning itself -- as the needed next step, not yet
done. This script does that, plus rear+turn and locomotion, so a feature set
can be judged on the behavior it's actually meant to detect, and so a
combined feature set can be checked for whether it detects rearing AND
turning at once (the user's stated goal: reward specific rear orientations).

Each template follows the SAME 3-stage structure §8v validated for rearing,
just with a different, template-specific SEED step (stage 1):

  1. SEED (behavior-specific, built from SLEAP's own signal, no DANNCE):
       rearing   : SLEAP matches the real production template (PC0/PC1,
                   bounds x1.25) -- exactly stage18-23's Pass 1.
       turning   : hysteresis event detector (rise/reset/min_frames, the same
                   generic algorithm ev.detect_rears implements) on SLEAP's
                   own |angle_torsion_xy|. Thresholds calibrated empirically
                   (45/25 deg) to give event counts/session comparable to the
                   rearing seed pool across all 3 rats (72-118 events).
       rear+turn : the SUBSET of rearing-seed windows whose trailing 30
                   frames also cross the turning-seed rise threshold in
                   |angle_torsion_xy| -- i.e. rears where SLEAP's own signal
                   says the body was also significantly turned.
       locomotion: hysteresis on SLEAP's own `speed`, EXCLUDING any bout that
                   overlaps a rearing-seed window (so locomotion means
                   floor-level movement, not a moving rear).
  2. Rebuild a DANNCE-only template IN THE TESTED FEATURE SET's own space
     from DANNCE's values at the seed windows (stage23's fix -- no shared
     PC0/PC1-anchored ground truth).
  3. DANNCE matches its own template -> gt_F, the ground truth for THIS
     (feature set, template) pair.
  4. Rebuild both a SLEAP and a DANNCE template in feature space F from gt_F.
  5. Match each modality independently against its own template, pair within
     10 frames -> TP/FP/FN.

Same corrector-OFF/new-SLEAP-model session restriction as stage22/23 (the
most-accurate tracking era, §8r).

Writes:
  results/spatial_confound/stage24_seed_counts.csv     (per-session seed pool sizes, sanity check)
  results/spatial_confound/stage24_results.csv          (one row per template x combo x session)
  results/spatial_confound/stage24_summary.csv          (pooled + per-rat precision/recall)
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

# --- template-specific seed thresholds, calibrated on real data (see module docstring) ---
TURN_RISE, TURN_RESET, TURN_MIN_FRAMES = 45.0, 25.0, 10
LOCO_RISE, LOCO_RESET, LOCO_MIN_FRAMES = 150.0, 60.0, 10

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

TEMPLATES = ['rearing', 'turning', 'rear_turn', 'locomotion']

# Candidate combos: every single feature, every height x turn pair (the user's
# stated goal -- detect rearing and turning together), plus a couple of
# height+turn+stretch triples built from the best-established candidates.
COMBOS = [(f,) for f in FEATURE_NAMES]
for h in HEIGHT_FEATURES:
    for t in TURN_FEATURES:
        COMBOS.append((h, t))
for turn in ['angle_torsion_xy', 'angle_skip_xy']:
    COMBOS.append(('ear_mean_z', turn, 'dist_earmean_hipmean'))
print(f'{len(COMBOS)} feature combinations x {len(TEMPLATES)} templates '
     f'= {len(COMBOS) * len(TEMPLATES)} combo-template evaluations', flush=True)


# --------------------------------------------------------------------------- #
# Session selection (identical to stage22/23)
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
# World-space alignment + features (identical to stage21/22/23)
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
# Generic helpers (identical to stage18/22/23)
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


# --------------------------------------------------------------------------- #
# The four behavior-specific SEED detectors, all built from SLEAP's own signal
# --------------------------------------------------------------------------- #
def seed_rearing(s_pcs, tm, pcu):
    tmpl0 = tm['template'][:, :2]
    bounds0 = ev.template_bounds(tm) * 1.25
    m1, _ = ev.get_template_match_indices(s_pcs[:, pcu], tmpl0, bounds0, refractory_frames=REFRACTORY)
    if len(m1) == 0:
        return np.array([], dtype=np.int64)
    return ev.correct_matches(s_pcs, tmpl0, m1)


def seed_turning(feat_sl):
    sig = np.abs(feat_sl['angle_torsion_xy'])
    events = ev.detect_rears(sig, rise=TURN_RISE, reset=TURN_RESET, min_frames=TURN_MIN_FRAMES)
    return np.array([e[1] for e in events], dtype=np.int64)


def seed_rear_turn(rear_ends, feat_sl, w=W):
    sig = np.abs(feat_sl['angle_torsion_xy'])
    n = len(sig)
    keep = []
    for e in rear_ends:
        s0 = max(0, e - w + 1)
        if sig[s0:e + 1].max(initial=0.0) >= TURN_RISE:
            keep.append(e)
    return np.array(keep, dtype=np.int64)


def seed_locomotion(feat_sl, rear_ends, w=W):
    n = len(feat_sl['speed'])
    reared = np.zeros(n, dtype=bool)
    for e in rear_ends:
        s0 = max(0, e - w + 1)
        reared[s0:e + 1] = True
    sig = feat_sl['speed']
    events = ev.detect_rears(sig, rise=LOCO_RISE, reset=LOCO_RESET, min_frames=LOCO_MIN_FRAMES)
    keep = [e[1] for e in events if not reared[e[0]:e[1] + 1].any()]
    return np.array(keep, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Per-session setup + per-(template, combo) match
# --------------------------------------------------------------------------- #
def session_data(rat, session):
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

    s_pcs, s_rot = ev.project_session_to_pcs(raw['triang_3d'][:n], tm, flip_z=True)
    pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)[:2]

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

    rear_ends = seed_rearing(s_pcs, tm, pcu)
    turn_ends = seed_turning(feat_sl)
    rearturn_ends = seed_rear_turn(rear_ends, feat_sl)
    loco_ends = seed_locomotion(feat_sl, rear_ends)

    seeds = dict(rearing=rear_ends, turning=turn_ends, rear_turn=rearturn_ends,
                locomotion=loco_ends)
    return dict(seeds=seeds, feat_sl=feat_sl, feat_dn=feat_dn)


def match_combo(combo, seed_ends, feat_sl, feat_dn):
    if len(seed_ends) < 5:
        return None, 0
    s_traj = np.stack([feat_sl[f] for f in combo], axis=1)
    d_traj = np.stack([feat_dn[f] for f in combo], axis=1)
    acc = max(1, round(1.5 * len(combo)))

    s_seg1, _ = windows_at(s_traj, seed_ends)
    d_seg1, _ = windows_at(d_traj, seed_ends)
    if len(s_seg1) < 5:
        return None, 0
    dannce_tmpl0, _, bounds1 = rebuild_and_bounds(d_seg1, s_seg1)

    gt_f, _ = ev.get_template_match_indices(d_traj, dannce_tmpl0, bounds1,
                                            refractory_frames=REFRACTORY, accepted_errors=acc)
    if len(gt_f) == 0:
        return None, 0
    gt_f = ev.correct_matches(d_traj, dannce_tmpl0, gt_f)

    s_seg, _ = windows_at(s_traj, gt_f)
    d_seg, _ = windows_at(d_traj, gt_f)
    if len(s_seg) < 5:
        return None, len(gt_f)
    sleap_tmpl, dannce_tmpl, bounds = rebuild_and_bounds(s_seg, d_seg)

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

    rows, seed_rows = [], []
    for rat, sessions in sessions_by_rat.items():
        for i, session in enumerate(sessions):
            sd = session_data(rat, session)
            if sd is None:
                print(f'  [{rat}/{session}] unusable/excluded, skipping', flush=True)
                continue
            seed_rows.append(dict(rat=rat, session=session,
                                  **{f'n_seed_{k}': len(v) for k, v in sd['seeds'].items()}))
            for template in TEMPLATES:
                seed_ends = sd['seeds'][template]
                for combo in COMBOS:
                    r, n_gt = match_combo(combo, seed_ends, sd['feat_sl'], sd['feat_dn'])
                    if r is None:
                        continue
                    tp_, fp_, fn_ = r
                    rows.append(dict(rat=rat, session=session, template=template,
                                     combo='+'.join(combo), n_features=len(combo),
                                     tp=tp_, fp=fp_, fn=fn_, n_gt=n_gt))
            print(f'  [{i+1}/{len(sessions)}] {rat}/{session} done '
                 f'(seeds: rear={len(sd["seeds"]["rearing"])} turn={len(sd["seeds"]["turning"])} '
                 f'rear+turn={len(sd["seeds"]["rear_turn"])} loco={len(sd["seeds"]["locomotion"])})',
                 flush=True)

    pd.DataFrame(seed_rows).to_csv(f'{OUT}/stage24_seed_counts.csv', index=False)
    print(f'\nwrote {OUT}/stage24_seed_counts.csv', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/stage24_results.csv', index=False)
    print(f'wrote {OUT}/stage24_results.csv ({len(d)} rows)', flush=True)

    summ_rows = []
    for (template, combo), g in d.groupby(['template', 'combo']):
        tp_, fp_, fn_ = g.tp.sum(), g.fp.sum(), g.fn.sum()
        prec = tp_ / max(tp_ + fp_, 1); rec = tp_ / max(tp_ + fn_, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        summ_rows.append(dict(rat='pooled', template=template, combo=combo,
                              n_features=g.n_features.iloc[0], tp=tp_, fp=fp_, fn=fn_,
                              precision=prec, recall=rec, f1=f1,
                              meets_target=(prec >= 0.8 and rec >= 0.8)))
        for rat, gr in g.groupby('rat'):
            tp_, fp_, fn_ = gr.tp.sum(), gr.fp.sum(), gr.fn.sum()
            prec = tp_ / max(tp_ + fp_, 1); rec = tp_ / max(tp_ + fn_, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            summ_rows.append(dict(rat=rat, template=template, combo=combo,
                                  n_features=gr.n_features.iloc[0], tp=tp_, fp=fp_, fn=fn_,
                                  precision=prec, recall=rec, f1=f1,
                                  meets_target=(prec >= 0.8 and rec >= 0.8)))

    s = pd.DataFrame(summ_rows)
    s.to_csv(f'{OUT}/stage24_summary.csv', index=False)
    print(f'wrote {OUT}/stage24_summary.csv', flush=True)

    pooled = s[s.rat == 'pooled']
    for template in TEMPLATES:
        print(f'\n=== {template}: top 10 combos by F1, pooled ===')
        top = pooled[pooled.template == template].sort_values('f1', ascending=False).head(10)
        for row in top.itertuples():
            flag = '  <-- MEETS 80/80' if row.meets_target else ''
            print(f'  [{row.n_features}] {row.combo:45s}  P={row.precision:.3f}  R={row.recall:.3f}'
                 f'  F1={row.f1:.3f}{flag}')


if __name__ == '__main__':
    main()
