"""Stage 18 — multi_session.ipynb §2.1-style overlap analysis, extended with
a DANNCE-ground-truthing step and 5 feature spaces.

Per session (corrector-adjusted SLEAP `triang_3d` + aligned DANNCE, matching
§2.1's own data source exactly):

  1. Pass 1: SLEAP matches against the REAL production template, bounds x1.25.
  2. Rebuild a DANNCE template from DANNCE's values at those Pass-1 SLEAP
     match windows (bounds = std(sleap seg) + std(dannce seg), mirroring
     §2.1's Pass 2 formula exactly).
  3. Rerun matching in DANNCE space against that rebuilt template -> the
     "ground-truth" DANNCE match set (still somewhat circular through SLEAP's
     Pass-1 matches, as discussed, but a step more independent than using
     SLEAP matches directly).
  4. For each of 5 feature spaces (full23 2-PC baseline, axial 9kp PCA,
     snout height, body rotation angle, snout+bodyrot combined): rebuild BOTH
     a SLEAP template and a DANNCE template from the two modalities' values
     at those ground-truth DANNCE match windows (shared bounds = sum of both
     modalities' std, again mirroring §2.1).
  5. Run matching independently on SLEAP and on DANNCE for each feature
     space, pair up the two match-index lists within +/-10 frames
     (§2.1's get_match_overlap) -> SLEAP-only (FP), DANNCE-only (FN), both (TP).

accepted_errors scales with feature count to preserve relative tolerance:
round(1.5 * n_features) -> 2 for 1-feature spaces, 3 for 2-feature spaces
(matches the real rule's 3-of-60 = 5% convention already used in stage17b).

Writes:
  results/spatial_confound/stage18_events.csv       (one row per matched event)
  results/spatial_confound/stage18_session_summary.csv
  results/spatial_confound/axial_basis_<rat>.npz
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial import dual as du
from clirb.spatial.geometry import ARENA, sector_index
from config import NODE_IDX
from skeleton import normalize_skeleton_batch

OUT = 'results/spatial_confound'
SECTOR_W = 30
W = ev.TEMPLATE_LEN                 # 30
REFRACTORY = W
PAIR_THRESH = 10                    # frames, exactly §2.1's get_match_overlap default
AXIAL = ['Snout', 'EarL', 'EarR', 'SpineF', 'SpineM', 'SpineL', 'TailBase', 'HipL', 'HipR']
AXIAL_IDX = [NODE_IDX[n] for n in AXIAL]
RNG = np.random.default_rng(0)

FEATURES = ['full23', 'axial', 'axial2', 'snout_z', 'bodyrot', 'snout_bodyrot']
NFEAT = {'full23': 2, 'axial': 1, 'axial2': 2, 'snout_z': 1, 'bodyrot': 1, 'snout_bodyrot': 2}
ACCEPTED = {k: max(1, round(1.5 * v)) for k, v in NFEAT.items()}
AXIAL_VARIANTS = {'axial': 1, 'axial2': 2}   # name -> n_components, both fit the same way


def body_rotation_angle(pose):
    SF = pose[:, NODE_IDX['SpineF'], :]; SM = pose[:, NODE_IDX['SpineM'], :]
    TB = pose[:, NODE_IDX['TailBase'], :]
    v1 = SF - SM; v2 = SM - TB
    n1 = np.linalg.norm(v1, axis=-1); n2 = np.linalg.norm(v2, axis=-1)
    cos = (v1 * v2).sum(-1) / np.maximum(n1 * n2, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def dual_sessions(rat):
    out = []
    for f in sorted(glob.glob(f'data/cache/{rat}/*.npz')):
        session = os.path.basename(f)[:-4]
        raw = sc.get_raw_session(rat, session, want_dual=True)
        if raw is not None and raw.get('has_dual', False):
            out.append(session)
    return out


def fit_axial_basis(rat, sessions, n_per_sector=40, n_components=1):
    """Fit an n_components-PCA on the 9 axial keypoints' egocentric coords,
    pooled sector-balanced across all of this rat's sessions. Each component
    k is sign-oriented to correlate positively with the REAL template's PC k
    (pcs_to_use[k]) -- purely for interpretability (mean "more reared/more
    like real PC1" = higher); a sign flip would not change which frames pass
    the corridor check, since that test is symmetric around the mean.
    """
    subs, corrs = [], []
    for session in sessions:
        raw = sc.get_raw_session(rat, session, want_dual=True)
        tm = tp.load_template(rat, session)
        if tm is None:
            continue
        d3 = du.dannce_aligned(raw)
        if d3 is None:
            continue
        d_pcs, d_rot = ev.project_session_to_pcs(d3, tm, flip_z=False)
        com = raw['triang_3d'][:len(d3)].mean(axis=1)
        sec = sector_index(ARENA.theta(com[:, :2]), SECTOR_W)
        pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)[:n_components]
        for s in range(12):
            ii = np.where(sec == s)[0]
            if len(ii):
                pick = RNG.choice(ii, min(len(ii), n_per_sector), replace=False)
                subs.append(d_rot[pick][:, AXIAL_IDX, :].reshape(len(pick), -1))
                corrs.append(d_pcs[pick][:, pcu])
    X = np.concatenate(subs); Y = np.concatenate(corrs)          # Y: (N, n_components)
    mu = X.mean(axis=0)
    pca = PCA(n_components=n_components).fit(X - mu)
    comps = pca.components_.copy()                               # (n_components, 27)
    for k in range(n_components):
        if np.corrcoef((X - mu) @ comps[k], Y[:, k])[0, 1] < 0:
            comps[k] = -comps[k]
    return mu, comps


def feature_traj(name, rot, axial_bases=None):
    """rot: (T,23,3) egocentric+causal-median pose. Returns (T, n_features)."""
    if name == 'full23':
        return None  # handled directly via pcs, not rot
    if name in AXIAL_VARIANTS:
        mu, comps = axial_bases[name]
        sub = rot[:, AXIAL_IDX, :].reshape(len(rot), -1)
        return (sub - mu) @ comps.T                              # (T, n_components)
    if name == 'snout_z':
        return rot[:, NODE_IDX['Snout'], 2:3]
    if name == 'bodyrot':
        return body_rotation_angle(rot)[:, None]
    if name == 'snout_bodyrot':
        return np.concatenate([rot[:, NODE_IDX['Snout'], 2:3],
                               body_rotation_angle(rot)[:, None]], axis=1)
    raise ValueError(name)


def rebuild_and_bounds(seg_a, seg_b):
    """seg_a/seg_b: (n, W, k) windows from two modalities at the SAME anchor
    frames. Returns (tmpl_a, tmpl_b, shared_bounds) -- mean of each, bounds =
    std(a) + std(b), mirroring analyse_dual_session's Pass 2 exactly."""
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


def process_session(rat, session, axial_bases, session_idx):
    raw = sc.get_raw_session(rat, session, want_dual=True)
    tm = tp.load_template(rat, session)
    if tm is None:
        return []
    d3 = du.dannce_aligned(raw)
    if d3 is None:
        return []
    n = min(len(raw['triang_3d']), len(d3))
    s_pcs, s_rot = ev.project_session_to_pcs(raw['triang_3d'][:n], tm, flip_z=True)
    d_pcs, d_rot = ev.project_session_to_pcs(d3[:n], tm, flip_z=False)
    pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)[:2]
    com = raw['triang_3d'][:n].mean(axis=1)
    theta_all = ARENA.theta(com[:, :2]); sec_all = sector_index(theta_all, SECTOR_W)

    # ---- Pass 1: SLEAP matches, bounds x1.25
    tmpl0 = tm['template'][:, :2]
    bounds0 = ev.template_bounds(tm) * 1.25
    m1, _ = ev.get_template_match_indices(s_pcs[:, pcu], tmpl0, bounds0,
                                          refractory_frames=REFRACTORY)
    if len(m1) == 0:
        return []
    m1 = ev.correct_matches(s_pcs, tmpl0, m1)

    # ---- rebuild DANNCE template from Pass-1 SLEAP matches (shared bounds)
    s_seg1, e1 = windows_at(s_pcs[:, pcu], m1)
    d_seg1, _ = windows_at(d_pcs[:, pcu], m1)
    if len(s_seg1) == 0:
        return []
    dannce_tmpl0, _, bounds1 = rebuild_and_bounds(d_seg1, s_seg1)

    # ---- ground-truth DANNCE matches
    gt, _ = ev.get_template_match_indices(d_pcs[:, pcu], dannce_tmpl0, bounds1,
                                          refractory_frames=REFRACTORY)
    if len(gt) == 0:
        return []
    gt = ev.correct_matches(d_pcs, dannce_tmpl0, gt)

    rows = []
    for feat in FEATURES:
        if feat == 'full23':
            s_traj, d_traj = s_pcs[:, pcu], d_pcs[:, pcu]
        else:
            s_traj = feature_traj(feat, s_rot, axial_bases)
            d_traj = feature_traj(feat, d_rot, axial_bases)

        s_seg, ok_e = windows_at(s_traj, gt)
        d_seg, _ = windows_at(d_traj, gt)
        if len(s_seg) < 5:
            continue
        sleap_tmpl, dannce_tmpl, bounds = rebuild_and_bounds(s_seg, d_seg)
        acc = ACCEPTED[feat]

        sm, _ = ev.get_template_match_indices(s_traj, sleap_tmpl, bounds,
                                              refractory_frames=REFRACTORY, accepted_errors=acc)
        dm, _ = ev.get_template_match_indices(d_traj, dannce_tmpl, bounds,
                                              refractory_frames=REFRACTORY, accepted_errors=acc)
        sm = ev.correct_matches(s_traj, sleap_tmpl, sm) if len(sm) else np.array([], dtype=int)
        dm = ev.correct_matches(d_traj, dannce_tmpl, dm) if len(dm) else np.array([], dtype=int)

        sleap_only, dannce_only, both = get_match_overlap(dm, sm)
        for frame in sleap_only:
            rows.append(dict(rat=rat, session=session, session_idx=session_idx, feature=feat,
                             category='sleap_only', frame=int(frame),
                             sector=int(sec_all[min(frame, n - 1)]),
                             theta=float(theta_all[min(frame, n - 1)])))
        for frame in dannce_only:
            rows.append(dict(rat=rat, session=session, session_idx=session_idx, feature=feat,
                             category='dannce_only', frame=int(frame),
                             sector=int(sec_all[min(frame, n - 1)]),
                             theta=float(theta_all[min(frame, n - 1)])))
        for frame in both:
            rows.append(dict(rat=rat, session=session, session_idx=session_idx, feature=feat,
                             category='both', frame=int(frame),
                             sector=int(sec_all[min(frame, n - 1)]),
                             theta=float(theta_all[min(frame, n - 1)])))
    return rows


def main():
    all_rows = []
    for rat in ['R4', 'R5', 'R6']:
        sessions = dual_sessions(rat)
        print(f'{rat}: {len(sessions)} dual-modality sessions', flush=True)
        axial_bases = {name: fit_axial_basis(rat, sessions, n_components=k)
                       for name, k in AXIAL_VARIANTS.items()}
        np.savez(f'{OUT}/axial_basis_{rat}.npz',
                mu1=axial_bases['axial'][0], comps1=axial_bases['axial'][1],
                mu2=axial_bases['axial2'][0], comps2=axial_bases['axial2'][1])
        for i, session in enumerate(sessions):
            rows = process_session(rat, session, axial_bases, i)
            all_rows.extend(rows)
            if i % 20 == 0:
                print(f'  [{i+1}/{len(sessions)}] {session}: {len(rows)} events so far this session',
                      flush=True)
        print(f'{rat}: done, {sum(1 for r in all_rows if r["rat"]==rat)} total events', flush=True)

    d = pd.DataFrame(all_rows)
    d.to_csv(f'{OUT}/stage18_events.csv', index=False)
    print(f'\nwrote {OUT}/stage18_events.csv: {len(d)} events')

    summ = d.groupby(['rat', 'session', 'session_idx', 'feature', 'category']).size().unstack(fill_value=0)
    for c in ['sleap_only', 'dannce_only', 'both']:
        if c not in summ.columns:
            summ[c] = 0
    summ = summ.reset_index()
    summ['precision'] = summ['both'] / (summ['both'] + summ['sleap_only']).replace(0, np.nan)
    summ['recall'] = summ['both'] / (summ['both'] + summ['dannce_only']).replace(0, np.nan)
    summ.to_csv(f'{OUT}/stage18_session_summary.csv', index=False)
    print(f'wrote {OUT}/stage18_session_summary.csv')

    print('\nPooled (all sessions), baseline feature (full23):')
    for rat, g in d[d.feature == 'full23'].groupby('rat'):
        c = g.category.value_counts()
        tp_, fp_, fn_ = c.get('both', 0), c.get('sleap_only', 0), c.get('dannce_only', 0)
        prec = tp_ / max(tp_ + fp_, 1); rec = tp_ / max(tp_ + fn_, 1)
        print(f'  {rat}: TP={tp_} FP={fp_} FN={fn_}  precision={prec:.3f}  recall={rec:.3f}')

    print('\nPooled (all sessions, all rats), by feature:')
    for feat, g in d.groupby('feature'):
        c = g.category.value_counts()
        tp_, fp_, fn_ = c.get('both', 0), c.get('sleap_only', 0), c.get('dannce_only', 0)
        prec = tp_ / max(tp_ + fp_, 1); rec = tp_ / max(tp_ + fn_, 1)
        print(f'  {feat:14s}: TP={tp_:5d} FP={fp_:5d} FN={fn_:5d}  precision={prec:.3f}  recall={rec:.3f}')


if __name__ == '__main__':
    main()
