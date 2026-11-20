"""September baseline (R4/R5/R6): SLEAP-vs-DANNCE agreement in keypoint and PC space.

Pre-processing follows multi_session.ipynb's `analyse_dual_session` exactly:

    SLEAP  : flip z -> normalize_skeleton_batch -> 5-frame causal median
    DANNCE : (no flip)  -> normalize_skeleton_batch -> 5-frame causal median

i.e. the raw keypoints are NOT median filtered; the causal median goes AFTER
egocentric normalisation, which is the order the online rule applies it.  The
notebook's shim calls `causal_median_5_kp(..., warmup='pad')`; its own comment
says new code should use `warmup='online'` (the rule's real behaviour), and the
two differ only on frames 1-3, far before the first checkable window.  This
uses 'online'.

Egocentric normalisation centres on SpineM and rotates the SpineF-SpineM axis
onto +X, so SpineM is identically zero in both modalities and carries no error
by construction.  It is reported as 0 and excluded from aggregate statistics.

PC SPACE.  Both modalities are projected through ONE common basis
(R5_template_8's `pc_weights` / `feature_means`) rather than each session's own
production template.  A per-session basis would make "PC3" a different axis in
every session, so the across-animal aggregate the grant figure shows would be
meaningless.  Per-session native-basis numbers are computed alongside as a
robustness check and written to the CSV.

QUALITY GATE.  Four sessions recorded on 2026_09_08 (R4_1, R5_1, R6_1, R6_2)
contain acquisition dropouts: their MEDIAN keypoint error is normal (16-31 mm)
but a small number of frames are off by metres, so the RMSE blows up to
1070-22427 mm.  Sessions are flagged `qc_pass = frame_rmse_p90 <= 100 mm`,
which separates them by more than a factor of ten (clean sessions sit at
25-40 mm, the four failures at 386-1715 mm) rather than by a judgement call.
The CSVs keep every session; aggregates and figures use `qc_pass` only.

Two scorings are reported:
  whole session   every frame -- "how well does SLEAP reproduce DANNCE's pose"
  match windows   only the 30-frame template-match windows, the notebook's
                  convention -- "how well does it reproduce it where the
                  closed-loop rule is deciding"

Writes results/grant_sept/*.csv
"""
import os, sys, glob
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb import templates as tp
from skeleton import normalize_skeleton_batch, project_to_pcs
from config import DATA_ROOT, NODE_IDX

OUT = 'results/grant_sept'
os.makedirs(OUT, exist_ok=True)

RATS = ['R4', 'R5', 'R6']
N_PC = 5
W = 30                                   # template length, the online rule's
QC_P90_MM = 100.0                        # acquisition-dropout gate
COMMON_TEMPLATE = f'{DATA_ROOT}/R5/templates/R5_template_8.npz'
KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]
SPINEM = NODE_IDX['SpineM']


def september_sessions():
    out = []
    for rat in RATS:
        for s in sorted(os.listdir(f'{DATA_ROOT}/{rat}')):
            if not s.startswith('2026_09'):
                continue
            if os.path.isfile(f'{DATA_ROOT}/{rat}/{s}/processed/sleap_dannce_keys.mat'):
                out.append((rat, s))
    return out


def load_basis(path):
    z = np.load(path, allow_pickle=True)
    return dict(pc_weights=z['pc_weights'], feature_means=z['feature_means'])


def prep(raw):
    """The multi_session.ipynb pipeline. Returns (sleap_ego, dannce_ego) in mm."""
    sl = np.asarray(raw['sleap_keys_3D'], np.float32)
    dn = np.asarray(raw['dannce_keys_3D'])
    dn = (dn.squeeze(axis=1).transpose(0, 2, 1) if dn.ndim == 4
          else np.transpose(dn, [0, 2, 1]))
    idx = np.asarray(raw['dannce_idx_for_sleap_cams'], int).ravel()[1:]
    if idx.size == 0:
        return None
    dn = dn[np.clip(idx, 0, len(dn) - 1)].astype(np.float32)
    n = min(len(sl), len(dn))
    sl, dn = sl[:n], dn[:n]

    sl_f = sl.copy(); sl_f[:, :, 2] = -sl_f[:, :, 2]          # SLEAP only
    sl_ego = ev.causal_median_5_kp(normalize_skeleton_batch(sl_f)[0],
                                   warmup='online')
    dn_ego = ev.causal_median_5_kp(normalize_skeleton_batch(dn)[0],
                                   warmup='online')
    return np.asarray(sl_ego, np.float32), np.asarray(dn_ego, np.float32)


def match_windows(sl_pcs, dn_pcs, template):
    """The notebook's stage-1..3 rebuild, reduced to what we need: the union of
    SLEAP and DANNCE match end-frames under rebuilt templates."""
    t = np.load(template['path'], allow_pickle=True)
    pc_t = t['template'][:, :2]
    bounds = t['bounds_to_use']
    m1, _ = ev.get_template_match_indices(sl_pcs[:, :2], pc_t, bounds * 1.25,
                                          refractory_frames=W)
    m1 = ev.correct_matches(sl_pcs, pc_t, m1)
    if len(m1) == 0:
        return np.array([], int)
    s1 = np.array([sl_pcs[m - W + 1:m + 1, :2] for m in m1 if m - W + 1 >= 0])
    d1 = np.array([dn_pcs[m - W + 1:m + 1, :2] for m in m1 if m - W + 1 >= 0])
    if not len(s1):
        return np.array([], int)
    gt, _ = ev.get_template_match_indices(dn_pcs[:, :2], d1.mean(axis=0),
                                          s1.std(axis=0) + d1.std(axis=0),
                                          refractory_frames=W)
    gt = ev.correct_matches(dn_pcs, d1.mean(axis=0), gt)
    if len(gt) == 0:
        return np.array([], int)
    ss = np.array([sl_pcs[m - W + 1:m + 1, :2] for m in gt if m - W + 1 >= 0])
    dd = np.array([dn_pcs[m - W + 1:m + 1, :2] for m in gt if m - W + 1 >= 0])
    if not len(ss):
        return np.array([], int)
    rb = ss.std(axis=0) + dd.std(axis=0)
    ms, _ = ev.get_template_match_indices(sl_pcs[:, :2], ss.mean(axis=0), rb,
                                          refractory_frames=W)
    md, _ = ev.get_template_match_indices(dn_pcs[:, :2], dd.mean(axis=0), rb,
                                          refractory_frames=W)
    ms = ev.correct_matches(sl_pcs, ss.mean(axis=0), ms)
    md = ev.correct_matches(dn_pcs, dd.mean(axis=0), md)
    return np.unique(np.concatenate([ms, md]).astype(int))


def window_mask(ends, n):
    m = np.zeros(n, bool)
    for e in ends:
        s = max(0, int(e) - W + 1)
        if int(e) < n:
            m[s:int(e) + 1] = True
    return m


def main():
    basis = load_basis(COMMON_TEMPLATE)
    sessions = september_sessions()
    print(f'{len(sessions)} September sessions with dual data', flush=True)

    kp_rows, pc_rows, sess_rows = [], [], []
    frame_store = {}

    for i, (rat, s) in enumerate(sessions, 1):
        raw = sc.get_raw_session(rat, s, want_dual=True)
        if raw is None or not raw.get('has_dual', False):
            print(f'[{i}/{len(sessions)}] {rat}/{s}: no dual data', flush=True)
            continue
        out = prep(raw)
        if out is None:
            print(f'[{i}/{len(sessions)}] {rat}/{s}: no alignment index', flush=True)
            continue
        sl_ego, dn_ego = out
        n = len(sl_ego)

        # --- keypoint space -------------------------------------------------
        d = np.linalg.norm(sl_ego - dn_ego, axis=-1)            # (n, 23) mm
        frame_rmse = np.sqrt((np.delete(d, SPINEM, axis=1) ** 2).mean(axis=1))

        # --- PC space, common basis ----------------------------------------
        sl_pcs = project_to_pcs(sl_ego, basis['pc_weights'], basis['feature_means'])
        dn_pcs = project_to_pcs(dn_ego, basis['pc_weights'], basis['feature_means'])
        pc_err = sl_pcs[:, :N_PC] - dn_pcs[:, :N_PC]

        # --- native-basis robustness check ----------------------------------
        nat_rmse = np.full(N_PC, np.nan)
        tpath = tp.find_template_file(rat, s)
        if tpath:
            try:
                z = np.load(tpath, allow_pickle=True)
                if z['pc_weights'].shape[0] >= N_PC:
                    a = project_to_pcs(sl_ego, z['pc_weights'], z['feature_means'])
                    b = project_to_pcs(dn_ego, z['pc_weights'], z['feature_means'])
                    nat_rmse = np.sqrt(((a[:, :N_PC] - b[:, :N_PC]) ** 2).mean(axis=0))
            except Exception:
                pass

        # --- match windows (the notebook's scoring domain) -------------------
        try:
            ends = match_windows(sl_pcs, dn_pcs, {'path': tpath}) if tpath else np.array([], int)
        except Exception:
            ends = np.array([], int)
        mmask = window_mask(ends, n)

        for j, name in enumerate(KP_NAMES):
            rec = dict(rat=rat, session=s, keypoint=name, n_frames=n,
                       mse_mm2=float((d[:, j] ** 2).mean()),
                       rmse_mm=float(np.sqrt((d[:, j] ** 2).mean())),
                       median_mm=float(np.median(d[:, j])),
                       p90_mm=float(np.percentile(d[:, j], 90)))
            rec['rmse_mm_match'] = (float(np.sqrt((d[mmask, j] ** 2).mean()))
                                    if mmask.sum() > W else np.nan)
            kp_rows.append(rec)

        for k in range(N_PC):
            dn_sd = float(dn_pcs[:, k].std())
            pc_rows.append(dict(
                rat=rat, session=s, pc=k + 1, n_frames=n,
                mse=float((pc_err[:, k] ** 2).mean()),
                rmse=float(np.sqrt((pc_err[:, k] ** 2).mean())),
                dannce_sd=dn_sd,
                rmse_frac_sd=float(np.sqrt((pc_err[:, k] ** 2).mean()) / dn_sd)
                             if dn_sd > 0 else np.nan,
                pearson_r=float(np.corrcoef(sl_pcs[:, k], dn_pcs[:, k])[0, 1]),
                rmse_match=(float(np.sqrt((pc_err[mmask, k] ** 2).mean()))
                            if mmask.sum() > W else np.nan),
                rmse_native=float(nat_rmse[k]),
            ))

        sess_rows.append(dict(
            rat=rat, session=s, n_frames=n,
            kp_rmse_mm=float(np.sqrt((np.delete(d, SPINEM, axis=1) ** 2).mean())),
            kp_median_mm=float(np.median(np.delete(d, SPINEM, axis=1))),
            frame_rmse_median=float(np.median(frame_rmse)),
            frame_rmse_p90=float(np.percentile(frame_rmse, 90)),
            n_match_windows=int(len(ends)),
            frac_frames_in_match=float(mmask.mean()),
            template=os.path.basename(tpath) if tpath else ''))
        frame_store[(rat, s)] = frame_rmse.astype(np.float32)
        print(f'[{i}/{len(sessions)}] {rat}/{s}: n={n} kp RMSE '
              f'{sess_rows[-1]["kp_rmse_mm"]:.2f} mm, {len(ends)} match windows',
              flush=True)

    kp = pd.DataFrame(kp_rows); pc = pd.DataFrame(pc_rows); se = pd.DataFrame(sess_rows)
    se['qc_pass'] = se.frame_rmse_p90 <= QC_P90_MM
    qc = se.set_index(['rat', 'session']).qc_pass
    for f in (kp, pc):
        f['qc_pass'] = [qc.loc[(r, s)] for r, s in zip(f.rat, f.session)]
    bad = se[~se.qc_pass]
    if len(bad):
        print(f'\nQC: excluding {len(bad)} session(s) with per-frame RMSE p90 > '
              f'{QC_P90_MM} mm (acquisition dropouts):', flush=True)
        print(bad[['rat', 'session', 'kp_rmse_mm', 'kp_median_mm',
                   'frame_rmse_p90']].round(1).to_string(index=False))
    kp, pc, se_all = kp[kp.qc_pass], pc[pc.qc_pass], se
    se = se[se.qc_pass]
    kp.to_csv(f'{OUT}/keypoint_error.csv', index=False)
    pc.to_csv(f'{OUT}/pc_error.csv', index=False)
    se_all.to_csv(f'{OUT}/session_summary.csv', index=False)
    np.savez_compressed(f'{OUT}/frame_rmse.npz',
                        **{f'{r}|{s}': v for (r, s), v in frame_store.items()})

    pd.set_option('display.width', 220)
    print(f'\n{"="*86}\nSESSIONS\n{"="*86}')
    print(se.groupby('rat').agg(sessions=('session', 'count'),
                                frames=('n_frames', 'sum'),
                                kp_rmse_mm=('kp_rmse_mm', 'median'),
                                frame_p90=('frame_rmse_p90', 'median')).round(3).to_string())

    print(f'\n{"="*86}\nKEYPOINT ERROR, pooled over every frame of every session (mm)\n{"="*86}')
    g = kp.groupby('keypoint').apply(
        lambda t: pd.Series({
            'rmse_mm': np.sqrt(np.average(t.mse_mm2, weights=t.n_frames)),
            'median_mm': np.average(t.median_mm, weights=t.n_frames),
            'p90_mm': np.average(t.p90_mm, weights=t.n_frames)}),
        include_groups=False).sort_values('rmse_mm')
    print(g.round(3).to_string())
    pooled = np.sqrt(np.average(kp[kp.keypoint != 'SpineM'].mse_mm2,
                                weights=kp[kp.keypoint != 'SpineM'].n_frames))
    print(f'\n  all keypoints except SpineM: RMSE {pooled:.3f} mm')

    print(f'\n{"="*86}\nPC ERROR, first {N_PC} PCs, common basis (R5_template_8)\n{"="*86}')
    gp = pc.groupby('pc').apply(
        lambda t: pd.Series({
            'rmse': np.sqrt(np.average(t.mse, weights=t.n_frames)),
            'dannce_sd': np.average(t.dannce_sd, weights=t.n_frames),
            'rmse_pct_of_sd': 100 * np.sqrt(np.average(t.mse, weights=t.n_frames))
                              / np.average(t.dannce_sd, weights=t.n_frames),
            'pearson_r': np.average(t.pearson_r, weights=t.n_frames),
            'rmse_match': np.sqrt(np.average(t.rmse_match.fillna(0) ** 2,
                                             weights=t.n_frames)),
            'rmse_native_median': t.rmse_native.median()}),
        include_groups=False)
    print(gp.round(3).to_string())

    print(f'\n{"="*86}\nPER RAT\n{"="*86}')
    for metric, frame, col in [('keypoint RMSE (mm)', kp, 'mse_mm2'),
                               ('PC RMSE', pc, 'mse')]:
        sub = frame[frame.keypoint != 'SpineM'] if 'keypoint' in frame else frame
        key = ['rat'] if 'keypoint' in frame else ['rat', 'pc']
        t = sub.groupby(key).apply(
            lambda t: np.sqrt(np.average(t[col], weights=t.n_frames)),
            include_groups=False)
        print(f'\n  {metric}:'); print(t.round(3).to_string())

    print(f'\nwrote {OUT}/keypoint_error.csv, pc_error.csv, session_summary.csv')


if __name__ == '__main__':
    main()
