"""Stage 18b — why does body rotation angle win in stage17b but lose badly in
stage18? Three hypotheses, tested directly:

  A. Sample size / independence -- stage17b's small in-sample fit-then-score
     vs stage18's larger, independent, per-session rebuild.
  B. Bounds instability -- a per-session std(sleap)+std(dannce) bound for a
     single scalar, estimated from ~100 events, may be far noisier
     session-to-session than a pooled max-deviation bound from many sessions.
  C. Selection-criterion mismatch -- stage18's ground-truth anchor set is
     selected via the 2-PC (full23) space at every step. Axial correlates
     strongly with that space by construction (it was fit to). Body rotation
     is an independent signal never optimized to agree with PC0 -- so the
     "good" windows PC0 selects may not be the frames where body rotation
     itself is well-behaved.
"""
import os, sys, glob
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial import dual as du
from clirb.spatial.geometry import ARENA, sector_index
from config import NODE_IDX

sys.path.insert(0, REPO)
from scripts.stage18_dannce_ground_truth_overlap import (
    dual_sessions, fit_axial_basis, feature_traj, rebuild_and_bounds,
    windows_at, body_rotation_angle, REFRACTORY, W, SECTOR_W,
)

OUT = 'results/spatial_confound'
RNG = np.random.default_rng(0)


def main():
    corr_rows = []
    stab_rows = []
    dist_rows = []

    for rat in ['R4', 'R5', 'R6']:
        sessions = dual_sessions(rat)
        mu, comp = fit_axial_basis(rat, sessions)
        all_bodyrot, all_snout, all_pc0 = [], [], []

        per_session_templates = {f: [] for f in ['axial', 'snout_z', 'bodyrot']}

        for session in sessions:
            raw = sc.get_raw_session(rat, session, want_dual=True)
            tm = tp.load_template(rat, session)
            if tm is None:
                continue
            d3 = du.dannce_aligned(raw)
            if d3 is None:
                continue
            n = min(len(raw['triang_3d']), len(d3))
            s_pcs, s_rot = ev.project_session_to_pcs(raw['triang_3d'][:n], tm, flip_z=True)
            d_pcs, d_rot = ev.project_session_to_pcs(d3[:n], tm, flip_z=False)
            pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)[:2]

            # ---- Pass 1 + DANNCE rebuild + ground-truth matches (same as stage18)
            tmpl0 = tm['template'][:, :2]; bounds0 = ev.template_bounds(tm) * 1.25
            m1, _ = ev.get_template_match_indices(s_pcs[:, pcu], tmpl0, bounds0, refractory_frames=REFRACTORY)
            if len(m1) == 0:
                continue
            m1 = ev.correct_matches(s_pcs, tmpl0, m1)
            s_seg1, _ = windows_at(s_pcs[:, pcu], m1); d_seg1, _ = windows_at(d_pcs[:, pcu], m1)
            if len(s_seg1) == 0:
                continue
            dannce_tmpl0, _, bounds1 = rebuild_and_bounds(d_seg1, s_seg1)
            gt, _ = ev.get_template_match_indices(d_pcs[:, pcu], dannce_tmpl0, bounds1, refractory_frames=REFRACTORY)
            if len(gt) < 5:
                continue
            gt = ev.correct_matches(d_pcs, dannce_tmpl0, gt)

            # ---- (C) correlation of each feature with PC0, pooled over ALL frames
            bodyrot_full = body_rotation_angle(s_rot)
            snout_full = s_rot[:, NODE_IDX['Snout'], 2]
            all_bodyrot.append(bodyrot_full); all_snout.append(snout_full); all_pc0.append(s_pcs[:, pcu[0]])

            # ---- (B) per-session rebuilt template stability: build from the SLEAP
            # side at the ground-truth anchor frames, record mean level + bounds width
            for feat in ['axial', 'snout_z', 'bodyrot']:
                s_traj = feature_traj(feat, s_rot, mu, comp)
                d_traj = feature_traj(feat, d_rot, mu, comp)
                s_seg, e_ok = windows_at(s_traj, gt)
                d_seg, _ = windows_at(d_traj, gt)
                if len(s_seg) < 5:
                    continue
                s_tmpl, d_tmpl, bounds = rebuild_and_bounds(s_seg, d_seg)
                per_session_templates[feat].append(dict(
                    rat=rat, session=session, n_gt=len(s_seg),
                    sleap_mean_level=float(np.mean(s_tmpl)),
                    bounds_width=float(np.mean(bounds)),
                    sleap_std_at_gt=float(np.mean(s_seg.std(axis=0))),
                ))

        # ---- (C) correlation, pooled across this rat's sessions
        bodyrot_all = np.concatenate(all_bodyrot); snout_all = np.concatenate(all_snout)
        pc0_all = np.concatenate(all_pc0)
        r_body = np.corrcoef(bodyrot_all, pc0_all)[0, 1]
        r_snout = np.corrcoef(snout_all, pc0_all)[0, 1]
        corr_rows.append(dict(rat=rat, feature='bodyrot', corr_with_pc0=r_body))
        corr_rows.append(dict(rat=rat, feature='snout_z', corr_with_pc0=r_snout))
        print(f'{rat}: corr(bodyrot, PC0)={r_body:+.3f}   corr(snout_z, PC0)={r_snout:+.3f}')

        # ---- distribution sanity for bodyrot (wraparound / skew check)
        from scipy import stats as st
        dist_rows.append(dict(rat=rat, feature='bodyrot', mean=bodyrot_all.mean(),
                              std=bodyrot_all.std(), skew=float(st.skew(bodyrot_all)),
                              p01=np.percentile(bodyrot_all, 1), p99=np.percentile(bodyrot_all, 99),
                              near_0_frac=float((bodyrot_all < 5).mean()),
                              near_180_frac=float((bodyrot_all > 175).mean())))

        # ---- (B) session-to-session stability, this rat
        for feat in ['axial', 'snout_z', 'bodyrot']:
            t = pd.DataFrame(per_session_templates[feat])
            if len(t) < 3:
                continue
            cv_mean = t.sleap_mean_level.std() / (abs(t.sleap_mean_level.mean()) + 1e-9)
            cv_bounds = t.bounds_width.std() / (t.bounds_width.mean() + 1e-9)
            stab_rows.append(dict(rat=rat, feature=feat, n_sessions=len(t),
                                  mean_level_mean=t.sleap_mean_level.mean(),
                                  mean_level_cv=cv_mean, bounds_width_mean=t.bounds_width.mean(),
                                  bounds_width_cv=cv_bounds, median_n_gt=t.n_gt.median()))

    pd.DataFrame(corr_rows).to_csv(f'{OUT}/diag_feature_pc0_correlation.csv', index=False)
    pd.DataFrame(stab_rows).to_csv(f'{OUT}/diag_template_stability.csv', index=False)
    pd.DataFrame(dist_rows).to_csv(f'{OUT}/diag_bodyrot_distribution.csv', index=False)

    print('\n' + '=' * 78)
    print('HYPOTHESIS C: correlation of each feature with the selection criterion (PC0)')
    print('=' * 78)
    print(pd.DataFrame(corr_rows).pivot(index='feature', columns='rat', values='corr_with_pc0').to_string())

    print('\n' + '=' * 78)
    print('HYPOTHESIS B: session-to-session rebuilt-template stability (coefficient of variation)')
    print('=' * 78)
    sdf = pd.DataFrame(stab_rows)
    print(sdf.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    print('\n' + '=' * 78)
    print('BODY ROTATION distribution sanity (wraparound/skew check)')
    print('=' * 78)
    print(pd.DataFrame(dist_rows).to_string(index=False, float_format=lambda v: f'{v:.3f}'))


if __name__ == '__main__':
    main()
