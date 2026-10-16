"""Stage 25 -- does angle_skip_xy (and the other turn candidates) correlate
with PC2/PC3 of the real production template's PCA basis? And what do PC2/
PC3's own loadings look like -- are they interpretable as "turnedness"?

Both angle_skip_xy and PC2/PC3 are computed from the SAME representation
(`ev.project_session_to_pcs`'s "rotated_med" output -- egocentric-normalized,
z-flipped for SLEAP, 5-frame causal median filtered -- exactly what the
online rule itself uses), so this is a direct, frame-matched comparison, not
confounded by the world-space-vs-egocentric distinction that mattered for
SLEAP-vs-DANNCE reliability (stage20+). Computed independently for SLEAP and
for DANNCE (within-modality correlation -- does the relationship hold up
using each modality's own self-consistent measurement).

Part 1 -- correlation: for every angle feature (all 8 from stage21-24),
Pearson r against PC2 and PC3, per session, for SLEAP and for DANNCE
separately. Also a joint R^2 (angle ~ PC2 + PC3, OLS) since the target
"turnedness" construct might be a genuine linear combination of both.

Part 2 -- PC2/PC3 loadings: for every DISTINCT template file actually used
across the analysis session set, decompose pc_weights[2] and pc_weights[3]
(each 69-dim = 23 keypoints x xyz) into per-keypoint contribution magnitude,
to see which keypoints/axes dominate and whether that's consistent with a
"body turned left/right" interpretation.

Writes:
  results/spatial_confound/stage25_pc_correlations.csv
  results/spatial_confound/stage25_pc_loadings.csv
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial import dual as du
from config import NODE_IDX

OUT = 'results/spatial_confound'

SNOUT, EARL, EARR = NODE_IDX['Snout'], NODE_IDX['EarL'], NODE_IDX['EarR']
SPINEF, SPINEM, SPINEL, TAILBASE = (NODE_IDX['SpineF'], NODE_IDX['SpineM'],
                                    NODE_IDX['SpineL'], NODE_IDX['TailBase'])
HIPL, HIPR, SHOULDERL, SHOULDERR = (NODE_IDX['HipL'], NODE_IDX['HipR'],
                                    NODE_IDX['ShoulderL'], NODE_IDX['ShoulderR'])
NODE_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]


def signed_angle_xy(v1, v2):
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]
    return np.degrees(np.arctan2(cross, dot))


def compute_angles(rot):
    """rot: (T, 23, 3) egocentric rotated+median keypoints (one modality)."""
    snout, earl, earr = rot[:, SNOUT], rot[:, EARL], rot[:, EARR]
    spinef, spinem, spinel, tb = rot[:, SPINEF], rot[:, SPINEM], rot[:, SPINEL], rot[:, TAILBASE]
    hipl, hipr, shoulderl, shoulderr = rot[:, HIPL], rot[:, HIPR], rot[:, SHOULDERL], rot[:, SHOULDERR]
    ear_mean = (earl + earr) / 2.0
    return {
        'angle_lower_xy': signed_angle_xy((spinem - tb)[:, :2], (spinef - spinem)[:, :2]),
        'angle_upper_xy': signed_angle_xy((spinef - spinem)[:, :2], (snout - spinef)[:, :2]),
        'angle_mid_xy': signed_angle_xy((spinem - spinel)[:, :2], (spinef - spinem)[:, :2]),
        'angle_skip_xy': signed_angle_xy((spinef - spinel)[:, :2], (snout - spinef)[:, :2]),
        'angle_torsion_xy': signed_angle_xy((hipr - hipl)[:, :2], (shoulderr - shoulderl)[:, :2]),
        'angle_body_head_xy': signed_angle_xy((spinef - tb)[:, :2], (ear_mean - spinef)[:, :2]),
        'angle_mid_head_xy': signed_angle_xy((spinem - spinel)[:, :2], (ear_mean - spinem)[:, :2]),
        'angle_torsion_ear_xy': signed_angle_xy((hipr - hipl)[:, :2], (earr - earl)[:, :2]),
    }


def ols_r2(y, X):
    """R^2 of y ~ X (X already has an intercept column appended)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-9), beta


def main():
    sessions_by_rat = {rat: [] for rat in ['R4', 'R5', 'R6']}
    import glob
    for rat in ['R4', 'R5', 'R6']:
        for f in sorted(glob.glob(f'data/cache/{rat}/*.npz')):
            session = os.path.basename(f)[:-4]
            raw = sc.get_raw_session(rat, session, want_dual=True)
            if raw is not None and raw.get('has_dual', False):
                sessions_by_rat[rat].append(session)

    corr_rows = []
    templates_seen = {}
    angle_names = ['angle_lower_xy', 'angle_upper_xy', 'angle_mid_xy', 'angle_skip_xy',
                  'angle_torsion_xy', 'angle_body_head_xy', 'angle_mid_head_xy',
                  'angle_torsion_ear_xy']

    for rat, sessions in sessions_by_rat.items():
        print(f'{rat}: {len(sessions)} dual sessions', flush=True)
        for i, session in enumerate(sessions):
            raw = sc.get_raw_session(rat, session, want_dual=True)
            tm = tp.load_template(rat, session)
            if tm is None:
                continue
            d3 = du.dannce_aligned(raw)
            if d3 is None:
                continue
            n = min(len(raw['triang_3d']), len(d3))
            if n < 100:
                continue
            s_pcs, s_rot = ev.project_session_to_pcs(raw['triang_3d'][:n], tm, flip_z=True)
            d_pcs, d_rot = ev.project_session_to_pcs(d3[:n], tm, flip_z=False)

            sl_angles = compute_angles(s_rot)
            dn_angles = compute_angles(d_rot)

            for modality, angles, pcs in [('SLEAP', sl_angles, s_pcs), ('DANNCE', dn_angles, d_pcs)]:
                pc2, pc3 = pcs[:, 2], pcs[:, 3]
                X = np.stack([pc2, pc3, np.ones_like(pc2)], axis=1)
                for name in angle_names:
                    y = angles[name]
                    r2, beta = ols_r2(y, X)
                    r_pc2 = np.corrcoef(y, pc2)[0, 1]
                    r_pc3 = np.corrcoef(y, pc3)[0, 1]
                    corr_rows.append(dict(rat=rat, session=session, modality=modality,
                                          feature=name, r_pc2=r_pc2, r_pc3=r_pc3,
                                          r2_pc2_pc3=r2, beta_pc2=beta[0], beta_pc3=beta[1],
                                          n_frames=n, template_file=tm['_file']))

            templates_seen[tm['_file']] = tm['pc_weights']
            if i % 20 == 0:
                print(f'  [{i+1}/{len(sessions)}] {session} done', flush=True)

    corr = pd.DataFrame(corr_rows)
    corr.to_csv(f'{OUT}/stage25_pc_correlations.csv', index=False)
    print(f'\nwrote {OUT}/stage25_pc_correlations.csv ({len(corr)} rows, '
         f'{len(templates_seen)} distinct templates)', flush=True)

    print('\n=== Median |r| with PC2/PC3, and joint R^2, by feature (pooled SLEAP+DANNCE) ===')
    summ = corr.groupby('feature').agg(
        median_abs_r_pc2=('r_pc2', lambda x: np.median(np.abs(x))),
        median_abs_r_pc3=('r_pc3', lambda x: np.median(np.abs(x))),
        median_r2=('r2_pc2_pc3', 'median'),
    ).sort_values('median_r2', ascending=False)
    print(summ.to_string(float_format=lambda v: f'{v:.3f}'))

    # --- Part 2: PC2/PC3 loadings for every distinct template ---
    load_rows = []
    for fname, pcw in templates_seen.items():
        for pc_idx in [2, 3]:
            loading = pcw[pc_idx].reshape(len(NODE_NAMES), 3)
            mag = np.linalg.norm(loading, axis=1)
            order = np.argsort(-mag)
            for rank, kp_idx in enumerate(order):
                load_rows.append(dict(template_file=fname, pc=f'PC{pc_idx}', rank=rank,
                                      keypoint=NODE_NAMES[kp_idx], magnitude=mag[kp_idx],
                                      x=loading[kp_idx, 0], y=loading[kp_idx, 1], z=loading[kp_idx, 2]))
    loadings = pd.DataFrame(load_rows)
    loadings.to_csv(f'{OUT}/stage25_pc_loadings.csv', index=False)
    print(f'\nwrote {OUT}/stage25_pc_loadings.csv ({len(templates_seen)} templates)')

    print('\n=== Top 6 keypoints by loading magnitude, PC2 vs PC3, per template ===')
    for fname in templates_seen:
        print(f'\n-- {fname} --')
        for pc in ['PC2', 'PC3']:
            top = loadings[(loadings.template_file == fname) & (loadings.pc == pc)].head(6)
            print(f'  {pc}:')
            for row in top.itertuples():
                print(f'    {row.keypoint:10s} mag={row.magnitude:.3f}  (x={row.x:+.3f} y={row.y:+.3f} z={row.z:+.3f})')


if __name__ == '__main__':
    main()
