"""Console + CSV summary tables for the post-recalibration evaluation."""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from scripts.newcal_eras import ERAS, ERA_SHORT, add_group, ordered_groups
OUT = 'results/newcal'

METRICS = [
    ('world_kp_med_mm',      'median keypoint error, all frames (mm)', 2),
    ('world_frame_medkp_p50','median per-frame MEDIAN-kp error (mm)', 2),
    ('world_frame_rmse_p50', 'median per-frame RMSE (mm)', 2),
    ('world_frame_rmse_p90', 'p90 per-frame RMSE (mm)', 1),
    ('world_rmse_mm',        'session RMS over all frames (mm)', 1),
    ('world_gross_frac',     'frames >40mm RMSE (%)', 2),
    ('flyaway_frac_any_kp',  'frames with >=1 kp >100mm off (%)', 2),
    ('ego_rmse_mm',          'egocentric RMSE (mm)', 1),
    ('com_pc0_rmse',         'PC0 RMSE (common basis)', 1),
    ('com_pc1_rmse',         'PC1 RMSE (common basis)', 1),
    ('com_pc0_r',            'PC0 SLEAP-DANNCE r', 3),
    ('com_pc1_r',            'PC1 SLEAP-DANNCE r', 3),
    ('com_pc0_nrmse',        'PC0 RMSE / SD(DANNCE)', 3),
    ('com_pc1_nrmse',        'PC1 RMSE / SD(DANNCE)', 3),
    ('com_pc0_err_over_bound', 'PC0 RMSE / template bound', 3),
    ('com_pc1_err_over_bound', 'PC1 RMSE / template bound', 3),
    ('fit_residual_mm',      'Procrustes fit residual (mm)', 2),
    ('fit_scale',            'Procrustes scale (DANNCE->SLEAP)', 4),
    ('best_frame_lag',       'best DANNCE/SLEAP frame lag', 1),
    ('torso_ratio_sleap_over_dannce', 'SpineF-TailBase SLEAP/DANNCE', 4),
    ('bone_frame_cv_sleap_p50',  'SLEAP bone-length CV (median frame)', 3),
    ('bone_frame_cv_dannce_p50', 'DANNCE bone-length CV (median frame)', 3),
    ('sleap_conf_mean',      'SLEAP 2D confidence', 3),
    ('sleap_reproj_px_p50',  'SLEAP 3D-vs-own-2D residual (px)', 2),
    ('dannce_reproj_px_p50', 'DANNCE vs SLEAP confident 2D (px)', 2),
    ('sleap_offimage_frac',  'frames with kp off-image (%)', 3),
]
PCT = {'world_gross_frac', 'flyaway_frac_any_kp', 'sleap_offimage_frac'}


def main():
    d = add_group(pd.read_csv(f'{OUT}/session_metrics.csv'), sep=' ')
    order = ordered_groups(d)

    rows = []
    for col, label, dp in METRICS:
        if col not in d.columns:
            continue
        r = {'metric': label}
        for g in order:
            v = d[d.grp == g][col].dropna().values
            if not len(v):
                r[g] = ''
                continue
            m = np.median(v)
            if col in PCT:
                m *= 100
            r[g] = round(float(m), dp)
        # pooled era medians
        for era in ERAS:
            v = d[d.era == era][col].dropna().values
            m = np.median(v) * (100 if col in PCT else 1) if len(v) else np.nan
            r[f'ALL {ERA_SHORT[era]}'] = round(float(m), dp) if len(v) else ''
        rows.append(r)

    t = pd.DataFrame(rows)
    cols = ['metric'] + order + [f'ALL {ERA_SHORT[e]}' for e in ERAS]
    t = t[cols]
    t.to_csv(f'{OUT}/summary_table.csv', index=False)

    n = d.groupby('grp').size().to_dict()
    nx = d[d.excluded].groupby('grp').size().to_dict()
    print('sessions per group:', {g: n.get(g, 0) for g in order})
    print('  of which flagged by the >60mm Procrustes/RMSE guard:',
          {g: nx.get(g, 0) for g in order})
    with pd.option_context('display.width', 250, 'display.max_columns', 40,
                           'display.max_colwidth', 40):
        print(t.to_string(index=False))
    print(f'\nwrote {OUT}/summary_table.csv')

    print('\nper-session detail (post-recalibration):')
    show = ['rat', 'session', 'n_frames', 'world_frame_rmse_p50', 'world_rmse_mm',
            'world_gross_frac', 'flyaway_frac_any_kp', 'com_pc0_rmse', 'com_pc1_rmse',
            'com_pc0_r', 'com_pc1_r', 'fit_residual_mm', 'fit_scale',
            'best_frame_lag', 'bone_frame_cv_sleap_p50', 'sleap_conf_mean']
    show = [c for c in show if c in d.columns]
    with pd.option_context('display.width', 250, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(d[d.era != 'sept_baseline'].sort_values(['era', 'rat', 'session'])[show].to_string(index=False))
        print('\nSeptember R5 sessions (the within-animal control):')
        print(d[(d.era == 'sept_baseline') & (d.rat == 'R5')][show].to_string(index=False))


if __name__ == '__main__':
    main()
