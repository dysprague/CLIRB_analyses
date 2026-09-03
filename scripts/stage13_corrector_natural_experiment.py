"""Stage 13 — the corrector_ckpt discovery: was 'triang_3d' ever raw SLEAP?

While testing the §8c bone-length-bundle-adjustment hypothesis, direct
inspection of ~/campy-CLIRB/campy/behavior.py found that `correct_triangulations()`
is a dead stub (`return points_3d  # TODO: add triangulation corrections`) and
`optimize_with_bones()` (the real bundle-adjustment function) is never called
anywhere in the repo. Bone-length bundle adjustment CANNOT explain triang_3d's
low bone-length CV (0.118) vs raw DLT (0.366) -- it never runs online.

But behavior.py DOES run something else between triangulate() and the reward
rule: `new_points_3d = corrector.correct(keypoints_3D, ...)` when
`corrector_ckpt` is set. Grepping every session's `sleap/campy_config.yaml`
shows this checkpoint (a temporal MLP trained on R1/R2/R3 to map
SLEAP-space -> DANNCE-space) was ACTIVE for essentially the entire analysis
window (2026-06-09 to 2026-07-27/28) and then set to null starting 2026-07-29
onward -- a natural on/off experiment that happens to be already cached
locally (data/cache/<rat>/*.npz extends through 2026-08-26).

This script uses it to test: is the corrector (not bone-length bundle
adjustment) what actually explains triang_3d's rigidity?
"""
import os, sys
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial.quality import bone_length_cv
from config import EDGES

ON = ['2026_06_20_1', '2026_07_01_1', '2026_07_08_1', '2026_07_15_1', '2026_06_26_1']
OFF_CANDIDATES = ['2026_08_05_1', '2026_08_06_1', '2026_08_11_1', '2026_08_25_1',
                  '2026_08_26_1', '2026_08_25_2', '2026_08_27_1', '2026_08_28_1']


def cv_for(rat, session):
    c = sc.get_raw_session(rat, session, want_dual=False, verbose=False)
    if c is None:
        return None
    t3 = c['triang_3d'].astype(np.float64)
    com = t3[:, :, :2].mean(axis=1)
    rad = np.linalg.norm(com - np.median(com, axis=0), axis=1)
    ok = np.isfinite(t3).all(axis=(1, 2))
    if ok.sum() < 200:
        return None
    cv = bone_length_cv(t3[ok], edges=EDGES)
    return float(np.nanmedian(cv)), int(ok.sum())


def main():
    rows = []
    for rat in ['R4', 'R5', 'R6']:
        for sess in ON:
            r = cv_for(rat, sess)
            if r:
                rows.append(dict(rat=rat, session=sess, group='corrector_ON', bone_cv=r[0], n=r[1]))
        offs = [s for s in OFF_CANDIDATES if os.path.isfile(f'data/cache/{rat}/{s}.npz')]
        for sess in offs[:5]:
            r = cv_for(rat, sess)
            if r:
                rows.append(dict(rat=rat, session=sess, group='corrector_OFF', bone_cv=r[0], n=r[1]))

    d = pd.DataFrame(rows)
    d.to_csv('results/spatial_confound/corrector_onoff_bone_cv.csv', index=False)
    print(d.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n' + '=' * 70)
    print('MEDIAN BONE-LENGTH CV BY GROUP')
    print('=' * 70)
    summ = d.groupby(['rat', 'group']).bone_cv.median().unstack()
    print(summ.to_string(float_format=lambda v: f'{v:.4f}'))
    print('\npooled:')
    print(d.groupby('group').bone_cv.median())
    print('\n(reference from SPATIAL_CONFOUND_RESULTS.md §8c: stored triang_3d overall '
          '0.118, raw DLT re-triangulation 0.365-0.366)')
    d.to_csv('results/spatial_confound/corrector_onoff_bone_cv.csv', index=False)


if __name__ == '__main__':
    main()
