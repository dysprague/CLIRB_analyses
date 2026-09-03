"""Stage 17a — sector-bias remeasurement with (a) axial-keypoint-only PC0 and
(b) two hand-crafted single features, on the already-local, original-SLEAP-2D
dataset (`dataset_original.npz`, §8j/8k). No new SMB reads.

Candidates:
  full23   - baseline: the real per-rat template's own PC0 (all 23 keypoints)
  axial    - a fresh 1-component PCA fit on 9 axial/core keypoints only
             (Snout, EarL, EarR, SpineF, SpineM, SpineL, TailBase, HipL, HipR)
             -- everything EXCEPT the four limbs, chosen for having the lowest
             aspect_amp in stage16's mechanism table.
  snout_z  - egocentric height of Snout above SpineM (no fitting needed)
  bodyrot  - angle between (SpineF-SpineM) and (SpineM-TailBase), i.e. how
             arched/curled the spine is (rotation-invariant, no fitting needed)

For each, compute Delta(feature) = SLEAP - DANNCE by sector, by rat, and
compare the SECTOR-DEPENDENCE (range/CI) against the full-23kp baseline.

Writes results/spatial_confound/sector_bias_feature_comparison.csv
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb import events as ev, templates as tp
from config import NODE_IDX

OUT = 'results/spatial_confound'
AXIAL = ['Snout', 'EarL', 'EarR', 'SpineF', 'SpineM', 'SpineL', 'TailBase', 'HipL', 'HipR']
AXIAL_IDX = [NODE_IDX[n] for n in AXIAL]
NSEC = 12
RNG = np.random.default_rng(0)


def body_rotation_angle(pose):
    """pose: (T,23,3). Angle (deg) between SpineF-SpineM and SpineM-TailBase."""
    SF = pose[:, NODE_IDX['SpineF'], :]; SM = pose[:, NODE_IDX['SpineM'], :]
    TB = pose[:, NODE_IDX['TailBase'], :]
    v1 = SF - SM; v2 = SM - TB
    n1 = np.linalg.norm(v1, axis=-1); n2 = np.linalg.norm(v2, axis=-1)
    cos = (v1 * v2).sum(-1) / np.maximum(n1 * n2, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def sector_balanced_sample(sector, n_per_sector=4000, rng=RNG):
    idx = []
    for s in range(NSEC):
        ii = np.where(sector == s)[0]
        if len(ii):
            idx.extend(rng.choice(ii, min(len(ii), n_per_sector), replace=False))
    return np.array(idx)


def main():
    z = np.load(f'{OUT}/dataset_original.npz')
    XC, dn_cal, sector, rat_arr = z['XCAL'], z['DNCAL'], z['SECTOR'], z['RAT']

    rows = []
    for rat in ['R4', 'R5', 'R6']:
        sel = rat_arr == rat
        if sel.sum() < 500:
            continue
        tm = None
        import glob
        for s_ in sorted({os.path.basename(f)[:-4].split('__')[1]
                          for f in glob.glob(f'results/corrector_v2/keys2d/{rat}__*.npz')}):
            tm = tp.load_template(rat, s_)
            if tm is not None:
                break
        if tm is None:
            print(f'{rat}: no template'); continue

        Xr = XC[sel].astype(np.float64); Dr = dn_cal[sel].astype(np.float64); secr = sector[sel]

        # ---- candidate: full23 (baseline PC0, real template)
        s_pcs, s_rot = ev.project_session_to_pcs(Xr, tm, flip_z=True)
        d_pcs, d_rot = ev.project_session_to_pcs(Dr, tm, flip_z=False)
        pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)
        d_full = s_pcs[:, pcu[0]] - d_pcs[:, pcu[0]]

        # ---- candidate: axial-only PC0, fit fresh on a sector-balanced DANNCE sample
        fit_idx = sector_balanced_sample(secr)
        d_axial_sub = d_rot[fit_idx][:, AXIAL_IDX, :].reshape(len(fit_idx), -1)
        mu = d_axial_sub.mean(axis=0)
        pca = PCA(n_components=1).fit(d_axial_sub - mu)
        comp = pca.components_[0]
        # orient so it correlates positively with the real PC0 (i.e. "more reared = higher")
        proxy_corr = np.corrcoef(((d_rot[fit_idx][:, AXIAL_IDX, :].reshape(len(fit_idx), -1) - mu) @ comp),
                                 d_pcs[fit_idx, pcu[0]])[0, 1]
        if proxy_corr < 0:
            comp = -comp
        s_axial = (s_rot[:, AXIAL_IDX, :].reshape(len(s_rot), -1) - mu) @ comp
        d_axial = (d_rot[:, AXIAL_IDX, :].reshape(len(d_rot), -1) - mu) @ comp
        d_axial_delta = s_axial - d_axial
        print(f'{rat}: axial-PC0 corr with real PC0 (post-orient) = {abs(proxy_corr):.3f}')

        # ---- candidate: snout height (egocentric z of Snout)
        s_snout = s_rot[:, NODE_IDX['Snout'], 2]; d_snout = d_rot[:, NODE_IDX['Snout'], 2]
        d_snout_delta = s_snout - d_snout

        # ---- candidate: body rotation angle
        s_ang = body_rotation_angle(s_rot); d_ang = body_rotation_angle(d_rot)
        d_ang_delta = s_ang - d_ang

        for name, delta in [('full23', d_full), ('axial', d_axial_delta),
                            ('snout_z', d_snout_delta), ('bodyrot', d_ang_delta)]:
            for s in range(NSEC):
                m = (secr == s) & np.isfinite(delta)
                if m.sum() < 30:
                    continue
                boot = [np.mean(RNG.choice(delta[m], m.sum(), replace=True)) for _ in range(500)]
                rows.append(dict(rat=rat, feature=name, sector=s, theta=s * 30 - 165,
                                 n=int(m.sum()), mu=float(np.mean(delta[m])),
                                 lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5))))

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/sector_bias_feature_comparison.csv', index=False)

    print('\n' + '=' * 78)
    print('SECTOR-DEPENDENCE (range of per-sector mean, and normalized range) BY FEATURE')
    print('=' * 78)
    summ = []
    for (rat, feat), g in d.groupby(['rat', 'feature']):
        rng_ = g.mu.max() - g.mu.min()
        scale = g.mu.abs().median() + 1e-9
        summ.append(dict(rat=rat, feature=feat, range=rng_, sd_across_sectors=g.mu.std(),
                         norm_range=rng_ / scale))
    s = pd.DataFrame(summ)
    print(s.pivot(index='feature', columns='rat', values='range').round(2).to_string())
    print('\n(normalized by median |mu|, so scales are comparable across features)')
    print(s.pivot(index='feature', columns='rat', values='norm_range').round(2).to_string())
    print(f'\nwrote {OUT}/sector_bias_feature_comparison.csv')


if __name__ == '__main__':
    main()
