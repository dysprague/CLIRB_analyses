"""Stage 8 — mechanism tests M1 / M2 / M4-fix.

A. WORLD vs EGOCENTRIC bias decomposition (the M1 test).
   Rotating zero-mean noise cannot create a mean bias, so M1 requires a
   systematic WORLD-frame bias vector whose EGOCENTRIC expression rotates with
   heading. Prediction:
       b_world(theta)  ~ CONSTANT across sectors
       b_ego(theta)    ~ varies, and ~ R(-heading(theta)) @ b_world
   The competing accounts (M2/M6, a genuinely position-dependent tracking error)
   predict b_world itself varies with theta.

B. TRIANGULATION CONDITION NUMBER + principal error axis per arena position
   (what reprojection residual is structurally blind to).

C. ROTATION-INVARIANT features: does the SLEAP-DANNCE discrepancy lose its
   sector dependence when expressed as pairwise distances instead of
   egocentric coordinates?
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from scipy import stats

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial import dual as du
from clirb.spatial import quality as q
from clirb.spatial.geometry import ARENA, sector_index
from config import NODE_IDX
from skeleton import normalize_skeleton_batch
from projection import project_3d_to_2d

OUT = 'results/spatial_confound'
SECTOR_W, NSEC = 30, 12
res = {}


# ===================================================================== A
def part_A():
    print('=' * 76)
    print('A. WORLD-frame vs EGOCENTRIC bias  (the M1 test)')
    print('=' * 76)
    allrows = []
    for rat in ['R4', 'R5', 'R6']:
        W_sum = np.zeros((NSEC, 3)); E_sum = np.zeros((NSEC, 3))
        H_sum = np.zeros(NSEC, dtype=complex); N = np.zeros(NSEC)
        for f in sorted(glob.glob(f'data/cache/{rat}/*.npz')):
            session = os.path.basename(f)[:-4]
            raw = sc.get_raw_session(rat, session, want_dual=True)
            if raw is None or not raw.get('has_dual', False):
                continue
            t = tp.load_template(rat, session)
            if t is None:
                continue
            d3 = du.dannce_aligned(raw)
            if d3 is None:
                continue
            s3 = raw['triang_3d'].astype(np.float32).copy()
            s3[:, :, 2] = -s3[:, :, 2]              # SLEAP z-flip (pipeline)
            n = min(len(s3), len(d3))
            s3, d3 = s3[:n], d3[:n].astype(np.float32)

            pcu = np.asarray(t['pcs_to_use']).ravel().astype(int)
            s_pcs, _ = ev.project_session_to_pcs(raw['triang_3d'][:n], t, flip_z=True)
            rears = [r for r in ev.detect_rears(s_pcs[:, pcu[0]]) if r[1] < n]
            if not rears:
                continue
            mask = np.zeros(n, bool)
            for a, b, _na in rears:
                mask[a:b + 1] = True
            com = s3.mean(axis=1)
            th = ARENA.theta(com[:, :2]); rad = ARENA.radius_of(com[:, :2])
            mask &= rad > 250
            if mask.sum() < 50:
                continue
            sec = sector_index(th, SECTOR_W)

            # world-frame bias (mm), averaged over keypoints
            bw = (s3 - d3).mean(axis=1)                      # (n,3)
            # egocentric bias: each modality rotated by ITS OWN heading, as the
            # pipeline does
            se, _, _ = normalize_skeleton_batch(s3)
            de, _, _ = normalize_skeleton_batch(d3)
            be = (se - de).mean(axis=1)                       # (n,3)
            SF = s3[:, NODE_IDX['SpineF'], :2]; SM = s3[:, NODE_IDX['SpineM'], :2]
            head = np.arctan2(SF[:, 1] - SM[:, 1], SF[:, 0] - SM[:, 0])

            for k in range(NSEC):
                m = mask & (sec == k)
                if m.sum() < 20:
                    continue
                W_sum[k] += bw[m].sum(axis=0); E_sum[k] += be[m].sum(axis=0)
                H_sum[k] += np.exp(1j * head[m]).sum(); N[k] += m.sum()
        ok = N >= 200
        Wm = W_sum[ok] / N[ok, None]; Em = E_sum[ok] / N[ok, None]
        Hm = np.angle(H_sum[ok] / N[ok])
        th_c = (np.arange(NSEC) * 30 - 165)[ok]
        d = pd.DataFrame(dict(rat=rat, theta=th_c, n=N[ok].astype(int),
                              bw_x=Wm[:, 0], bw_y=Wm[:, 1], bw_z=Wm[:, 2],
                              be_x=Em[:, 0], be_y=Em[:, 1], be_z=Em[:, 2],
                              heading=np.degrees(Hm)))
        allrows.append(d)
        # spatial variability of each, in mm
        sd_w = np.hypot(d.bw_x.std(), d.bw_y.std())
        sd_e = np.hypot(d.be_x.std(), d.be_y.std())
        print(f'\n{rat}  ({int(N[ok].sum()):,} rear-wall frames, {ok.sum()} sectors)')
        print(d[['theta', 'n', 'bw_x', 'bw_y', 'be_x', 'be_y', 'heading']].to_string(
            index=False, float_format=lambda v: f'{v:.2f}'))
        print(f'   across-sector SD of WORLD bias (xy)      = {sd_w:.2f} mm')
        print(f'   across-sector SD of EGOCENTRIC bias (xy) = {sd_e:.2f} mm')
        print(f'   ratio ego/world = {sd_e/max(sd_w,1e-9):.2f}')
        res[f'{rat}_sd_world_mm'] = float(sd_w); res[f'{rat}_sd_ego_mm'] = float(sd_e)

        # quantitative prediction: rotate the CONSTANT world bias by the sector's
        # mean heading -> predicted egocentric bias
        b0 = np.array([d.bw_x.mean(), d.bw_y.mean()])
        pred = np.stack([[np.cos(-h) * b0[0] - np.sin(-h) * b0[1],
                          np.sin(-h) * b0[0] + np.cos(-h) * b0[1]]
                         for h in np.radians(d.heading)])
        for ax, nm in [(0, 'x'), (1, 'y')]:
            r, p = stats.pearsonr(pred[:, ax], d[f'be_{nm}'])
            print(f'   predicted vs observed egocentric bias, {nm}: r={r:+.3f} p={p:.4f}')
            res[f'{rat}_pred_r_{nm}'] = float(r); res[f'{rat}_pred_p_{nm}'] = float(p)
    pd.concat(allrows).to_csv(f'{OUT}/bias_world_vs_ego.csv', index=False)


# ===================================================================== B
def part_B():
    print('\n' + '=' * 76)
    print('B. TRIANGULATION CONDITION NUMBER + principal error axis vs theta')
    print('=' * 76)
    cal = q.calib_dir('R5', '2026_07_08_1')
    P, K, D, R, T = q.load_calib(cal)
    rows = []
    for k in range(NSEC):
        th = np.radians(k * 30 - 165)
        for z in [-60.0, 60.0]:                 # floor-ish and rear-apex height
            X = np.array([ARENA.cx + 300 * np.cos(th), ARENA.cy + 300 * np.sin(th), z])
            M = np.zeros((3, 3))
            for c in range(3):
                J = np.zeros((2, 3))
                for a in range(3):
                    e = np.zeros(3); e[a] = 1.0
                    p1 = project_3d_to_2d((X + e)[None], K[c], R[c], T[c], D[c])[0]
                    p0 = project_3d_to_2d((X - e)[None], K[c], R[c], T[c], D[c])[0]
                    J[:, a] = (p1 - p0) / 2.0
                M += J.T @ J
            C = np.linalg.inv(M)                # covariance up to pixel-noise scale
            w, V = np.linalg.eigh(C)
            axis = V[:, -1]
            rows.append(dict(theta=k * 30 - 165, z=z,
                             cond=float(np.sqrt(w[-1] / w[0])),
                             sigma_major_mm=float(np.sqrt(w[-1])),
                             axis_azimuth=float(np.degrees(np.arctan2(axis[1], axis[0]))),
                             axis_elev=float(np.degrees(np.arcsin(np.clip(axis[2], -1, 1))))))
    d = pd.DataFrame(rows)
    for z, g in d.groupby('z'):
        print(f'\n  z = {z:+.0f} mm')
        print(g[['theta', 'cond', 'sigma_major_mm', 'axis_azimuth', 'axis_elev']].to_string(
            index=False, float_format=lambda v: f'{v:.2f}'))
    d.to_csv(f'{OUT}/triangulation_geometry.csv', index=False)
    hi = d[d.z > 0]
    print(f'\n  condition number at rear height spans {hi.cond.min():.1f}-{hi.cond.max():.1f}')
    print(f'  principal-error azimuth rotates {hi.axis_azimuth.min():.0f} to '
          f'{hi.axis_azimuth.max():.0f} deg across sectors')
    res['cond_min'] = float(hi.cond.min()); res['cond_max'] = float(hi.cond.max())


# ===================================================================== C
def part_C():
    print('\n' + '=' * 76)
    print('C. ROTATION-INVARIANT features — does the sector dependence survive?')
    print('=' * 76)
    ii, jj = np.triu_indices(23, k=1)
    for rat in ['R4', 'R5', 'R6']:
        ego_by_sec = {k: [] for k in range(NSEC)}
        pd_by_sec = {k: [] for k in range(NSEC)}
        for f in sorted(glob.glob(f'data/cache/{rat}/*.npz'))[::2]:
            session = os.path.basename(f)[:-4]
            raw = sc.get_raw_session(rat, session, want_dual=True)
            if raw is None or not raw.get('has_dual', False):
                continue
            t = tp.load_template(rat, session)
            if t is None:
                continue
            d3 = du.dannce_aligned(raw)
            if d3 is None:
                continue
            s3 = raw['triang_3d'].astype(np.float32).copy()
            s3[:, :, 2] = -s3[:, :, 2]
            n = min(len(s3), len(d3)); s3, d3 = s3[:n], d3[:n].astype(np.float32)
            pcu = np.asarray(t['pcs_to_use']).ravel().astype(int)
            s_pcs, _ = ev.project_session_to_pcs(raw['triang_3d'][:n], t, flip_z=True)
            rears = [r for r in ev.detect_rears(s_pcs[:, pcu[0]]) if r[1] < n]
            if not rears:
                continue
            fr = np.concatenate([np.arange(a, b + 1) for a, b, _na in rears])
            fr = fr[fr < n]
            if len(fr) < 50:
                continue
            fr = fr[::3]
            com = s3.mean(axis=1)
            sec = sector_index(ARENA.theta(com[:, :2]), SECTOR_W)
            rad = ARENA.radius_of(com[:, :2])
            fr = fr[rad[fr] > 250]
            if len(fr) < 20:
                continue
            se, _, _ = normalize_skeleton_batch(s3[fr])
            de, _, _ = normalize_skeleton_batch(d3[fr])
            ego = np.linalg.norm(se - de, axis=-1).mean(axis=1)      # mm
            sD = np.linalg.norm(s3[fr][:, ii] - s3[fr][:, jj], axis=-1)
            dD = np.linalg.norm(d3[fr][:, ii] - d3[fr][:, jj], axis=-1)
            pdd = np.abs(sD - dD).mean(axis=1)                        # mm
            for k in range(NSEC):
                m = sec[fr] == k
                if m.sum():
                    ego_by_sec[k].append(ego[m]); pd_by_sec[k].append(pdd[m])
        rows = []
        for k in range(NSEC):
            if not ego_by_sec[k]:
                continue
            e = np.concatenate(ego_by_sec[k]); p = np.concatenate(pd_by_sec[k])
            if len(e) < 200:
                continue
            rows.append(dict(theta=k * 30 - 165, n=len(e),
                             ego_mm=float(np.median(e)), pdist_mm=float(np.median(p))))
        d = pd.DataFrame(rows)
        # coefficient of variation across sectors: the scale-free comparison
        cv_e = d.ego_mm.std() / d.ego_mm.mean()
        cv_p = d.pdist_mm.std() / d.pdist_mm.mean()
        print(f'\n{rat}')
        print(d.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
        print(f'   across-sector CV, egocentric coords   = {cv_e:.4f}')
        print(f'   across-sector CV, pairwise distances  = {cv_p:.4f}')
        print(f'   reduction = {(1-cv_p/cv_e)*100:.0f}%')
        res[f'{rat}_cv_ego'] = float(cv_e); res[f'{rat}_cv_pdist'] = float(cv_p)


if __name__ == '__main__':
    part_A(); part_B(); part_C()
    with open(f'{OUT}/stage8_results.json', 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f'\nwrote {OUT}/stage8_results.json')
