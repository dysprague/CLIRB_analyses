"""Stage 12b — propagate the 2D error field through triangulation -> predicted mu(theta).

THE TEST. If aspect-dependent 2D error is what makes reward position-dependent,
then pushing the MEASURED 2D field (stage12a) through the ACTUAL camera geometry
at each arena position must reproduce the MEASURED mu(theta) (stage6).

Chain, per arena sector:
  1. place a canonical DANNCE rear pose at the wall with that sector's measured
     mean heading                                     -> world (calibration) coords
  2. per camera: viewing aspect, projected body-axis angle phi, and the
     projection Jacobian J_c = d(proj)/dX at every keypoint
  3. look up the body-relative 2D bias/cov for (keypoint, camera, aspect) and
     rotate it into that camera's image frame by +phi
  4. linearised triangulation:
         M        = sum_c J_c^T J_c
         bias_3D  = M^-1 sum_c J_c^T bias_2D_c
         cov_3D   = M^-1 (sum_c J_c^T cov_c J_c) M^-1
  5. egocentric transform, INCLUDING the yaw-estimate coupling (a bias on
     SpineF/SpineM rotates every other keypoint), then project through the rat's
     own pc_weights                                   -> predicted PC0 bias
  6. correlate predicted vs measured mu(theta)

Everything except step 6 is pure geometry plus the measured field: no fitting.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from scipy import stats

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb.spatial import quality as q
from clirb.spatial.geometry import ARENA, sector_index, sector_center
from clirb import templates as tp, events as ev
from config import NODE_IDX
from projection import project_3d_to_2d
from skeleton import normalize_skeleton_batch

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
CAL = f'{DR}/calibrations/2026_05_04'
OUT = 'results/spatial_confound'
NASP, NSEC, NKP = 12, 12, 23
EPS = 1.0                       # mm, for the numerical Jacobian


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def proj_jacobian(X, K, R, T, D):
    """(n,2,3) d(pixel)/d(mm) by central differences."""
    n = len(X)
    J = np.zeros((n, 2, 3))
    for a in range(3):
        e = np.zeros(3); e[a] = EPS
        p1 = project_3d_to_2d(X + e, K, R, T, D)
        p0 = project_3d_to_2d(X - e, K, R, T, D)
        J[:, :, a] = (p1 - p0) / (2 * EPS)
    return J


def canonical_pose_and_headings():
    """Median egocentric rear pose (DANNCE) + measured mean heading per sector."""
    egos, head_by_sec = [], [[] for _ in range(NSEC)]
    for f in sorted(glob.glob('results/corrector_v2/keys2d/*.npz'))[:8]:
        rat, session = os.path.basename(f)[:-4].split('__')
        c = sc.get_raw_session(rat, session, want_dual=False)
        if c is None:
            continue
        t3 = c['triang_3d'].astype(np.float64)
        tm = tp.load_template(rat, session)
        if tm is None:
            continue
        pcs, _ = ev.project_session_to_pcs(t3, tm, flip_z=True)
        pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)
        rears = ev.detect_rears(pcs[:, pcu[0]])
        if not rears:
            continue
        fr = np.concatenate([np.arange(a, b + 1) for a, b, _ in rears])
        fr = fr[fr < len(t3)]
        com = t3[:, :, :2].mean(axis=1)
        rad = ARENA.radius_of(com); fr = fr[rad[fr] > 250]
        if len(fr) < 50:
            continue
        rot, _, _ = normalize_skeleton_batch(t3[fr])
        egos.append(rot)
        SF = t3[fr, NODE_IDX['SpineF'], :2]; SM = t3[fr, NODE_IDX['SpineM'], :2]
        h = np.degrees(np.arctan2(SF[:, 1] - SM[:, 1], SF[:, 0] - SM[:, 0]))
        s = sector_index(ARENA.theta(com[fr]), 30)
        for si, hh in zip(s, h):
            head_by_sec[si].append(hh)
    ego = np.median(np.concatenate(egos), axis=0)          # (23,3) egocentric
    heads = np.array([np.degrees(np.angle(np.mean(np.exp(1j * np.radians(v)))))
                      if len(v) > 20 else np.nan for v in head_by_sec])
    return ego, heads


def place(ego, pos_xy, heading_deg, z0=-40.0):
    """Egocentric pose -> world (calibration) coords at pos with heading."""
    h = np.radians(heading_deg)
    Rm = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    w = np.empty_like(ego)
    w[:, :2] = ego[:, :2] @ Rm.T + np.asarray(pos_xy)[None, :]
    w[:, 2] = ego[:, 2] + z0
    return w


def egocentric_delta(X, dX):
    """First-order change in the EGOCENTRIC pose given a world-space bias dX,
    including the yaw-estimate coupling."""
    iF, iM = NODE_IDX['SpineF'], NODE_IDX['SpineM']
    dxy = X[iF, :2] - X[iM, :2]
    L2 = float(dxy @ dxy)
    ddx = dX[iF, 0] - dX[iM, 0]; ddy = dX[iF, 1] - dX[iM, 1]
    drot = (dxy[1] * ddx - dxy[0] * ddy) / max(L2, 1e-9)
    rot = np.arctan2(-dxy[1], dxy[0])
    c, s = np.cos(rot), np.sin(rot)
    Rm = np.array([[c, -s], [s, c]])
    dRm = np.array([[-s, -c], [c, -s]]) * drot
    rel = (X - X[iM])[:, :2]
    d = np.empty_like(X)
    d[:, :2] = (dX - dX[iM])[:, :2] @ Rm.T + rel @ dRm.T
    d[:, 2] = dX[:, 2] - dX[iM, 2]
    return d


def main():
    P, K, D, R, T = q.load_calib(CAL)
    cams = np.stack([-R[c].T @ T[c] for c in range(3)])
    fz = np.load(f'{OUT}/field2d.npz')
    F_mean, F_cov, F_n = fz['mean'], fz['cov'], fz['count']

    ego, heads = canonical_pose_and_headings()
    print('canonical egocentric rear pose built; per-sector mean heading:')
    print('  ' + ' '.join(f'{h:+6.0f}' if np.isfinite(h) else '   nan' for h in heads))

    rows = []
    for s in range(NSEC):
        if not np.isfinite(heads[s]):
            continue
        th = np.radians(sector_center(s, 30))
        pos = np.array([ARENA.cx + (ARENA.radius - 40) * np.cos(th),
                        ARENA.cy + (ARENA.radius - 40) * np.sin(th)])
        Xw = place(ego, pos, heads[s])
        com = Xw[:, :2].mean(axis=0)
        SM3, SF3 = Xw[NODE_IDX['SpineM']], Xw[NODE_IDX['SpineF']]
        headdeg = np.degrees(np.arctan2(SF3[1] - SM3[1], SF3[0] - SM3[0]))

        M = np.zeros((NKP, 3, 3)); JtB = np.zeros((NKP, 3))
        JtCJ = np.zeros((NKP, 3, 3))
        for c in range(3):
            J = proj_jacobian(Xw, K[c], R[c], T[c], D[c])          # (23,2,3)
            pM = project_3d_to_2d(SM3[None], K[c], R[c], T[c], D[c])[0]
            pF = project_3d_to_2d(SF3[None], K[c], R[c], T[c], D[c])[0]
            phi = np.arctan2(pF[1] - pM[1], pF[0] - pM[0])
            cp, sp = np.cos(phi), np.sin(phi)
            Rot = np.array([[cp, -sp], [sp, cp]])                  # body -> image
            bear = np.degrees(np.arctan2(com[1] - cams[c][1], com[0] - cams[c][0]))
            ab = int(np.clip((wrap(headdeg - bear) + 180) / (360 / NASP), 0, NASP - 1))
            b_img = F_mean[:, c, ab] @ Rot.T                        # (23,2)
            C_img = np.einsum('ab,kbc,dc->kad', Rot, F_cov[:, c, ab], Rot)
            M += np.einsum('kab,kac->kbc', J, J)
            JtB += np.einsum('kab,ka->kb', J, b_img)
            JtCJ += np.einsum('kab,kbc,kdc->kad', J.transpose(0, 2, 1)[:, :, :, ],
                              C_img, J.transpose(0, 2, 1)) if False else \
                    np.einsum('kba,kbc,kcd->kad', J, C_img, J)
        Minv = np.linalg.inv(M + 1e-9 * np.eye(3)[None])
        dX = np.einsum('kab,kb->ka', Minv, JtB)                    # (23,3) mm bias
        C3 = np.einsum('kab,kbc,kcd->kad', Minv, JtCJ, Minv)

        dEgo = egocentric_delta(Xw, dX)
        rows.append(dict(sector=s, theta=sector_center(s, 30),
                         heading=heads[s], aspect_used=ab,
                         bias_mm=float(np.linalg.norm(dX, axis=-1).mean()),
                         noise_mm=float(np.sqrt(np.trace(C3, axis1=-2, axis2=-1)).mean()),
                         dEgo=dEgo, C3=C3))
    print(f'\n{len(rows)} sectors predicted')

    # --- project into each rat's own PC basis and compare with measured mu(theta)
    res = {}
    print('\n' + '=' * 74)
    print('PREDICTED vs MEASURED  mu(theta)   [PC0 bias]')
    print('=' * 74)
    for rat in ['R4', 'R5', 'R6']:
        mf = f'{OUT}/residual_field_{rat}.csv'
        if not os.path.isfile(mf):
            continue
        meas = pd.read_csv(mf)
        meas['sector'] = ((meas.theta_c + 180) // 30).astype(int)
        sess = sorted(os.path.basename(f)[:-4]
                      for f in glob.glob(f'data/cache/{rat}/*.npz'))
        tm = None
        for s_ in sess:
            tm = tp.load_template(rat, s_)
            if tm is not None:
                break
        if tm is None:
            continue
        pcw = tm['pc_weights']; pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)
        pred = []
        for r in rows:
            dpc = r['dEgo'].reshape(-1) @ pcw.T
            sd = np.sqrt(np.einsum('j,kjl,l->', pcw[pcu[0]].reshape(-1),
                                   np.zeros((1, 69, 69)), pcw[pcu[0]].reshape(-1))) \
                if False else np.nan
            pred.append(dict(sector=r['sector'], theta=r['theta'],
                             pred_pc0=float(dpc[pcu[0]]),
                             bias_mm=r['bias_mm'], noise_mm=r['noise_mm']))
        pdf = pd.DataFrame(pred).merge(meas[['sector', 'mu_pc0']], on='sector', how='inner')
        if len(pdf) < 6:
            continue
        rho, p = stats.spearmanr(pdf.pred_pc0, pdf.mu_pc0)
        rp, pp = stats.pearsonr(pdf.pred_pc0, pdf.mu_pc0)
        print(f'\n{rat}  (n={len(pdf)} sectors)')
        print(pdf[['theta', 'pred_pc0', 'mu_pc0', 'bias_mm', 'noise_mm']].to_string(
            index=False, float_format=lambda v: f'{v:.2f}'))
        print(f'   Spearman rho = {rho:+.3f} (p={p:.3f})   Pearson r = {rp:+.3f} (p={pp:.3f})')
        res[rat] = dict(rho=float(rho), p=float(p), r=float(rp),
                        pred_range=float(pdf.pred_pc0.max() - pdf.pred_pc0.min()),
                        meas_range=float(pdf.mu_pc0.max() - pdf.mu_pc0.min()))
        pdf.to_csv(f'{OUT}/propagation_{rat}.csv', index=False)
    json.dump(res, open(f'{OUT}/stage12_results.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/stage12_results.json')


if __name__ == '__main__':
    main()
