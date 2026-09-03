"""Stage 15c — apples-to-apples propagation test, ORIGINAL production SLEAP 2D,
conditioned on (aspect, limb-configuration cluster). Retest of §8h at §8j's scale.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy import stats

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial import quality as q
from clirb.spatial.geometry import sector_center
from clirb import templates as tp, events as ev
from config import NODE_IDX
from projection import project_3d_to_2d
from skeleton import normalize_skeleton_batch

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
CAL = f'{DR}/calibrations/2026_05_04'
OUT = 'results/spatial_confound'
NASP, NSEC, NKP = 12, 12, 23
EPS = 1.0
PER_SEC = 200
RNG = np.random.default_rng(0)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def proj_jacobian(X, K, R, T, D):
    n = len(X); J = np.zeros((n, 2, 3))
    for a in range(3):
        e = np.zeros(3); e[a] = EPS
        J[:, :, a] = (project_3d_to_2d(X + e, K, R, T, D)
                      - project_3d_to_2d(X - e, K, R, T, D)) / (2 * EPS)
    return J


def egocentric_delta(X, dX):
    iF, iM = NODE_IDX['SpineF'], NODE_IDX['SpineM']
    dxy = X[iF, :2] - X[iM, :2]; L2 = float(dxy @ dxy)
    ddx = dX[iF, 0] - dX[iM, 0]; ddy = dX[iF, 1] - dX[iM, 1]
    drot = (dxy[1] * ddx - dxy[0] * ddy) / max(L2, 1e-9)
    rot = np.arctan2(-dxy[1], dxy[0]); c, s = np.cos(rot), np.sin(rot)
    Rm = np.array([[c, -s], [s, c]]); dRm = np.array([[-s, -c], [c, -s]]) * drot
    rel = (X - X[iM])[:, :2]
    d = np.empty_like(X)
    d[:, :2] = (dX - dX[iM])[:, :2] @ Rm.T + rel @ dRm.T
    d[:, 2] = dX[:, 2] - dX[iM, 2]
    return d


def assign_cluster(Xw, mu, sd, centroids):
    ego, _, _ = normalize_skeleton_batch(Xw[None])
    z = (ego.reshape(1, -1) - mu) / sd
    dd = np.linalg.norm(centroids[None, :, :] - z[:, None, :], axis=-1)
    return int(np.argmin(dd[0]))


def main():
    P, K, D, R, T = q.load_calib(CAL)
    cams = np.stack([-R[c].T @ T[c] for c in range(3)])
    fz = np.load(f'{OUT}/field2d_original_limbconfig.npz')
    F_mean = fz['mean']
    mu, sd, centroids = fz['norm_mu'], fz['norm_sd'], fz['centroids']
    NCLUST = int(fz['nclust'])
    print(f'limb-config clusters: {NCLUST}')

    z = np.load(f'{OUT}/dataset_original.npz')
    XC, dn_cal, sector, rat_arr = z['XCAL'], z['DNCAL'], z['SECTOR'], z['RAT']

    rows = []
    for rat in ['R4', 'R5', 'R6']:
        sel_rat = rat_arr == rat
        if sel_rat.sum() < 500:
            continue
        tm = None
        for s_ in sorted({os.path.basename(f)[:-4].split('__')[1]
                          for f in glob.glob(f'results/corrector_v2/keys2d/{rat}__*.npz')}):
            tm = tp.load_template(rat, s_)
            if tm is not None:
                break
        if tm is None:
            print(f'{rat}: no template, skip'); continue
        pcw = tm['pc_weights']; pcu = np.asarray(tm['pcs_to_use']).ravel().astype(int)

        pcs_raw, _ = ev.project_session_to_pcs(XC[sel_rat].astype(np.float64), tm, flip_z=True)
        pcs_dn, _ = ev.project_session_to_pcs(dn_cal[sel_rat].astype(np.float64), tm, flip_z=True)
        d_pc0 = pcs_raw[:, pcu[0]] - pcs_dn[:, pcu[0]]
        sec_r = sector[sel_rat]
        rad_ok = np.linalg.norm(XC[sel_rat] - XC[sel_rat].mean(axis=1, keepdims=True),
                                axis=-1).max(axis=1) < 250

        for s in range(NSEC):
            m = (sec_r == s) & rad_ok & np.isfinite(d_pc0)
            if m.sum() < 60:
                continue
            meas = float(np.median(d_pc0[m]))
            idx = np.where(m)[0]
            pick = RNG.choice(idx, min(PER_SEC, len(idx)), replace=False)
            Xall = dn_cal[sel_rat][pick].astype(np.float64)
            preds = []
            for Xw in Xall:
                com = Xw[:, :2].mean(axis=0)
                SM3, SF3 = Xw[NODE_IDX['SpineM']], Xw[NODE_IDX['SpineF']]
                hd = np.degrees(np.arctan2(SF3[1] - SM3[1], SF3[0] - SM3[0]))
                cl = assign_cluster(Xw, mu, sd, centroids)
                Mm = np.zeros((NKP, 3, 3)); JtB = np.zeros((NKP, 3))
                for c in range(3):
                    J = proj_jacobian(Xw, K[c], R[c], T[c], D[c])
                    pM = project_3d_to_2d(SM3[None], K[c], R[c], T[c], D[c])[0]
                    pF = project_3d_to_2d(SF3[None], K[c], R[c], T[c], D[c])[0]
                    phi = np.arctan2(pF[1] - pM[1], pF[0] - pM[0])
                    cp, sp = np.cos(phi), np.sin(phi)
                    Rot = np.array([[cp, -sp], [sp, cp]])
                    bear = np.degrees(np.arctan2(com[1] - cams[c][1], com[0] - cams[c][0]))
                    ab = int(np.clip((wrap(hd - bear) + 180) / (360 / NASP), 0, NASP - 1))
                    Mm += np.einsum('kab,kac->kbc', J, J)
                    JtB += np.einsum('kab,ka->kb', J, F_mean[:, c, ab, cl] @ Rot.T)
                dX = np.einsum('kab,kb->ka', np.linalg.inv(Mm + 1e-9 * np.eye(3)[None]), JtB)
                preds.append((egocentric_delta(Xw, dX).reshape(-1) @ pcw.T)[pcu[0]])
            rows.append(dict(rat=rat, sector=s, theta=sector_center(s, 30),
                             n=int(m.sum()), measured=meas, predicted=float(np.median(preds))))
            print(f'  {rat} sector {s:2d} ({sector_center(s,30):+4.0f}): '
                  f'n={m.sum():6d}  measured={meas:+7.2f}  predicted={np.median(preds):+7.2f}',
                  flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/propagation_matched_original_limbconfig.csv', index=False)
    print('\n' + '=' * 74); print('MATCHED TEST, ORIGINAL SLEAP + LIMB CONFIGURATION'); print('=' * 74)
    res = {}
    for rat, g in d.groupby('rat'):
        if len(g) < 6:
            continue
        rho, p = stats.spearmanr(g.predicted, g.measured)
        rp, pp = stats.pearsonr(g.predicted, g.measured)
        print(f'{rat} (n={len(g)}): Spearman {rho:+.3f} (p={p:.3f})  Pearson {rp:+.3f} (p={pp:.3f})  '
              f'range pred {g.predicted.max()-g.predicted.min():.1f} vs meas {g.measured.max()-g.measured.min():.1f}')
        res[rat] = dict(rho=float(rho), p=float(p), r=float(rp),
                        pred_range=float(g.predicted.max() - g.predicted.min()),
                        meas_range=float(g.measured.max() - g.measured.min()))
    if len(d) >= 8:
        rho, p = stats.spearmanr(d.predicted, d.measured)
        rp, pp = stats.pearsonr(d.predicted, d.measured)
        print(f'\nPOOLED (n={len(d)}): Spearman {rho:+.3f} (p={p:.3f})  Pearson {rp:+.3f} (p={pp:.3f})')
        res['pooled'] = dict(rho=float(rho), p=float(p), r=float(rp), n=int(len(d)))

    old = json.load(open(f'{OUT}/stage14b_results.json')) if os.path.isfile(f'{OUT}/stage14b_results.json') else {}
    print('\nBEFORE (aspect only, original SLEAP, stage14b) vs AFTER (+limb config, stage15c)')
    for rat in ['R4', 'R5', 'R6', 'pooled']:
        if rat in old and rat in res:
            print(f'  {rat:8s}  rho {old[rat]["rho"]:+.3f} -> {res[rat]["rho"]:+.3f}   '
                  f'p {old[rat]["p"]:.3f} -> {res[rat]["p"]:.3f}')
    json.dump(res, open(f'{OUT}/stage15c_results.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/propagation_matched_original_limbconfig.csv and stage15c_results.json')


if __name__ == '__main__':
    main()
