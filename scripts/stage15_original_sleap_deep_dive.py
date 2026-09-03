"""Stage 15 — two retests on the REAL production SLEAP 2D (`sleap_keys_2D.npy`),
combined into one SMB pass since both need the same expensive per-session read
+ DANNCE alignment that stage14a already validated (correct Procrustes
direction, correct processed-frame-counter bridge via
`dannce_idx_for_sleap_cams_keys`).

Recommendation (1): retest §8g/§8h's posture and limb-configuration
conditioning at the §8j scale (540k frames, real model) instead of the small
fine-tuned-reinference dataset that understated the aspect mechanism 2-3x.
Posture bin edges and limb-config KMeans centroids are fit from the ALREADY
LOCAL `dataset_original.npz` (no SMB needed for that part) so only one new
pass over `sleap_keys_2D.npy` is required.

Recommendation (3): redo stage11's aspect-mechanism characterization
(H-OCC self-occlusion, H-SWAP left/right confusion, aspect-profile shape,
H-LAB confidence-honesty) on the same real, ~540k-frame dataset instead of the
fine-tuned reinference's ~30k frames.

Writes:
  results/spatial_confound/field2d_original_posture.npz
  results/spatial_confound/field2d_original_limbconfig.npz
  results/spatial_confound/aspect_profile_by_keypoint_original.csv
  results/spatial_confound/aspect_mechanism_by_keypoint_original.csv
  results/spatial_confound/stage16_results.json
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial import quality as q
from clirb.spatial.geometry import sector_index
from config import NODES, NODE_IDX
from projection import project_3d_to_2d
from corrector.world_alignment import fit_procrustes
from skeleton import normalize_skeleton_batch

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
CAL = f'{DR}/calibrations/2026_05_04'
SESS_SRC = 'results/corrector_v2/keys2d'
OUT = 'results/spatial_confound'
CONF = 0.5
NASP = 12
NPOST = 3
NCLUST = 4
MAXRES = 120.0
NKP = 23
SECTOR_W = 30
RNG = np.random.default_rng(0)

PAIRS = []
for i, n in enumerate(NODES):
    if n.endswith('L') and (n[:-1] + 'R') in NODES:
        PAIRS.append((i, NODE_IDX[n[:-1] + 'R'], n[:-1]))
VIRTUAL = {'SpineF', 'SpineM', 'SpineL', 'TailBase', 'HipL', 'HipR',
           'ShoulderL', 'ShoulderR'}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def robust_proc(A, B, iters=5, keep=0.7):
    idx = np.arange(len(A)); f = None
    for _ in range(iters):
        f = fit_procrustes(A[idx], B[idx], try_z_flip=True)
        r = np.linalg.norm(f['apply'](A) - B, axis=-1).mean(axis=1)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 200:
            break
    return f


def fit_posture_and_cluster_bins():
    z = np.load(f'{OUT}/dataset_original.npz')
    dncal = z['DNCAL'].astype(np.float64)
    posture = dncal[:, :, 2].max(axis=1) - dncal[:, NODE_IDX['SpineM'], 2]
    edges = np.quantile(posture, np.linspace(0, 1, NPOST + 1))
    edges[0] -= 1.0; edges[-1] += 1.0
    print(f'posture edges (from {len(posture)} local frames): {np.round(edges, 1)}')

    n_fit = min(60000, len(dncal))
    fit_idx = RNG.choice(len(dncal), n_fit, replace=False)
    ego, _, _ = normalize_skeleton_batch(dncal[fit_idx])
    flat = ego.reshape(len(ego), -1)
    mu, sd = flat.mean(axis=0), flat.std(axis=0) + 1e-9
    zf = (flat - mu) / sd
    km = KMeans(n_clusters=NCLUST, n_init=10, random_state=0).fit(zf)
    print(f'KMeans fit on {n_fit} frames, cluster sizes: {np.bincount(km.labels_)}')
    return edges, mu, sd, km.cluster_centers_


def main():
    post_edges, lc_mu, lc_sd, lc_centroids = fit_posture_and_cluster_bins()

    P, K, D, R, T = q.load_calib(CAL)
    cams = np.stack([-R[c].T @ T[c] for c in range(3)])

    # posture field
    S0p = np.zeros((NKP, 3, NASP, NPOST))
    S1p = np.zeros((NKP, 3, NASP, NPOST, 2))
    S2p = np.zeros((NKP, 3, NASP, NPOST, 2, 2))
    # limb-config field
    S0c = np.zeros((NKP, 3, NASP, NCLUST))
    S1c = np.zeros((NKP, 3, NASP, NCLUST, 2))
    S2c = np.zeros((NKP, 3, NASP, NCLUST, 2, 2))

    # stage11-style mechanism accumulators
    pair_near_e = {nm: [] for _, _, nm in PAIRS}; pair_far_e = {nm: [] for _, _, nm in PAIRS}
    pair_near_c = {nm: [] for _, _, nm in PAIRS}; pair_far_c = {nm: [] for _, _, nm in PAIRS}
    pair_asis = {nm: [] for _, _, nm in PAIRS}; pair_swap = {nm: [] for _, _, nm in PAIRS}
    asp_edges12 = np.linspace(-180, 180, NASP + 1)
    prof_e = [[[] for _ in range(NASP)] for _ in range(NKP)]
    prof_c = [[[] for _ in range(NASP)] for _ in range(NKP)]
    kp_err = [[] for _ in range(NKP)]; kp_conf = [[] for _ in range(NKP)]

    sessions = sorted({os.path.basename(f)[:-4] for f in glob.glob(f'{SESS_SRC}/*.npz')})
    for base in sessions:
        rat, session = base.split('__')
        ap = f'{DR}/{rat}/{session}/processed/aligned_data.npz'
        dnp = f'{DR}/{rat}/{session}/processed/sleap_dannce_keys.npz'
        orig2d_path = f'{DR}/{rat}/{session}/sleap/sleap_keys_2D.npy'
        if not (os.path.isfile(ap) and os.path.isfile(dnp) and os.path.isfile(orig2d_path)):
            print(f'  {rat}/{session}: missing a required file, skip'); continue
        with np.load(ap, allow_pickle=True) as z:
            dk = z['dannce_idx_for_sleap_cams_keys']
        with np.load(dnp, allow_pickle=True) as z:
            dn_raw = z['dannce_keys_3D']
        dn = (dn_raw.squeeze(axis=1).transpose(0, 2, 1) if dn_raw.ndim == 4
              else np.transpose(dn_raw, [0, 2, 1]))

        n_proc = len(dk)
        k2_full = np.load(orig2d_path, mmap_mode='r')
        n_avail = min(n_proc, k2_full.shape[0])
        k = np.arange(n_avail)
        k2 = np.asarray(k2_full[:n_avail])

        tgt = dn[np.clip(dk[k], 0, len(dn) - 1)].astype(np.float64)
        und = q.undistort_all(np.nan_to_num(k2[..., :2]).astype(np.float64), K, D)
        X = q.triangulate_batch(und, P)
        good = (np.isfinite(X).all(axis=(1, 2)) & np.isfinite(tgt).all(axis=(1, 2)))
        rad = np.linalg.norm(X - X.mean(axis=1, keepdims=True), axis=-1).max(axis=1)
        good &= rad < 250
        if good.sum() < 400:
            print(f'  {rat}/{session}: only {good.sum()} usable frames, skip'); continue
        X, tgt, k2g = X[good], tgt[good], k2[good]

        com = X[:, :, :2].mean(axis=1)
        th = np.degrees(np.arctan2(com[:, 1], com[:, 0]))
        sec = sector_index(th, SECTOR_W)
        samp = []
        for sname in range(int(round(360 / SECTOR_W))):
            ii = np.where(sec == sname)[0]
            if len(ii) > 20:
                samp.extend(RNG.choice(ii, min(len(ii), 400), replace=False))
        samp = np.array(samp) if len(samp) > 300 else np.arange(len(X))
        fit = robust_proc(tgt[samp], X[samp])                 # CORRECT direction (tgt -> X)
        dn_cal = fit['apply'](tgt)
        resid = float(np.median(np.linalg.norm(dn_cal - X, axis=-1)))
        if resid > 60:
            print(f'  {rat}/{session}: Procrustes residual {resid:.1f} mm too high, skip')
            continue

        SF3 = dn_cal[:, NODE_IDX['SpineF']]; SM3 = dn_cal[:, NODE_IDX['SpineM']]
        head = np.degrees(np.arctan2(SF3[:, 1] - SM3[:, 1], SF3[:, 0] - SM3[:, 0]))
        posture = dn_cal[:, :, 2].max(axis=1) - SM3[:, 2]
        pbin = np.clip(np.digitize(posture, post_edges) - 1, 0, NPOST - 1)
        ego, _, _ = normalize_skeleton_batch(dn_cal)
        ez = (ego.reshape(len(ego), -1) - lc_mu) / lc_sd
        cbin = np.argmin(np.linalg.norm(lc_centroids[None, :, :] - ez[:, None, :], axis=-1), axis=1)

        n_frames = len(dn_cal)
        for c in range(3):
            proj = project_3d_to_2d(dn_cal.reshape(-1, 3), K[c], R[c], T[c],
                                    D[c]).reshape(n_frames, NKP, 2)
            pM = project_3d_to_2d(SM3, K[c], R[c], T[c], D[c])
            pF = project_3d_to_2d(SF3, K[c], R[c], T[c], D[c])
            phi = np.arctan2(pF[:, 1] - pM[:, 1], pF[:, 0] - pM[:, 0])
            cp, sp = np.cos(-phi), np.sin(-phi)
            bear = np.degrees(np.arctan2(com[:, 1] - cams[c][1], com[:, 0] - cams[c][0]))
            aspect = wrap(head - bear)
            abin = np.clip(((aspect + 180) / (360 / NASP)).astype(int), 0, NASP - 1)

            det = k2g[:, c, :, :2].astype(np.float64)
            cf = k2g[:, c, :, 2]
            res = det - proj                                    # raw pixel residual (stage11-style)
            rb = np.stack([cp[:, None] * res[..., 0] - sp[:, None] * res[..., 1],
                           sp[:, None] * res[..., 0] + cp[:, None] * res[..., 1]],
                          axis=-1)                                # body-relative (field-style)
            dist_cam = np.linalg.norm(dn_cal - cams[c][None, None, :], axis=-1)

            keep_rb = (cf >= CONF) & np.isfinite(rb).all(axis=-1) & \
                     (np.linalg.norm(rb, axis=-1) < MAXRES)
            lin_p = abin * NPOST + pbin
            lin_c = abin * NCLUST + cbin
            for j in range(NKP):
                m = keep_rb[:, j]
                if not m.any():
                    continue
                v = rb[m, j]
                ip = lin_p[m]; ic = lin_c[m]
                np.add.at(S0p[j, c].reshape(NASP * NPOST), ip, 1.0)
                np.add.at(S1p[j, c].reshape(NASP * NPOST, 2), ip, v)
                np.add.at(S2p[j, c].reshape(NASP * NPOST, 2, 2), ip, v[:, :, None] * v[:, None, :])
                np.add.at(S0c[j, c].reshape(NASP * NCLUST), ic, 1.0)
                np.add.at(S1c[j, c].reshape(NASP * NCLUST, 2), ic, v)
                np.add.at(S2c[j, c].reshape(NASP * NCLUST, 2, 2), ic, v[:, :, None] * v[:, None, :])

            keep_raw = (cf >= CONF) & np.isfinite(res).all(axis=-1) & \
                      (np.linalg.norm(res, axis=-1) < MAXRES)
            for j in range(NKP):
                m = keep_raw[:, j]
                if not m.any():
                    continue
                e = np.linalg.norm(res[m, j], axis=-1).astype(np.float32)
                kp_err[j].append(e); kp_conf[j].append(cf[m, j].astype(np.float32))
                ab = abin[m]
                for b in range(NASP):
                    mb = ab == b
                    if mb.any():
                        prof_e[j][b].append(e[mb]); prof_c[j][b].append(cf[m, j][mb].astype(np.float32))

            for pl, pr, nm in PAIRS:
                mm = keep_raw[:, pl] & keep_raw[:, pr]
                if not mm.any():
                    continue
                dl, dr = dist_cam[mm, pl], dist_cam[mm, pr]
                el = np.linalg.norm(res[mm, pl], axis=-1); er = np.linalg.norm(res[mm, pr], axis=-1)
                cl, cr = cf[mm, pl], cf[mm, pr]
                l_is_far = dl > dr
                pair_far_e[nm].append(np.where(l_is_far, el, er).astype(np.float32))
                pair_near_e[nm].append(np.where(l_is_far, er, el).astype(np.float32))
                pair_far_c[nm].append(np.where(l_is_far, cl, cr).astype(np.float32))
                pair_near_c[nm].append(np.where(l_is_far, cr, cl).astype(np.float32))
                a_ = (np.linalg.norm(det[mm, pl] - proj[mm, pl], axis=-1) +
                      np.linalg.norm(det[mm, pr] - proj[mm, pr], axis=-1))
                s_ = (np.linalg.norm(det[mm, pl] - proj[mm, pr], axis=-1) +
                      np.linalg.norm(det[mm, pr] - proj[mm, pl], axis=-1))
                pair_asis[nm].append(a_.astype(np.float32)); pair_swap[nm].append(s_.astype(np.float32))

        print(f'  {rat}/{session}: {n_frames} frames, Procrustes resid {resid:.1f} mm', flush=True)

    # ---------------------------------------------------------- save fields
    def finalize(S0, S1, S2):
        n = np.maximum(S0, 1)[..., None]
        mean = S1 / n
        cov = S2 / np.maximum(S0, 1)[..., None, None] - mean[..., :, None] * mean[..., None, :]
        return mean, cov

    mean_p, cov_p = finalize(S0p, S1p, S2p)
    np.savez_compressed(f'{OUT}/field2d_original_posture.npz', count=S0p, mean=mean_p, cov=cov_p,
                        nasp=NASP, npost=NPOST, post_edges=post_edges)
    print(f'\nwrote field2d_original_posture.npz: '
          f'{(S0p >= 50).sum()}/{S0p.size} cells >=50 obs')

    mean_c, cov_c = finalize(S0c, S1c, S2c)
    np.savez_compressed(f'{OUT}/field2d_original_limbconfig.npz', count=S0c, mean=mean_c, cov=cov_c,
                        nasp=NASP, nclust=NCLUST, centroids=lc_centroids, norm_mu=lc_mu, norm_sd=lc_sd)
    print(f'wrote field2d_original_limbconfig.npz: '
          f'{(S0c >= 50).sum()}/{S0c.size} cells >=50 obs')

    # ---------------------------------------------------- stage11-style report
    res_json = {}
    print('\n' + '=' * 76); print('H-OCC  far/near bilateral error ratio (original SLEAP)'); print('=' * 76)
    rows = []
    for _, _, nm in PAIRS:
        if not pair_far_e[nm]:
            continue
        fe = np.concatenate(pair_far_e[nm]); ne = np.concatenate(pair_near_e[nm])
        fc = np.concatenate(pair_far_c[nm]); nc = np.concatenate(pair_near_c[nm])
        rows.append(dict(pair=nm, n=len(fe), near_px=np.median(ne), far_px=np.median(fe),
                         ratio=np.median(fe) / max(np.median(ne), 1e-9),
                         near_conf=np.median(nc), far_conf=np.median(fc)))
    t = pd.DataFrame(rows).sort_values('ratio', ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    res_json['occ_median_ratio'] = float(t.ratio.median())
    res_json['occ_conf_drop'] = float((t.near_conf - t.far_conf).median())

    print('\n' + '=' * 76); print('H-SWAP  would swapping L/R reduce the residual?'); print('=' * 76)
    rows = []
    for _, _, nm in PAIRS:
        if not pair_asis[nm]:
            continue
        a = np.concatenate(pair_asis[nm]); s = np.concatenate(pair_swap[nm])
        rows.append(dict(pair=nm, n=len(a), asis_px=np.median(a), swap_px=np.median(s),
                         frac_swap_better=float((s < a).mean())))
    t = pd.DataFrame(rows).sort_values('frac_swap_better', ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    res_json['swap_frac'] = float(t.frac_swap_better.median())

    print('\n' + '=' * 76); print('SHAPE  aspect-profile'); print('=' * 76)
    ctr = (asp_edges12[:-1] + asp_edges12[1:]) / 2
    prof = np.full((NKP, NASP), np.nan); cprof = np.full((NKP, NASP), np.nan)
    for j in range(NKP):
        for b in range(NASP):
            if prof_e[j][b]:
                e = np.concatenate(prof_e[j][b]); c_ = np.concatenate(prof_c[j][b])
                if len(e) > 100:
                    prof[j, b] = np.median(e); cprof[j, b] = np.median(c_)
    pooled = np.nanmedian(prof, axis=0)
    print('  aspect bin centre :', ' '.join(f'{c:+6.0f}' for c in ctr))
    print('  pooled median err :', ' '.join(f'{v:6.2f}' for v in pooled))
    side = np.nanmean(pooled[(np.abs(ctr) > 60) & (np.abs(ctr) < 120)])
    endon = np.nanmean(pooled[(np.abs(ctr) < 30) | (np.abs(ctr) > 150)])
    print(f'  side-on mean={side:.2f}px  end-on mean={endon:.2f}px  ratio={side/max(endon,1e-9):.3f}')
    res_json['side_px'] = float(side); res_json['end_px'] = float(endon)
    pd.DataFrame(prof, index=NODES, columns=np.round(ctr).astype(int)).to_csv(
        f'{OUT}/aspect_profile_by_keypoint_original.csv')

    print('\n' + '=' * 76); print('H-LAB  does confidence track the error?'); print('=' * 76)
    rows = []
    for j in range(NKP):
        if not kp_err[j]:
            continue
        e = np.concatenate(kp_err[j]); c_ = np.concatenate(kp_conf[j])
        rho, _ = stats.spearmanr(c_, e)
        amp = np.nanmax(prof[j]) - np.nanmin(prof[j]) if np.isfinite(prof[j]).sum() > 6 else np.nan
        rows.append(dict(kp=NODES[j], n=len(e), med_px=np.median(e), conf=np.median(c_),
                         rho_conf_err=rho, aspect_amp=amp,
                         kind='virtual' if NODES[j] in VIRTUAL else 'visible'))
    t = pd.DataFrame(rows).sort_values('aspect_amp', ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    t.to_csv(f'{OUT}/aspect_mechanism_by_keypoint_original.csv', index=False)
    res_json['rho_conf_err'] = float(t.rho_conf_err.median())
    res_json['amp_virtual'] = float(t[t.kind == 'virtual'].aspect_amp.median())
    res_json['amp_visible'] = float(t[t.kind == 'visible'].aspect_amp.median())

    json.dump(res_json, open(f'{OUT}/stage16_results.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/stage16_results.json')


if __name__ == '__main__':
    main()
