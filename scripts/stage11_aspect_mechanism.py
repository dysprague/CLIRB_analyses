"""Stage 11 — WHY is the SLEAP 2D error viewing-aspect dependent?

Uses the fine-tuned model's 2D (corrector_v2) against the independent 6-camera
DANNCE pose reprojected into each SLEAP camera, so the residual is measured
against something that does not share SLEAP's errors.

Four competing accounts, each with a distinct signature:

  H-OCC  self-occlusion. The single-instance net must emit all 23 parts even
         when the far-side ones are hidden, so it guesses.
         -> error is HIGHER for the bilateral partner FARTHER from the camera
         -> error peaks when the animal is SIDE-ON (|aspect| ~ 90 deg)

  H-SWAP left/right identity confusion.
         -> swapping the L/R assignment REDUCES the residual at some aspects
         -> error concentrated in bilateral pairs, magnitude ~ the L-R separation

  H-FORE foreshortening: a part pointing toward/away from the camera projects
         to a short, broad heatmap whose argmax is unstable.
         -> error peaks at aspect ~ 0 / 180 deg, and grows with how much of the
            body axis lies along the viewing ray

  H-LAB  the LABELS themselves are aspect-dependent: SpineM/HipL/TailBase are
         subcutaneous landmarks a human infers from the silhouette, so the
         training target moves with viewpoint. The net then learns that bias.
         -> the net is CONFIDENT while wrong (conf does not track the error)
         -> affects 'virtual' landmarks more than visible ones (Snout, ears)

Confidence is the referee between "the net knows it cannot see" (OCC/FORE) and
"the net is confidently wrong" (LAB/SWAP).
"""
import os, sys, glob, json
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from clirb.spatial import quality as q
from config import NODES, NODE_IDX
from projection import project_3d_to_2d
from corrector.world_alignment import fit_procrustes

DR = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
CAL = f'{DR}/calibrations/2026_05_04'
IN = 'results/corrector_v2/keys2d'
OUT = 'results/spatial_confound'
CONF = 0.5
ASP_BINS = 12
RNG = np.random.default_rng(0)

# bilateral pairs, by name
PAIRS = []
for i, n in enumerate(NODES):
    if n.endswith('L') and (n[:-1] + 'R') in NODES:
        PAIRS.append((i, NODE_IDX[n[:-1] + 'R'], n[:-1]))
# landmarks a human must INFER from the silhouette vs ones directly visible
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


def collect():
    P, K, D, R, T = q.load_calib(CAL)
    cams = np.stack([-R[c].T @ T[c] for c in range(3)])
    recs = []
    for f in sorted(glob.glob(f'{IN}/*.npz')):
        rat, session = os.path.basename(f)[:-4].split('__')
        with np.load(f) as z:
            k = z['k']; k2 = z['keys2d']
        ap = f'{DR}/{rat}/{session}/processed/aligned_data.npz'
        dnp = f'{DR}/{rat}/{session}/processed/sleap_dannce_keys.npz'
        if not (os.path.isfile(ap) and os.path.isfile(dnp)):
            continue
        with np.load(ap, allow_pickle=True) as z:
            dk = z['dannce_idx_for_sleap_cams_keys']
        with np.load(dnp, allow_pickle=True) as z:
            dn_raw = z['dannce_keys_3D']
        dn = (dn_raw.squeeze(axis=1).transpose(0, 2, 1) if dn_raw.ndim == 4
              else np.transpose(dn_raw, [0, 2, 1]))
        ok = k < len(dk)
        k, k2 = k[ok], k2[ok]
        tgt = dn[np.clip(dk[k], 0, len(dn) - 1)].astype(np.float64)

        und = q.undistort_all(np.nan_to_num(k2[..., :2]).astype(np.float64), K, D)
        X = q.triangulate_batch(und, P)
        good = (np.isfinite(X).all(axis=(1, 2)) & np.isfinite(tgt).all(axis=(1, 2)))
        rad = np.linalg.norm(X - X.mean(axis=1, keepdims=True), axis=-1).max(axis=1)
        good &= rad < 250                      # drop DLT blow-ups
        if good.sum() < 400:
            continue
        X, tgt, k2 = X[good], tgt[good], k2[good]
        # DANNCE -> calibration frame (so it can be reprojected into the cameras)
        fit = robust_proc(tgt, X)
        dn_cal = fit['apply'](tgt)

        com = dn_cal[:, :, :2].mean(axis=1)
        SF = dn_cal[:, NODE_IDX['SpineF'], :2]; SM = dn_cal[:, NODE_IDX['SpineM'], :2]
        head = np.degrees(np.arctan2(SF[:, 1] - SM[:, 1], SF[:, 0] - SM[:, 0]))
        for c in range(3):
            bear = np.degrees(np.arctan2(com[:, 1] - cams[c][1], com[:, 0] - cams[c][0]))
            aspect = wrap(head - bear)
            proj = project_3d_to_2d(dn_cal.reshape(-1, 3), K[c], R[c], T[c],
                                    D[c]).reshape(len(dn_cal), 23, 2)
            det = k2[:, c, :, :2].astype(np.float64)
            cf = k2[:, c, :, 2]
            res = det - proj
            dist_cam = np.linalg.norm(dn_cal - cams[c][None, None, :], axis=-1)
            keep = (cf >= CONF) & np.isfinite(res).all(axis=-1) & \
                   (np.linalg.norm(res, axis=-1) < 120)
            recs.append(dict(rat=rat, session=session, cam=c, det=det, proj=proj,
                             res=res, conf=cf, keep=keep, aspect=aspect,
                             dist=dist_cam))
        print(f'  {rat}/{session}: {good.sum()} frames', flush=True)
    return recs


def main():
    recs = collect()
    print(f'\n{len(recs)} (session, camera) blocks')
    res = {}

    # ---------------------------------------------------------------- H-OCC
    print('\n' + '=' * 76)
    print('H-OCC  is the FARTHER member of each bilateral pair worse?')
    print('=' * 76)
    rows = []
    for pl, pr, nm in PAIRS:
        near_e, far_e, near_c, far_c = [], [], [], []
        for r in recs:
            m = r['keep'][:, pl] & r['keep'][:, pr]
            if not m.any():
                continue
            dl, dr = r['dist'][m, pl], r['dist'][m, pr]
            el = np.linalg.norm(r['res'][m, pl], axis=-1)
            er = np.linalg.norm(r['res'][m, pr], axis=-1)
            cl, cr = r['conf'][m, pl], r['conf'][m, pr]
            l_is_far = dl > dr
            far_e.append(np.where(l_is_far, el, er)); near_e.append(np.where(l_is_far, er, el))
            far_c.append(np.where(l_is_far, cl, cr)); near_c.append(np.where(l_is_far, cr, cl))
        if not far_e:
            continue
        fe = np.concatenate(far_e); ne = np.concatenate(near_e)
        fc = np.concatenate(far_c); nc = np.concatenate(near_c)
        rows.append(dict(pair=nm, n=len(fe), near_px=np.median(ne), far_px=np.median(fe),
                         ratio=np.median(fe) / max(np.median(ne), 1e-9),
                         near_conf=np.median(nc), far_conf=np.median(fc)))
    t = pd.DataFrame(rows).sort_values('ratio', ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    res['occ_median_ratio'] = float(t.ratio.median())
    res['occ_conf_drop'] = float((t.near_conf - t.far_conf).median())
    print(f'\n  median far/near error ratio = {t.ratio.median():.3f}')
    print(f'  median confidence drop near->far = {(t.near_conf - t.far_conf).median():+.4f}')

    # --------------------------------------------------------------- H-SWAP
    print('\n' + '=' * 76)
    print('H-SWAP  would swapping L/R reduce the residual?')
    print('=' * 76)
    rows = []
    for pl, pr, nm in PAIRS:
        asis, swap = [], []
        for r in recs:
            m = r['keep'][:, pl] & r['keep'][:, pr]
            if not m.any():
                continue
            a = (np.linalg.norm(r['det'][m, pl] - r['proj'][m, pl], axis=-1) +
                 np.linalg.norm(r['det'][m, pr] - r['proj'][m, pr], axis=-1))
            s = (np.linalg.norm(r['det'][m, pl] - r['proj'][m, pr], axis=-1) +
                 np.linalg.norm(r['det'][m, pr] - r['proj'][m, pl], axis=-1))
            asis.append(a); swap.append(s)
        if not asis:
            continue
        a = np.concatenate(asis); s = np.concatenate(swap)
        rows.append(dict(pair=nm, n=len(a), asis_px=np.median(a), swap_px=np.median(s),
                         frac_swap_better=float((s < a).mean())))
    t = pd.DataFrame(rows).sort_values('frac_swap_better', ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    res['swap_frac'] = float(t.frac_swap_better.median())
    print(f'\n  median fraction of frames where a swap would be better: '
          f'{t.frac_swap_better.median():.3f}  (0.5 = chance, <<0.5 = no swaps)')

    # ---------------------------------------------------- aspect PROFILE shape
    print('\n' + '=' * 76)
    print('SHAPE  where in the aspect cycle does error peak?')
    print('  side-on peaks (|asp|~90)  -> occlusion;  0/180 peaks -> foreshortening')
    print('=' * 76)
    edges = np.linspace(-180, 180, ASP_BINS + 1)
    ctr = (edges[:-1] + edges[1:]) / 2
    prof = np.full((23, ASP_BINS), np.nan)
    cprof = np.full((23, ASP_BINS), np.nan)
    for j in range(23):
        for b in range(ASP_BINS):
            e, cc = [], []
            for r in recs:
                m = r['keep'][:, j] & (np.digitize(r['aspect'], edges) - 1 == b)
                if m.any():
                    e.append(np.linalg.norm(r['res'][m, j], axis=-1)); cc.append(r['conf'][m, j])
            if e and sum(len(x) for x in e) > 100:
                prof[j, b] = np.median(np.concatenate(e))
                cprof[j, b] = np.median(np.concatenate(cc))
    pooled = np.nanmedian(prof, axis=0)
    print('  aspect bin centre :', ' '.join(f'{c:+6.0f}' for c in ctr))
    print('  pooled median err :', ' '.join(f'{v:6.2f}' for v in pooled))
    print('  pooled median conf:', ' '.join(f'{v:6.3f}' for v in np.nanmedian(cprof, axis=0)))
    side = np.nanmean(pooled[(np.abs(ctr) > 60) & (np.abs(ctr) < 120)])
    endon = np.nanmean(pooled[(np.abs(ctr) < 30) | (np.abs(ctr) > 150)])
    print(f'\n  side-on (|asp| 60-120) mean = {side:.2f} px')
    print(f'  end-on  (|asp|<30 or >150) mean = {endon:.2f} px')
    print(f'  ratio side/end = {side/max(endon,1e-9):.3f}  '
          f'({"OCCLUSION-like" if side>endon else "FORESHORTENING-like"})')
    res['side_px'] = float(side); res['end_px'] = float(endon)
    pd.DataFrame(prof, index=NODES, columns=np.round(ctr).astype(int)).to_csv(
        f'{OUT}/aspect_profile_by_keypoint.csv')

    # ----------------------------------------------------------------- H-LAB
    print('\n' + '=' * 76)
    print('H-LAB  does CONFIDENCE track the error? (does the net know?)')
    print('=' * 76)
    from scipy import stats
    rows = []
    for j in range(23):
        e, c = [], []
        for r in recs:
            m = r['keep'][:, j]
            if m.any():
                e.append(np.linalg.norm(r['res'][m, j], axis=-1)); c.append(r['conf'][m, j])
        if not e:
            continue
        e = np.concatenate(e); c = np.concatenate(c)
        rho, _ = stats.spearmanr(c, e)
        amp = np.nanmax(prof[j]) - np.nanmin(prof[j]) if np.isfinite(prof[j]).sum() > 6 else np.nan
        rows.append(dict(kp=NODES[j], n=len(e), med_px=np.median(e), conf=np.median(c),
                         rho_conf_err=rho, aspect_amp=amp,
                         kind='virtual' if NODES[j] in VIRTUAL else 'visible'))
    t = pd.DataFrame(rows).sort_values('aspect_amp', ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    t.to_csv(f'{OUT}/aspect_mechanism_by_keypoint.csv', index=False)
    print(f'\n  median rho(confidence, error) = {t.rho_conf_err.median():+.3f}')
    print('  aspect amplitude by landmark kind (median px):')
    print(t.groupby('kind').aspect_amp.median().round(2).to_string())
    res['rho_conf_err'] = float(t.rho_conf_err.median())
    res['amp_virtual'] = float(t[t.kind == 'virtual'].aspect_amp.median())
    res['amp_visible'] = float(t[t.kind == 'visible'].aspect_amp.median())

    json.dump(res, open(f'{OUT}/stage11_results.json', 'w'), indent=2)
    print(f'\nwrote {OUT}/stage11_results.json')


if __name__ == '__main__':
    main()
