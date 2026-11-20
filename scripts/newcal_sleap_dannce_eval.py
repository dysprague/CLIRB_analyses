"""Post-recalibration SLEAP-vs-DANNCE tracking evaluation.

Question: after the arena/environment change and the 2026_10_14 recalibration,
is SLEAP still tracking as well (relative to DANNCE, treated as ground truth)
as it was in the September sessions for R4/R5/R6?

Two eras, same SLEAP model (n=9975 TRT FP32), same sDANNCE finetune + COM
checkpoint, online corrector OFF in both -- the only pipeline change is the
camera calibration (2026_08_28 -> 2026_10_14) and the physical environment:

  sept_baseline : every fully-processed 2026-09 session of R4, R5, R6
  post_recal    : R5/2026_10_14_1 and the fully-processed R7/R8/R9 sessions

Metrics per session (both modalities on the SLEAP 20 Hz frame grid):

  A. RAW 3D WORLD SPACE (stage20 / SPATIAL_CONFOUND_RESULTS.md section 8q
     convention).  One session-level rigid+isotropic-scale Procrustes fit maps
     DANNCE -> SLEAP's world frame, estimated on a spatially balanced,
     outlier-trimmed sample, then applied to every frame.  Removes only the
     rig-to-rig calibration offset, never references a single keypoint, so no
     keypoint's error leaks into another's.  Reports per-keypoint RMSE, overall
     RMSE, per-frame RMSE quantiles, and a gross-failure frame fraction.

     NOTE vs stage20: the sector-balanced sample is taken around each SESSION's
     OWN fitted COM centre rather than clirb.spatial.geometry.ARENA, whose
     centre/radius were fitted on R4/R5/R6 in the OLD arena and are not valid
     after the environment change.  Within a session this is the same
     "spatially balanced sample" idea; it just does not assume a fixed arena.

  B. EGOCENTRIC KEYPOINT SPACE (stage19 convention) -- z-flip (SLEAP only) ->
     normalize_skeleton -> 5-frame causal median, i.e. exactly the
     representation the online rule sees.  SpineM is 0 by construction.

  C. PC SPACE -- the 2-PC space template matching actually runs in.  Computed
     twice:
       common basis  : one fixed reference template basis (default
                       R5_template_8, the template in force for BOTH R5 bracket
                       sessions) applied to every session, so PC numbers are
                       commensurable across rats and eras.
       native basis  : the session's own template, where it has one (R7/R8/R9
                       have none yet).  Also expressed relative to that
                       template's own matching tolerance (bounds_to_use), which
                       is what decides whether an error changes a match.

  D. TRUTH-FREE, PER-MODALITY QUALITY (clirb.spatial.quality) -- needed to
     attribute a disagreement to one modality rather than the other:
       SLEAP : median reprojection residual (px) of the stored 3D pose against
               SLEAP's own 2D detections (respecting the processed-vs-
               acquisition frame-basis bridge), mean 2D detection confidence,
               per-frame bone-length CV.
       DANNCE: per-frame bone-length CV.
     On frames where the two modalities grossly disagree, whichever modality is
     violating its own rigidity / 2D self-consistency is the one that failed.

Reads only from olveczky_lab (via session_cache, which caches locally).
Writes results/newcal/*.csv .
"""
import os, sys, json
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev, templates as tp
from clirb.spatial import dual as du
from clirb.spatial import quality as ql
from corrector.world_alignment import fit_procrustes
from config import NODE_IDX, EDGES

DATA_ROOT = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
OUT = 'results/newcal'
FRAMEDIR = os.path.join(OUT, 'frame_traces')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FRAMEDIR, exist_ok=True)

KP_NAMES = [n for n, _ in sorted(NODE_IDX.items(), key=lambda kv: kv[1])]
N_KP = len(KP_NAMES)

SECTOR_W = 30
N_PER_SECTOR = 300
FIT_RESIDUAL_GUARD_MM = 60.0
GROSS_FRAME_RMSE_MM = 40.0      # per-frame RMSE above this = gross disagreement

COMMON_TEMPLATE = os.path.join(DATA_ROOT, 'R5', 'templates', 'R5_template_8.npz')


# --------------------------------------------------------------------------- #
# Procrustes helpers (stage20's robust_proc, verbatim convention)
# --------------------------------------------------------------------------- #
def robust_proc(A, B, iters=5, keep=0.7):
    """apply(A) ~ B.  robust_proc(dannce, sleap) maps DANNCE -> SLEAP."""
    idx = np.arange(len(A)); f = None
    for _ in range(iters):
        f = fit_procrustes(A[idx], B[idx], try_z_flip=True)
        r = np.linalg.norm(f['apply'](A) - B, axis=-1).mean(axis=1)
        idx = np.where(r <= np.quantile(r, keep))[0]
        if len(idx) < 200:
            break
    return f


def balanced_sample(com_xy, n_per_sector=N_PER_SECTOR, rng=None):
    """Spatially balanced frame sample: equal quota per angular sector around
    the session's OWN COM centroid (does not assume the fixed R4/R5/R6 arena)."""
    rng = rng or np.random.default_rng(0)
    c = np.nanmedian(com_xy, axis=0)
    th = np.degrees(np.arctan2(com_xy[:, 1] - c[1], com_xy[:, 0] - c[0]))
    edges = np.linspace(-180, 180, int(round(360 / SECTOR_W)) + 1)
    sec = np.clip(np.digitize(th, edges) - 1, 0, len(edges) - 2)
    samp = []
    for k in range(len(edges) - 1):
        ii = np.where(sec == k)[0]
        if len(ii):
            samp.extend(rng.choice(ii, size=min(len(ii), n_per_sector), replace=False))
    return np.array(sorted(samp), dtype=int), c


# --------------------------------------------------------------------------- #
# Metric blocks
# --------------------------------------------------------------------------- #
def bone_cv(kp):
    """Mean across-edge coefficient of variation of bone length."""
    cvs = []
    for a, b in EDGES:
        d = np.linalg.norm(kp[:, a] - kp[:, b], axis=-1)
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) > 100:
            cvs.append(d.std() / d.mean())
    return float(np.mean(cvs)) if cvs else np.nan


def pc_block(sl3, dn3, tmpl, prefix):
    """PC-space agreement in one template basis.  SLEAP z-flipped, DANNCE not,
    exactly as the online rule does it."""
    s_pcs, _ = ev.project_session_to_pcs(sl3, tmpl, flip_z=True)
    d_pcs, _ = ev.project_session_to_pcs(dn3, tmpl, flip_z=False)
    n = min(len(s_pcs), len(d_pcs))
    s_pcs, d_pcs = s_pcs[:n], d_pcs[:n]
    pcs = [int(v) for v in np.asarray(tmpl['pcs_to_use']).ravel()[:2]]
    out = {}
    for j, pci in enumerate(pcs):
        a, b = s_pcs[:, pci], d_pcs[:, pci]
        err = a - b
        out[f'{prefix}_pc{j}_rmse'] = float(np.sqrt(np.mean(err ** 2)))
        out[f'{prefix}_pc{j}_bias'] = float(np.mean(err))
        out[f'{prefix}_pc{j}_sd_dannce'] = float(np.std(b))
        out[f'{prefix}_pc{j}_sd_sleap'] = float(np.std(a))
        out[f'{prefix}_pc{j}_r'] = float(np.corrcoef(a, b)[0, 1])
        # normalised: tracking error as a fraction of the behaviour's own spread
        out[f'{prefix}_pc{j}_nrmse'] = out[f'{prefix}_pc{j}_rmse'] / max(np.std(b), 1e-9)
    out[f'{prefix}_pc01_rmse'] = float(np.sqrt(np.mean(
        (s_pcs[:, pcs] - d_pcs[:, pcs]) ** 2)))
    # error relative to this template's own matching tolerance band
    bt = np.asarray(tmpl.get('bounds_to_use', np.array([])))
    if bt.size:
        band = bt.mean(axis=0)   # mean half-width per PC over the 30 frames
        for j in range(min(2, len(band))):
            out[f'{prefix}_pc{j}_err_over_bound'] = (
                out[f'{prefix}_pc{j}_rmse'] / max(float(band[j]), 1e-9))
    return out


def best_lag(sl, dn_w, max_lag=25):
    """Integer frame lag minimising mean COM distance between the modalities.

    A non-zero optimum means the DANNCE->SLEAP frame resampling
    (`dannce_idx_for_sleap_cams`) is off, which would masquerade as tracking
    error.  Returns (best_lag, mean_com_dist_at_0, mean_com_dist_at_best)."""
    a = sl.mean(axis=1)
    b = dn_w.mean(axis=1)
    out = {}
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            x, y = a[L:], b[:len(b) - L] if L else b
        else:
            x, y = a[:len(a) + L], b[-L:]
        m = min(len(x), len(y))
        out[L] = float(np.nanmean(np.linalg.norm(x[:m] - y[:m], axis=-1)))
    L = min(out, key=out.get)
    return L, out[0], out[L]


def dannce_vs_sleap2d(rat, session, dn_world, keys2d_proc, fm_proc,
                      conf_thresh=0.8):
    """Median pixel distance between the reprojected DANNCE pose (already in
    SLEAP world coords) and SLEAP's high-confidence 2D detections, per
    ACQUISITION frame (NaN where no processed frame maps to it).

    Uses the processed-vs-acquisition frame bridge documented in
    clirb/spatial/quality.py: keys_2D row k corresponds to acquisition frame
    fm_proc[k]."""
    from corrector.data_world_2d import load_session_calibration, project_3d_to_2d_batch
    calib, _ = load_session_calibration(rat, session)
    n_acq = len(dn_world)
    fm = np.asarray(fm_proc, dtype=int)
    sel = (fm >= 0) & (fm < n_acq)
    fm_ok = fm[sel]
    det = np.asarray(keys2d_proc)[sel][..., :2].astype(np.float64)   # (K,3,23,2)
    conf = np.asarray(keys2d_proc)[sel][..., 2].astype(np.float64)   # (K,3,23)
    X = dn_world[fm_ok]                                              # (K,23,3)
    err = np.full(conf.shape, np.nan)
    for c in range(min(3, len(calib))):
        pr = project_3d_to_2d_batch(X, calib[c])                     # (K,23,2)
        e = np.linalg.norm(pr - det[:, c], axis=-1)
        e[conf[:, c] < conf_thresh] = np.nan
        err[:, c] = e
    with np.errstate(all='ignore'):
        per = np.nanmedian(err.reshape(len(err), -1), axis=1)
    out = np.full(n_acq, np.nan, dtype=np.float32)
    out[fm_ok] = per
    return out


def session_metrics(rat, session, era, rng, common_tmpl):
    raw = sc.get_raw_session(rat, session, want_dual=True)
    if raw is None or not raw.get('has_dual', False):
        return None, None, None
    d3 = du.dannce_aligned(raw)
    if d3 is None:
        return None, None, None
    n = min(len(raw['triang_3d']), len(d3))
    if n < 1000:
        return None, None, None
    sl = np.asarray(raw['triang_3d'][:n], dtype=np.float32)
    dn = np.asarray(d3[:n], dtype=np.float32)

    rec = dict(rat=rat, session=session, era=era, n_frames=int(n))

    # ---- A. world space ---------------------------------------------------
    com = sl.mean(axis=1)
    # Occupied extent of the SLEAP COM -- did the usable arena move/shrink?
    for ax, name in enumerate('xyz'):
        v = com[:, ax]
        rec[f'com_{name}_p01'] = float(np.nanpercentile(v, 1))
        rec[f'com_{name}_p50'] = float(np.nanpercentile(v, 50))
        rec[f'com_{name}_p99'] = float(np.nanpercentile(v, 99))
    samp, centre = balanced_sample(com[:, :2], rng=rng)
    rec['com_centre_x'], rec['com_centre_y'] = float(centre[0]), float(centre[1])
    rec['com_radius_med'] = float(np.nanmedian(
        np.hypot(com[:, 0] - centre[0], com[:, 1] - centre[1])))
    if len(samp) < 500:
        return None, None, None
    fit = robust_proc(dn[samp], sl[samp])
    dn_w = fit['apply'](dn)
    sq = ((sl - dn_w) ** 2).sum(axis=-1)          # (n, 23) mm^2
    dist = np.sqrt(sq)                            # (n, 23) mm
    frame_rmse = np.sqrt(sq.mean(axis=1))         # RMS over keypoints, per frame
    # Core-body measure: the MEDIAN keypoint error within each frame.  A single
    # fly-away keypoint cannot move it, so comparing this with frame_rmse
    # separates "the whole pose got worse" from "a few keypoints fly off".
    frame_med = np.median(dist, axis=1)
    rec.update(
        world_frame_medkp_p50=float(np.percentile(frame_med, 50)),
        world_frame_medkp_p90=float(np.percentile(frame_med, 90)),
        fit_scale=float(fit['s']), fit_z_flipped=bool(fit['z_flipped']),
        fit_residual_mm=float(fit['residual']),
        world_rmse_mm=float(np.sqrt(sq.mean())),
        world_frame_rmse_p50=float(np.percentile(frame_rmse, 50)),
        world_frame_rmse_p90=float(np.percentile(frame_rmse, 90)),
        world_frame_rmse_p99=float(np.percentile(frame_rmse, 99)),
        world_gross_frac=float(np.mean(frame_rmse > GROSS_FRAME_RMSE_MM)),
    )
    world_kp = np.sqrt(sq.mean(axis=0))           # (23,) mm
    rec['world_kp_med_mm'] = float(np.median(dist))   # median over all (frame, kp)
    # Per-keypoint "flyaway" rate: fraction of frames that keypoint sits more
    # than 100 mm from DANNCE's estimate of the same keypoint.
    kp_fly = (dist > 100.0).mean(axis=0)
    rec['flyaway_frac_any_kp'] = float(np.mean((dist > 100.0).any(axis=1)))
    rec['flyaway_frac_mean_kp'] = float(kp_fly.mean())
    L, d0, dL = best_lag(sl, dn_w)
    rec['best_frame_lag'] = int(L)
    rec['com_dist_lag0_mm'] = float(d0)
    rec['com_dist_bestlag_mm'] = float(dL)
    excluded = (fit['residual'] > FIT_RESIDUAL_GUARD_MM or
                rec['world_rmse_mm'] > 2 * FIT_RESIDUAL_GUARD_MM)
    rec['excluded'] = bool(excluded)

    # ---- B. egocentric space ----------------------------------------------
    _, s_rot = ev.project_session_to_pcs(sl, common_tmpl, flip_z=True)
    _, d_rot = ev.project_session_to_pcs(dn, common_tmpl, flip_z=False)
    esq = ((s_rot - d_rot) ** 2).sum(axis=-1)
    rec['ego_rmse_mm'] = float(np.sqrt(esq.mean()))
    ego_kp = np.sqrt(esq.mean(axis=0))

    # ---- C. PC space -------------------------------------------------------
    rec.update(pc_block(sl, dn, common_tmpl, 'com'))
    native = tp.load_template(rat, session)
    rec['native_template'] = native['_file'] if native else ''
    if native is not None:
        rec.update(pc_block(sl, dn, native, 'nat'))

    # ---- D. truth-free, per-modality quality -------------------------------
    # Absolute skeleton scale per modality: a calibration scale error shows up
    # here as a shift in reconstructed body length, independent of any fit.
    SF, TB = NODE_IDX['SpineF'], NODE_IDX['TailBase']
    rec['torso_mm_sleap'] = float(np.nanmedian(
        np.linalg.norm(sl[:, SF] - sl[:, TB], axis=-1)))
    rec['torso_mm_dannce'] = float(np.nanmedian(
        np.linalg.norm(dn[:, SF] - dn[:, TB], axis=-1)))
    rec['torso_ratio_sleap_over_dannce'] = (
        rec['torso_mm_sleap'] / max(rec['torso_mm_dannce'], 1e-9))
    rec['bone_cv_sleap'] = bone_cv(sl)
    rec['bone_cv_dannce'] = bone_cv(dn)
    sl_bcv = ql.bone_length_cv(sl)        # (n,) per-frame rigidity violation
    dn_bcv = ql.bone_length_cv(dn)
    rec['bone_frame_cv_sleap_p50'] = float(np.nanmedian(sl_bcv))
    rec['bone_frame_cv_dannce_p50'] = float(np.nanmedian(dn_bcv))

    reproj = None
    dannce_px_frames = None
    if raw.get('has_keys2d', False):
        k2 = np.asarray(raw['sleap_keys_2D'])
        fm = np.asarray(raw['frame_mapping_cam0'], dtype=int)
        nproc = min(len(fm), len(k2))      # keys_2D is zero-padded past nproc
        cal_dir = ql.calib_dir(rat, session)
        if cal_dir is not None and nproc > 1000:
            try:
                resid, per_cam, ncam, fov = ql.reprojection_residual_aligned(
                    k2[:nproc], sl, fm[:nproc], cal_dir)
                rec['sleap_reproj_px_p50'] = float(np.nanmedian(resid))
                rec['sleap_reproj_px_p90'] = float(np.nanpercentile(resid, 90))
                rec['sleap_reproj_px_p99'] = float(np.nanpercentile(resid, 99))
                rec['sleap_conf_mean'] = float(np.nanmean(per_cam))
                for c in range(per_cam.shape[1]):
                    rec[f'sleap_conf_cam{c}'] = float(np.nanmean(per_cam[:, c]))
                rec['sleap_ncam_lt3_frac'] = float(np.mean(ncam < 3))
                rec['sleap_offimage_frac'] = float(np.mean(np.nanmin(fov, axis=1) < 0))
                # scatter back onto the acquisition grid for the gross-frame split
                reproj = ql.scatter_to_acquisition(resid, fm[:nproc], n)
            except Exception as e:
                print(f'    reproj failed for {rat}/{session}: {e!r}', flush=True)

            # Independent placement check on DANNCE: reproject the
            # Procrustes-mapped DANNCE pose into the 3 SLEAP cameras and compare
            # with SLEAP's HIGH-CONFIDENCE 2D detections (conf >= 0.8), which are
            # direct image evidence.  Biased in SLEAP's favour by construction
            # (SLEAP's own 3D was triangulated from exactly these points), so it
            # is read only as a gross-failure detector for DANNCE, never as a
            # fair SLEAP-vs-DANNCE accuracy comparison.
            try:
                dn_px = dannce_vs_sleap2d(rat, session, dn_w, k2[:nproc], fm[:nproc])
                rec['dannce_reproj_px_p50'] = float(np.nanmedian(dn_px))
                rec['dannce_reproj_px_p90'] = float(np.nanpercentile(dn_px, 90))
                rec['dannce_reproj_px_p99'] = float(np.nanpercentile(dn_px, 99))
                rec['dannce_badplace_frac'] = float(np.nanmean(dn_px > 50.0))
                if reproj is not None:
                    rec['sleap_badplace_frac'] = float(np.nanmean(reproj > 50.0))
                dannce_px_frames = dn_px
            except Exception as e:
                print(f'    dannce reproj failed for {rat}/{session}: {e!r}', flush=True)

    # Attribution on the grossly-disagreeing frames
    gross = frame_rmse > GROSS_FRAME_RMSE_MM
    ok = ~gross
    if gross.sum() > 20 and ok.sum() > 20:
        rec['gross_sleap_bonecv'] = float(np.nanmedian(sl_bcv[gross]))
        rec['ok_sleap_bonecv'] = float(np.nanmedian(sl_bcv[ok]))
        rec['gross_dannce_bonecv'] = float(np.nanmedian(dn_bcv[gross]))
        rec['ok_dannce_bonecv'] = float(np.nanmedian(dn_bcv[ok]))
        if reproj is not None:
            with np.errstate(all='ignore'):
                rec['gross_sleap_reproj_px'] = float(np.nanmedian(reproj[gross]))
                rec['ok_sleap_reproj_px'] = float(np.nanmedian(reproj[ok]))
        if dannce_px_frames is not None:
            with np.errstate(all='ignore'):
                rec['gross_dannce_reproj_px'] = float(np.nanmedian(dannce_px_frames[gross]))
                rec['ok_dannce_reproj_px'] = float(np.nanmedian(dannce_px_frames[ok]))
    # Per-frame traces, for the pooled distribution / time-series figures.
    np.savez_compressed(
        os.path.join(FRAMEDIR, f'{rat}_{session}.npz'),
        rat=rat, session=session, era=era,
        frame_rmse=frame_rmse.astype(np.float32),
        sleap_bone_cv=np.asarray(sl_bcv, dtype=np.float32),
        dannce_bone_cv=np.asarray(dn_bcv, dtype=np.float32),
        sleap_reproj_px=(reproj if reproj is not None
                         else np.full(n, np.nan, dtype=np.float32)),
        dannce_reproj_px=(dannce_px_frames if dannce_px_frames is not None
                          else np.full(n, np.nan, dtype=np.float32)),
        com_xy=com[:, :2].astype(np.float32),
    )
    return rec, (world_kp, kp_fly), ego_kp


# --------------------------------------------------------------------------- #
def main():
    inv = pd.read_csv(f'{OUT}/session_inventory.csv')
    inv = inv[inv.ok & inv.has_dual]
    common = dict(np.load(COMMON_TEMPLATE, allow_pickle=True))
    print(f'common PC basis: {os.path.basename(COMMON_TEMPLATE)}  '
          f'pcs_to_use={common["pcs_to_use"]}', flush=True)

    rng = np.random.default_rng(0)
    recs, wkp_rows, ekp_rows = [], [], []
    for i, r in enumerate(inv.itertuples(), 1):
        try:
            rec, wkp, ekp = session_metrics(r.rat, r.session, r.era, rng, common)
        except Exception as e:
            print(f'[{i}/{len(inv)}] {r.rat}/{r.session}: ERROR {e!r}', flush=True)
            continue
        if rec is None:
            print(f'[{i}/{len(inv)}] {r.rat}/{r.session}: unusable', flush=True)
            continue
        recs.append(rec)
        wkp_rows.append(dict(rat=r.rat, session=r.session, era=r.era,
                             n_frames=rec['n_frames'],
                             **{f'world_rmse_{k}': float(v) for k, v in zip(KP_NAMES, wkp[0])},
                             **{f'flyaway_{k}': float(v) for k, v in zip(KP_NAMES, wkp[1])}))
        ekp_rows.append(dict(rat=r.rat, session=r.session, era=r.era,
                             n_frames=rec['n_frames'],
                             **{f'ego_rmse_{k}': float(v) for k, v in zip(KP_NAMES, ekp)}))
        print(f'[{i}/{len(inv)}] {r.rat}/{r.session} ({r.era}) '
              f'world={rec["world_rmse_mm"]:.2f}mm ego={rec["ego_rmse_mm"]:.2f}mm '
              f'pc0={rec["com_pc0_rmse"]:.1f} pc1={rec["com_pc1_rmse"]:.1f} '
              f'fit_res={rec["fit_residual_mm"]:.1f} scale={rec["fit_scale"]:.4f}'
              + ('  EXCLUDED' if rec['excluded'] else ''), flush=True)

    pd.DataFrame(recs).to_csv(f'{OUT}/session_metrics.csv', index=False)
    pd.DataFrame(wkp_rows).to_csv(f'{OUT}/session_keypoint_world_rmse.csv', index=False)
    pd.DataFrame(ekp_rows).to_csv(f'{OUT}/session_keypoint_ego_rmse.csv', index=False)
    print(f'\nwrote {OUT}/session_metrics.csv ({len(recs)} sessions)')


if __name__ == '__main__':
    main()
