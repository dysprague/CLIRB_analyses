"""Per-frame truth-free tracking-quality cache.

Writes data/cache_quality/<rat>/<session>.npz with per-frame vectors only
(~1 MB/session). Deliberately a SEPARATE cache with its own version, so the
11 GB raw session_cache (CACHE_VERSION=2) is never invalidated.
"""
import os, sys, glob
import numpy as np

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb.spatial import quality as q
from clirb.spatial.geometry import ARENA

QUALITY_VERSION = 2   # v2: FIXED the processed/acquisition frame-basis mismatch
OUT_ROOT = 'data/cache_quality'


def out_path(rat, session):
    return os.path.join(OUT_ROOT, rat, f'{session}.npz')


def build_one(rat, session, force=False):
    op = out_path(rat, session)
    if os.path.isfile(op) and not force:
        try:
            with np.load(op) as z:
                if int(z['quality_version']) == QUALITY_VERSION:
                    return 'hit'
        except Exception:
            pass
    c = sc.get_raw_session(rat, session, want_dual=True)
    if c is None:
        return 'no-raw'
    triang = c['triang_3d']
    com = triang.mean(axis=1)
    out = dict(quality_version=np.int64(QUALITY_VERSION),
               n_frames=np.int64(len(triang)),
               theta=ARENA.theta(com[:, :2]).astype(np.float32),
               radius=ARENA.radius_of(com[:, :2]).astype(np.float32),
               com_z=com[:, 2].astype(np.float32),
               bone_cv=q.bone_length_cv(triang),
               nan_frac=np.isnan(triang).mean(axis=(1, 2)).astype(np.float32))
    # jitter needs egocentric coords
    ident = {'feature_means': np.zeros(69), 'pc_weights': np.eye(69)}
    _, rot = ev.project_session_to_pcs(triang, ident, flip_z=True)
    out['jitter'] = q.jitter(rot)

    cd = q.calib_dir(rat, session)
    fm = np.asarray(c.get('frame_mapping_cam0', []), dtype=int)
    if c.get('has_keys2d', False) and cd is not None and len(fm):
        k2 = c['sleap_keys_2D']
        T = len(triang)
        try:
            # keys_2D is on the PROCESSED-frame basis; triang on the ACQUISITION
            # basis. Bridge with frame_mapping_cam0, then scatter back.
            resid, per_cam, ncam, fov = q.reprojection_residual_aligned(
                k2, triang, fm, cd)
            out['reproj_px'] = q.scatter_to_acquisition(resid, fm, T)
            out['conf_per_cam'] = q.scatter_to_acquisition(per_cam, fm, T)
            out['mean_conf'] = np.nanmean(out['conf_per_cam'], axis=1).astype(np.float32)
            nc = q.scatter_to_acquisition(ncam.astype(np.float32), fm, T)
            out['n_cams_conf50'] = nc
            out['fov_margin'] = q.scatter_to_acquisition(fov, fm, T)
        except Exception as e:
            print(f'    aligned quality failed {rat}/{session}: {e!r}')
            out['reproj_px'] = np.full(T, np.nan, np.float32)
            out['mean_conf'] = np.full(T, np.nan, np.float32)
            out['conf_per_cam'] = np.full((T, 3), np.nan, np.float32)
            out['n_cams_conf50'] = np.full(T, np.nan, np.float32)
            out['fov_margin'] = np.full((T, 3), np.nan, np.float32)
    else:
        T = len(triang)
        out['reproj_px'] = np.full(T, np.nan, np.float32)
        out['mean_conf'] = np.full(T, np.nan, np.float32)
        out['conf_per_cam'] = np.full((T, 3), np.nan, np.float32)
        out['n_cams_conf50'] = np.full(T, np.nan, np.float32)
        out['fov_margin'] = np.full((T, 3), np.nan, np.float32)

    os.makedirs(os.path.dirname(op), exist_ok=True)
    tmp = op + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, op)
    return 'built'


def main():
    sessions = []
    for rat in ['R4', 'R5', 'R6']:
        for f in sorted(glob.glob(f'data/cache/{rat}/*.npz')):
            sessions.append((rat, os.path.basename(f)[:-4]))
    counts = {}
    for i, (rat, s) in enumerate(sessions):
        r = build_one(rat, s)
        counts[r] = counts.get(r, 0) + 1
        if i % 10 == 0:
            print(f'  [{i+1}/{len(sessions)}] {rat}/{s} -> {r}', flush=True)
    print('done:', counts)


if __name__ == '__main__':
    main()
