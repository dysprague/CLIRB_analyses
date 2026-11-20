"""Is DANNCE's axial foreshortening coming from the COM/volume, or from the
keypoint predictions inside it?

sDANNCE predicts keypoints inside a fixed-size, WORLD-AXIS-ALIGNED box centred
on the COM. So the box, if it is the culprit, has to show up in world axes as a
hard ceiling on |keypoint - COM| -- a truncated offset distribution with a
pile-up at the boundary. The network's own output distribution, if that is the
culprit, shrinks smoothly with no edge.

Three outcomes this distinguishes:
  box_changed     the ceiling on |offset| differs between dates -> vol_size was
                  changed between runs
  box_clipping    the ceiling is the same on both dates and sharp, and the
                  later date has far more keypoints sitting against it -> the
                  box did not change but the animal outgrew it
  keypoint_space  no hard ceiling on either date and offsets well inside it,
                  but systematically smaller later -> the predictions moved,
                  not the box

Everything here is in DANNCE's OWN raw output coordinates, read straight from
  DANNCE/predict_results_finetune/com3d_used.mat   the volume centre per frame
  DANNCE/predict_results_finetune/save_data_AVG.mat the final keypoints
  DANNCE/predict_results_finetune/init_save_data_AVG.mat the first-pass keypoints
No SLEAP, no Procrustes, no resampling -- this is a within-DANNCE question.

Writes results/newcal/dannce_volume_check.csv
"""
import os, sys
import numpy as np
import pandas as pd
import scipy.io as sio

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)
from config import NODE_IDX as NI

DATA_ROOT = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
OUT = 'results/newcal'
CACHE = f'{OUT}/dannce_volume_cache'
os.makedirs(CACHE, exist_ok=True)
SESSIONS = [('R7', '2026_10_23_1'), ('R8', '2026_10_23_1'), ('R9', '2026_10_23_1'),
            ('R7', '2026_10_26_1'), ('R8', '2026_10_26_1'), ('R9', '2026_10_26_1')]
AX_PAIR = (NI['SpineF'], NI['TailBase'])


def _pred(m):
    """(n, 23, 3) from a dannce save_data_AVG mat, whatever the key layout."""
    for k in ('pred', 'predictions', 'data'):
        if k in m:
            a = np.asarray(m[k])
            break
    else:
        cand = [k for k in m if not k.startswith('__')]
        raise KeyError(f'no prediction key among {cand}')
    if a.ndim == 3 and a.shape[1] == 3:        # (n, 3, 23)
        a = a.transpose(0, 2, 1)
    return np.asarray(a, dtype=np.float64)


def load(rat, session):
    cp = f'{CACHE}/{rat}_{session}.npz'
    if os.path.isfile(cp):
        z = np.load(cp)
        return z['com'], z['kp'], z['kp0']
    d = f'{DATA_ROOT}/{rat}/{session}/dannce/DANNCE/predict_results_finetune'
    com = np.asarray(sio.loadmat(f'{d}/com3d_used.mat', simplify_cells=True)['com'],
                     dtype=np.float64)
    kp = _pred(sio.loadmat(f'{d}/save_data_AVG.mat', simplify_cells=True))
    try:
        kp0 = _pred(sio.loadmat(f'{d}/init_save_data_AVG.mat', simplify_cells=True))
    except Exception:
        kp0 = np.full_like(kp, np.nan)
    n = min(len(com), len(kp), len(kp0))
    com, kp, kp0 = com[:n], kp[:n], kp0[:n]
    np.savez_compressed(cp, com=com.astype(np.float32), kp=kp.astype(np.float32),
                        kp0=kp0.astype(np.float32))
    return com, kp, kp0


def stats(com, kp, tag):
    off = kp - com[:, None, :]                       # (n, 23, 3), world axes
    fin = np.isfinite(off).all(axis=-1)
    out = {}
    for ax, name in enumerate('xyz'):
        v = np.abs(off[:, :, ax][fin])
        out[f'{tag}_{name}_p50'] = float(np.percentile(v, 50))
        out[f'{tag}_{name}_p99'] = float(np.percentile(v, 99))
        out[f'{tag}_{name}_p999'] = float(np.percentile(v, 99.9))
        out[f'{tag}_{name}_max'] = float(v.max())
        # clipping signature: what fraction of offsets sit in the outer 2 % of
        # the observed range, and how sharply the distribution ends
        hi = out[f'{tag}_{name}_max']
        out[f'{tag}_{name}_frac_at_edge'] = float((v > 0.98 * hi).mean())
        out[f'{tag}_{name}_p999_over_max'] = out[f'{tag}_{name}_p999'] / hi
    r = np.linalg.norm(off, axis=-1)[fin]
    out[f'{tag}_r_p50'] = float(np.percentile(r, 50))
    out[f'{tag}_r_max'] = float(r.max())
    # body length measured inside the volume, to tie back to the earlier result
    a, b = AX_PAIR
    L = np.linalg.norm(kp[:, a] - kp[:, b], axis=-1)
    out[f'{tag}_spineF_tb'] = float(np.nanmedian(L))
    out[f'{tag}_com_to_kp_spread'] = float(np.nanmedian(
        np.linalg.norm(kp - com[:, None, :], axis=-1).mean(axis=1)))
    return out


def main():
    rows = []
    for rat, session in SESSIONS:
        try:
            com, kp, kp0 = load(rat, session)
        except Exception as e:
            print(f'{rat}/{session}: ERROR {e!r}', flush=True)
            continue
        rec = dict(rat=rat, session=session, n_frames=len(com),
                   com_x_sd=float(np.nanstd(com[:, 0])),
                   com_y_sd=float(np.nanstd(com[:, 1])),
                   com_z_med=float(np.nanmedian(com[:, 2])))
        rec.update(stats(com, kp, 'final'))
        if np.isfinite(kp0).any():
            rec.update(stats(com, kp0, 'init'))
        rows.append(rec)
        print(f'{rat}/{session}: |offset| max x/y/z = '
              f'{rec["final_x_max"]:.1f}/{rec["final_y_max"]:.1f}/{rec["final_z_max"]:.1f} mm  '
              f'SpineF-TailBase={rec["final_spineF_tb"]:.1f} mm', flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f'{OUT}/dannce_volume_check.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n=== volume geometry: |keypoint - COM| per world axis (mm) ===')
    cols = ['rat', 'session'] + [f'final_{a}_{s}' for a in 'xyz'
                                 for s in ('p50', 'p99', 'max')]
    print(d[cols].round(1).to_string(index=False))
    print('\n=== clipping signature (fraction of offsets within 2% of the max) ===')
    print(d[['rat', 'session'] + [f'final_{a}_frac_at_edge' for a in 'xyz']]
          .round(5).to_string(index=False))
    print('\n=== body size inside the volume ===')
    print(d[['rat', 'session', 'final_spineF_tb', 'final_com_to_kp_spread',
             'init_spineF_tb', 'init_com_to_kp_spread']].round(2).to_string(index=False))
    print(f'\nwrote {OUT}/dannce_volume_check.csv')


if __name__ == '__main__':
    main()
