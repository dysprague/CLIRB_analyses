"""Replay the online HeadHold rear detector offline, and check it against what
the rig actually did during the task.

From 2026-10-26 the rig's reward rule is `HeadHoldRule` (campy-CLIRB
campy/behavior_rule.py), not `TemplateMatch`: it fires once when **ear_mean_z**
-- the same feature calibrated in the previous appendix -- stays inside
[head_min, head_max] for `hold_time`, with a `refractory`, and a latch that
must see the head drop below `reset_threshold` before it can fire again.  The
rig logs its own per-frame `head_height` and `triggered` flag to
`sleap/behavior_log.csv`, which makes an exact offline replay checkable.

Three comparisons, in increasing distance from the rig:

  A. SIGNAL FIDELITY.  Recompute ear_mean_z from the saved `triang_keys_3D` and
     compare it frame by frame with the logged `head_height`.  This isolates
     the pose/filter path from the state machine.

  B. STATE-MACHINE FIDELITY.  Replay HeadHoldRule's own logic over the saved
     poses and compare the resulting triggers with the logged ones.  Together
     with A this says whether the session is reproducible from what was saved.

  C. ONLINE vs OFFLINE REAR DETECTION.  Compare the online triggers with the
     rears an offline hysteresis detector finds -- on SLEAP, and on DANNCE as
     ground truth -- restricted to the task window.  Misses are classified by
     cause rather than pooled, because several are the rule working as
     designed (refractory, latch not re-armed, hold never reached).

FRAME BOOKKEEPING, which is where this analysis can silently go wrong:
  * `triang_keys_3D.npy` is on the CAM0 ACQUISITION grid and already contains
    the rig's linear interpolation for frames that were not processed in time.
    The rule saw exactly this sequence, so the replay walks it in order.
  * `behavior_log.csv` is on the PROCESSED-frame basis, one row per frame that
    was actually processed, and `frame_mapping.csv` bridges the two.  The
    offset between the log's 0-based counter and frame_mapping's 1-based
    `processed_frame` is verified here rather than assumed.
  * Frames the rig interpolated were passed to the rule with `check_time=False`
    -- they feed the height buffer and the reset latch but cannot fire.
  * `curr_time` is `max(acquisition_time)` over the three cameras of the
    processed frame being handled; interpolated frames emitted while handling
    that frame carry the same timestamp.
  * The task context is ON while `cam0_frame / fps` is inside the
    `context_switch_time` interval.  A trigger outside it is logged as
    `triggered=1` but delivers no reward ("would have triggered"), so only the
    in-task ones are rewards.

Writes results/newcal/online_replay_{summary,events}.csv
"""
import os, sys, json
import numpy as np
import pandas as pd
import yaml

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

from config import NODE_IDX
from scripts.newcal_rear_detector import detect_rears_fast, match_overlap

DATA_ROOT = '/home/yutaka-sprague/olveczky_lab/Lab/CLIRB/data'
OUT = 'results/newcal'
EARL, EARR = NODE_IDX['EarL'], NODE_IDX['EarR']


# --------------------------------------------------------------------------- #
def load_session(rat, session):
    sd = f'{DATA_ROOT}/{rat}/{session}/sleap'
    cfg = yaml.safe_load(open(f'{sd}/campy_config.yaml'))
    bp = cfg['behavior_params']
    rp = bp['reward_params']
    par = dict(head_min=float(rp['head_min']), head_max=float(rp['head_max']),
               hold_time=float(rp['hold_time']), refractory=float(rp['refractory']),
               reset_threshold=float(rp.get('reset_threshold', rp['head_min'])),
               prev_frames=5, fps=int(cfg['frameRate']),
               rule=bp.get('reward_rule'), task_id=int(cfg.get('task_id', 0)),
               switches=list(np.atleast_1d(bp['context_switch_time'])))
    triang = np.load(f'{sd}/triang_keys_3D.npy')
    fm = pd.read_csv(f'{sd}/frame_mapping.csv', skipinitialspace=True)
    bl = pd.read_csv(f'{sd}/behavior_log.csv', skipinitialspace=True)
    fm['acq_time'] = fm[['cam0_acquisition_time', 'cam1_acquisition_time',
                         'cam2_acquisition_time']].max(axis=1)
    return par, triang, fm, bl


def replay(triang, fm, par):
    """Walk the acquisition grid exactly as the online loop did.
    Returns (head_z per acquisition frame, fired flags, is_real mask)."""
    n = len(triang)
    real = np.zeros(n, bool)
    cam0 = fm.cam0_frame.to_numpy()
    ok = (cam0 >= 0) & (cam0 < n)
    real[cam0[ok]] = True
    # each acquisition frame carries the acq time of the processed frame that
    # emitted it -- the next real frame at or after it
    t = np.full(n, np.nan)
    t[cam0[ok]] = fm.acq_time.to_numpy()[ok]
    idx = np.where(real)[0]
    nxt = np.searchsorted(idx, np.arange(n), side='left')
    nxt = np.clip(nxt, 0, len(idx) - 1)
    curr_t = t[idx[nxt]]

    z = -np.asarray(triang, dtype=np.float64)[:, :, 2]     # up-positive, (n,23)
    W = par['prev_frames']
    head = np.empty(n)
    buf = []
    for i in range(n):
        buf.append(z[i])
        if len(buf) > W:
            buf.pop(0)
        head[i] = np.median(np.array(buf), axis=0)[[EARL, EARR]].mean()

    fired = np.zeros(n, bool)
    enter_t = None
    ready = False
    last_trig = -np.inf
    for i in range(n):
        h = head[i]
        if h < par['reset_threshold']:
            ready = True
            enter_t = None
        if not real[i]:                       # check_time=False
            continue
        ct = curr_t[i]
        if ready and par['head_min'] <= h <= par['head_max']:
            if enter_t is None:
                enter_t = ct
            elif (ct - enter_t) >= par['hold_time'] and (ct - last_trig) >= par['refractory']:
                fired[i] = True
                ready = False
                last_trig = ct
        else:
            enter_t = None
    return head, fired, real, curr_t


def task_mask(n, par):
    """Task-context flag per acquisition frame (runtime = cam0_frame / fps)."""
    rt = np.arange(n) / par['fps']
    m = np.zeros(n, bool)
    if par['task_id'] != 2:
        return m if par['task_id'] == 0 else ~m
    state = False
    sw = sorted(float(x) for x in par['switches'])
    k = 0
    for i in range(n):
        while k < len(sw) and rt[i] > sw[k]:
            state = not state
            k += 1
        m[i] = state
    return m


def best_offset(bl, fm, head, max_off=3):
    """Verify the behavior_log -> frame_mapping offset instead of assuming it."""
    out = {}
    cam0 = fm.cam0_frame.to_numpy()
    pf = fm.processed_frame.to_numpy()
    for off in range(0, max_off + 1):
        m = dict(zip(pf, cam0))
        a = bl.frame.add(off).map(m)
        g = a.notna()
        if g.sum() < 100:
            out[off] = np.inf
            continue
        out[off] = float(np.nanmedian(np.abs(
            head[a[g].to_numpy().astype(int)] - bl.head_height.to_numpy()[g.to_numpy()])))
    return out


def main():
    sessions = [('R7', '2026_10_26_1'), ('R8', '2026_10_26_1'), ('R9', '2026_10_26_1')]
    summary, events = [], []
    for rat, session in sessions:
        par, triang, fm, bl = load_session(rat, session)
        head, fired, real, curr_t = replay(triang, fm, par)
        n = len(triang)
        task = task_mask(n, par)

        offs = best_offset(bl, fm, head)
        off = min(offs, key=offs.get)
        cam0_of_log = bl.frame.add(off).map(dict(zip(fm.processed_frame, fm.cam0_frame)))
        good = cam0_of_log.notna()
        acq = cam0_of_log[good].to_numpy().astype(int)
        logged_h = bl.head_height.to_numpy()[good.to_numpy()]
        logged_trig = bl.triggered.to_numpy()[good.to_numpy()].astype(bool)

        # ---- A: signal fidelity ---------------------------------------
        dh = np.abs(head[acq] - logged_h)
        # ---- B: state-machine fidelity --------------------------------
        rep_trig = fired[acq]
        both = int((rep_trig & logged_trig).sum())
        rec = dict(rat=rat, session=session, n_acq=n, n_processed=len(bl),
                   interp_frac=float(1 - real.mean()),
                   log_offset=int(off), offset_err_mm=float(offs[off]),
                   head_med_abs_err_mm=float(np.median(dh)),
                   head_p99_abs_err_mm=float(np.percentile(dh, 99)),
                   head_corr=float(np.corrcoef(head[acq], logged_h)[0, 1]),
                   n_trig_logged=int(logged_trig.sum()),
                   n_trig_replay=int(rep_trig.sum()), n_trig_both=both,
                   task_start_s=float(par['switches'][0]),
                   task_end_s=float(par['switches'][-1]) if len(par['switches']) > 1 else np.nan,
                   n_trig_logged_task=int((logged_trig & task[acq]).sum()),
                   head_min=par['head_min'], reset_threshold=par['reset_threshold'],
                   hold_time=par['hold_time'], refractory=par['refractory'])
        summary.append(rec)
        print(f'{rat}/{session}: offset={off} (err {offs[off]:.4f} mm)  '
              f'head |err| med={rec["head_med_abs_err_mm"]:.4f} p99={rec["head_p99_abs_err_mm"]:.3f}  '
              f'triggers logged={rec["n_trig_logged"]} replay={rec["n_trig_replay"]} '
              f'both={both}  in-task logged={rec["n_trig_logged_task"]}', flush=True)

        np.savez_compressed(f'{OUT}/online_replay_{rat}_{session}.npz',
                            head=head.astype(np.float32), fired=fired, real=real,
                            task=task, acq_of_log=acq, logged_h=logged_h.astype(np.float32),
                            logged_trig=logged_trig, par=json.dumps(par, default=float))
        for a, lg, rp in zip(acq, logged_trig, rep_trig):
            if lg or rp:
                events.append(dict(rat=rat, session=session, acq_frame=int(a),
                                   t_s=a / par['fps'], in_task=bool(task[a]),
                                   logged=bool(lg), replay=bool(rp),
                                   head_logged=float(head[a])))
    pd.DataFrame(summary).to_csv(f'{OUT}/online_replay_summary.csv', index=False)
    pd.DataFrame(events).to_csv(f'{OUT}/online_replay_events.csv', index=False)
    print(f'\nwrote {OUT}/online_replay_summary.csv and _events.csv')


if __name__ == '__main__':
    main()
