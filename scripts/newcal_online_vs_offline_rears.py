"""Do the rears the rig detected ONLINE during the task match the rears an
OFFLINE detector finds?

Builds on scripts/newcal_online_rear_replay.py, which established that the
online HeadHold rule replays exactly from the saved poses (every logged trigger
reproduced, height matching the log to its 3-decimal rounding).  That settles
reproducibility; this script asks the different, harder question of whether
those triggers correspond to real rears, using DANNCE as the independent
reference the rig never sees.

Four event sets per session, all restricted to the task context
(`context_switch_time`, 600-1800 s here):

  online        the rig's own `triggered` rows (behavior_log.csv)
  sleap_ep      rear EPISODES from SLEAP's ear_mean_z, hysteresis run offline
                with the rule's own parameters (rise = head_min, reset =
                reset_threshold, min duration = hold_time)
  dannce_ep     the same detector on DANNCE's ear_mean_z, Procrustes-aligned
                into SLEAP's world frame -- the ground-truth rear set
  replay        the exact HeadHold replay, kept as a control

HeadHold fires at most ONCE per rear (a latch requires the head to drop below
reset_threshold before it can fire again), so an online trigger is matched to an
episode by containment: the trigger frame falls inside the episode, with 300 ms
of slack at each end.

A DANNCE rear with no online trigger is NOT automatically a failure -- several
causes are the rule behaving as designed.  Misses are classified, in precedence
order:
  refractory    a trigger fired less than `refractory` before the episode could
                have fired
  not_rearmed   SLEAP's height never dropped below reset_threshold between the
                previous trigger and this episode, so the latch was still shut
  sleap_short   SLEAP's height did reach head_min inside the episode but never
                for hold_time continuously
  sleap_low     SLEAP's height never reached head_min inside the episode at all
                -- the only category that is a genuine tracking failure
  all_interp    every frame of the episode was interpolated, so no frame was
                eligible to fire

Writes results/newcal/online_vs_offline_{summary,misses}.csv
"""
import os, sys, json
import numpy as np
import pandas as pd

REPO = '/home/yutaka-sprague/CLIRB_analyses'
sys.path.insert(0, REPO); os.chdir(REPO)

import session_cache as sc
from clirb import events as ev
from clirb import features as cf
from clirb.spatial import dual as du
from scripts.newcal_rear_detector import detect_rears_fast
from scripts.newcal_sleap_dannce_eval import robust_proc, balanced_sample

OUT = 'results/newcal'
SLACK = 6            # +/-300 ms at 20 fps
SESSIONS = [('R7', '2026_10_26_1'), ('R8', '2026_10_26_1'), ('R9', '2026_10_26_1')]


def dannce_ear_z(rat, session, n):
    """DANNCE ear_mean_z on the SLEAP acquisition grid, up-positive mm."""
    raw = sc.get_raw_session(rat, session, want_dual=True)
    d3 = du.dannce_aligned(raw)
    m = min(len(raw['triang_3d']), len(d3), n)
    sl = np.asarray(raw['triang_3d'][:m], np.float32)
    dn = np.asarray(d3[:m], np.float32)
    samp, _ = balanced_sample(sl.mean(axis=1)[:, :2], rng=np.random.default_rng(0))
    fit = robust_proc(dn[samp], sl[samp])
    D = ev.causal_median_5_kp(fit['apply'](dn), warmup='online').copy()
    D[:, :, 2] *= -1.0
    z = cf.compute_features(D)['ear_mean_z']
    out = np.full(n, np.nan)
    out[:m] = z
    return out, float(fit['residual'])


def in_window(ep, lo, hi):
    return (ep[:, 1] >= lo) & (ep[:, 0] <= hi)


def main():
    rows, misses = [], []
    for rat, session in SESSIONS:
        z = np.load(f'{OUT}/online_replay_{rat}_{session}.npz', allow_pickle=True)
        par = json.loads(str(z['par']))
        head = z['head']; real = z['real']; task = z['task']
        acq = z['acq_of_log']; logged = z['logged_trig']
        n = len(head)
        fps = par['fps']
        rise, reset = par['head_min'], par['reset_threshold']
        mindur = int(round(par['hold_time'] * fps))

        dz, resid = dannce_ear_z(rat, session, n)

        sl_ep = detect_rears_fast(head, rise, reset, mindur)
        ok = np.isfinite(dz)
        dz_f = np.where(ok, dz, -1e3)
        dn_ep = detect_rears_fast(dz_f, rise, reset, mindur)

        t_idx = np.where(task)[0]
        lo, hi = (t_idx.min(), t_idx.max()) if len(t_idx) else (0, -1)
        trig = acq[logged]
        trig_task = trig[(trig >= lo) & (trig <= hi)]
        sl_t = sl_ep[in_window(sl_ep, lo, hi)] if len(sl_ep) else sl_ep
        dn_t = dn_ep[in_window(dn_ep, lo, hi)] if len(dn_ep) else dn_ep

        def contains(eps, f):
            return np.where((eps[:, 0] - SLACK <= f) & (f <= eps[:, 1] + SLACK))[0] if len(eps) else np.array([], int)

        # online trigger -> real rear?
        hit_dn = np.array([len(contains(dn_t, f)) > 0 for f in trig_task], bool)
        hit_sl = np.array([len(contains(sl_t, f)) > 0 for f in trig_task], bool)
        # DANNCE rear -> online trigger?
        ep_trig = np.array([len(contains(trig_task[:, None].ravel()[:, None].T.ravel()
                                         if False else trig_task, 0)) for _ in []], int)
        got = np.zeros(len(dn_t), bool)
        for k, (a, b, _) in enumerate(dn_t):
            got[k] = np.any((trig_task >= a - SLACK) & (trig_task <= b + SLACK))

        # classify the misses
        all_trig = np.sort(acq[logged])
        for k, (a, b, _) in enumerate(dn_t):
            if got[k]:
                continue
            seg = head[a:b + 1]
            segreal = real[a:b + 1]
            prev = all_trig[all_trig < a]
            prev_t = prev.max() if len(prev) else -10 ** 9
            above = seg >= rise
            runs = np.diff(np.concatenate([[0], above.view(np.int8), [0]]))
            longest = 0
            if above.any():
                st = np.where(runs == 1)[0]; en = np.where(runs == -1)[0]
                longest = int((en - st).max())
            if not segreal.any():
                why = 'all_interp'
            elif (a - prev_t) / fps < par['refractory']:
                why = 'refractory'
            elif len(prev) and head[prev_t:a].min() >= reset:
                why = 'not_rearmed'
            elif not above.any():
                why = 'sleap_low'
            elif longest < mindur:
                why = 'sleap_short'
            else:
                why = 'unexplained'
            misses.append(dict(rat=rat, session=session, start=int(a), end=int(b),
                               t_s=a / fps, dur_s=(b - a + 1) / fps,
                               dannce_peak=float(dz[a:b + 1].max()),
                               sleap_peak=float(seg.max()),
                               sleap_longest_above_s=longest / fps,
                               s_since_prev_trigger=(a - prev_t) / fps if len(prev) else np.nan,
                               reason=why))

        # latency of the trigger relative to the DANNCE rear onset
        lat = []
        for f in trig_task:
            j = contains(dn_t, f)
            if len(j):
                lat.append((f - dn_t[j[0], 0]) / fps)
        rows.append(dict(
            rat=rat, session=session, procrustes_residual_mm=resid,
            task_lo_s=lo / fps, task_hi_s=hi / fps,
            n_online_task=len(trig_task), n_online_total=int(logged.sum()),
            n_sleap_ep_task=len(sl_t), n_dannce_ep_task=len(dn_t),
            online_in_dannce_rear=float(hit_dn.mean()) if len(trig_task) else np.nan,
            online_in_sleap_rear=float(hit_sl.mean()) if len(trig_task) else np.nan,
            dannce_rear_triggered=float(got.mean()) if len(dn_t) else np.nan,
            trigger_latency_med_s=float(np.median(lat)) if lat else np.nan,
            trigger_latency_p90_s=float(np.percentile(lat, 90)) if lat else np.nan,
            interp_frac_task=float(1 - real[lo:hi + 1].mean()) if hi >= lo else np.nan,
        ))
        print(f'{rat}: online(task)={len(trig_task)} sleap_ep={len(sl_t)} '
              f'dannce_ep={len(dn_t)} | online inside a DANNCE rear='
              f'{100*hit_dn.mean() if len(trig_task) else float("nan"):.1f}% | '
              f'DANNCE rears triggered={100*got.mean() if len(dn_t) else float("nan"):.1f}%',
              flush=True)

    pd.DataFrame(rows).to_csv(f'{OUT}/online_vs_offline_summary.csv', index=False)
    m = pd.DataFrame(misses)
    m.to_csv(f'{OUT}/online_vs_offline_misses.csv', index=False)
    if len(m):
        print('\nmiss reasons (task window):')
        print(m.groupby(['rat', 'reason']).size().to_string())
    print(f'\nwrote {OUT}/online_vs_offline_summary.csv and _misses.csv')


if __name__ == '__main__':
    main()
