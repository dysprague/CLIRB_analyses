"""Replay the online reward rule over a recorded session.

One entry point, `replay_session`, so notebooks and scripts cannot drift apart
on which gates they apply or in what order. Everything it needs comes from the
files campy archives into the session folder -- nothing is hand-typed.

The offline pose pipeline here is bit-identical to campy/behavior_rule.py as of
the 5-frame-median fix; `reconcile` asserts that against the rig's own log.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_io as dio  # noqa: E402
from clirb import events as ev  # noqa: E402


def replay_session(rat, session, flip_z=True, triang=None):
    """Load a session and run the online rule over it.

    Returns dict:
      params        : OnlineParams from the session's campy_config.yaml
      template      : the template npz the session ran with
      pcs           : (T, n_pcs) online-equivalent projection
      is_real       : (T,) False on interpolated fills
      acq_frame     : (T,) camera acquisition times
      port_dist     : (T,) trailing-mean COM distance to the port (or None)
      result        : replay_online_rule output
      online_reward : camera frames where the rig actually delivered water
    """
    params = dio.load_session_config(rat, session)
    template = dio.load_session_template(rat, session)
    if triang is None:
        triang = dio.load_sleap_keys_3d(rat, session)
    n_frames = triang.shape[0]

    pcs, _ = ev.project_session_to_pcs(triang, template, flip_z=flip_z)
    align = dio.load_frame_alignment(rat, session, n_frames)

    port_dist = None
    if params.min_port_distance is not None:
        try:
            port = dio.load_water_port(rat, session)
            port_dist = ev.online_port_distance(
                triang, port, window=int(template['template'].shape[0]))
        except FileNotFoundError:
            print(f"warning: {rat}/{session} has min_port_distance="
                  f"{params.min_port_distance} but no water_port_location.npy; "
                  "port gate NOT applied")

    result = ev.replay_online_rule(
        pcs, template, params,
        is_real=align['is_real'], acq_times=align['acq_frame'],
        port_dist=port_dist)

    try:
        online = dio.load_online_rewards(rat, session, n_frames)
        online_reward = online['camera_frame'].to_numpy(dtype=int)
        logged_pcs = online.attrs['logged_pcs']
    except (FileNotFoundError, KeyError):
        online_reward, logged_pcs = None, None

    return {'rat': rat, 'session': session, 'params': params,
            'template': template, 'pcs': pcs, 'triang': triang,
            'is_real': align['is_real'], 'acq_frame': align['acq_frame'],
            'port_dist': port_dist, 'result': result,
            'online_reward': online_reward, 'logged_pcs': logged_pcs,
            'n_frames': n_frames}


def reconcile(rep, pc_tol=0.01, verbose=True):
    """Check the replay against what the rig logged.

    For a session recorded with the CURRENT campy code, pc_max_diff should sit
    at the behavior_log CSV's 3-decimal rounding floor and reward_frames_match
    should be True. Anything larger means the offline pose pipeline has drifted
    from campy/behavior_rule.py again -- that is the regression this guards.

    For a session recorded BEFORE the 5-frame-median fix, a large pc diff and
    extra replay rewards are expected: the replay implements the fixed rule, so
    the difference IS the improvement the fix buys. Warm-up rows (before the
    rule's median buffer fills) are excluded either way.
    """
    out = {'rat': rep['rat'], 'session': rep['session']}
    logged = rep['logged_pcs']
    if logged is None:
        out['pc_max_diff'] = np.nan
    else:
        cf = logged['camera_frame'].to_numpy(dtype=int)
        keep = cf >= 5                       # rule's median is unfilled before this
        lp = logged[['PC1', 'PC2']].to_numpy(dtype=float)[keep]
        d = np.abs(lp - rep['pcs'][cf[keep], :2])
        out['pc_max_diff'] = float(d.max())
        out['pc_mean_diff'] = float(d.mean())

    replay_reward = rep['result']['reward']
    online = rep['online_reward']
    out['n_replay_reward'] = int(len(replay_reward))
    out['n_online_reward'] = None if online is None else int(len(online))
    if online is not None:
        out['reward_frames_match'] = bool(
            len(online) == len(replay_reward)
            and np.array_equal(np.sort(online), np.sort(replay_reward)))
        out['only_replay'] = np.setdiff1d(replay_reward, online).tolist()
        out['only_online'] = np.setdiff1d(online, replay_reward).tolist()

    if verbose:
        print(f"{rep['rat']}/{rep['session']}: "
              f"PC max diff {out['pc_max_diff']:.4f}  "
              f"rewards replay={out['n_replay_reward']} "
              f"online={out['n_online_reward']}  "
              f"exact={out.get('reward_frames_match')}")
    return out


def summarize(rep):
    """One-line breakdown of where candidate windows were lost."""
    r = rep['result']
    c = r['counts']
    return {
        'rat': rep['rat'], 'session': rep['session'],
        'n_frames': rep['n_frames'],
        'minutes': rep['n_frames'] / rep['params'].fps / 60.0,
        'interp_frac': float((~rep['is_real']).mean()),
        'passing_windows': int(len(r['passing'])),
        'matches_ungated': int(len(r['matches_ungated'])),
        'lost_interp': c['interpolated'],
        'lost_port': c['port_gate'],
        'lost_refractory': c['refractory'],
        'lost_task': c['task_context'],
        'rewards': int(len(r['reward'])),
        'online_rewards': (None if rep['online_reward'] is None
                           else int(len(rep['online_reward']))),
    }
